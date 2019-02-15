#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.extra_ops import Unique
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


# ======================================================================================================================
class MfBasic(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        """
        构建 模型参数
        :param n_item: 商品items的数目
        :param n_in: rnn输入向量的维度
        :return:
        """
        rng = np.random.RandomState(123)
        self.thea_rng = RandomStreams(rng.randint(2 ** 30))     # 旗下随机函数可在GPU下运行。
        # 用mask进行补全后的train/test
        tra_buys_masks, tra_masks, tra_buys_neg_masks = train
        tes_buys_masks, tes_masks, tes_buys_neg_masks = test
        self.tra_buys_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_masks, dtype='int32'))
        self.tes_buys_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_masks, dtype='int32'))
        self.tra_masks = theano.shared(borrow=True, value=np.asarray(tra_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.tra_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_neg_masks, dtype='int32'))
        self.tes_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_neg_masks, dtype='int32'))
        # 把超参数shared
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        ux = uniform(-rang, rang, (n_user, n_in))
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # shape=(n_item, 20)
        self.ux = theano.shared(borrow=True, value=ux.astype(theano.config.floatX))
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        # 存放训练好的users、items表达。用于计算所有users对所有items的评分：users * items
        trained_items = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_users = uniform(-rang, rang, (n_user, n_hidden))
        self.trained_items = theano.shared(borrow=True, value=trained_items.astype(theano.config.floatX))
        self.trained_users = theano.shared(borrow=True, value=trained_users.astype(theano.config.floatX))

    def update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks):
        # 每个epoch都更新负样本
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def update_trained_items(self):
        # 更新最终的items表达
        lt = self.lt.get_value(borrow=True)    # self.lt是shared，用get_value()。
        self.trained_items.set_value(np.asarray(lt, dtype=theano.config.floatX), borrow=True)     # update

    def update_trained_users(self):
        # 更新最终的users表达
        ux = self.ux.get_value(borrow=True)
        self.trained_users.set_value(np.asarray(ux, dtype=theano.config.floatX), borrow=True)  # update

    def compute_sub_all_scores(self, start_end):    # 其实可以直接传过来实数参数
        # 计算users * items，每个用户对所有商品的评分(需去掉填充符)
        sub_all_scores = T.dot(self.trained_users[start_end], self.trained_items[:-1].T)
        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def compute_sub_auc_preference(self, start_end):
        # items.shape=(n_item+1, 20)，因为是mask形式，所以需要填充符。
        # 注意矩阵形式的索引方式。
        tes_items = self.trained_items[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_items_neg = self.trained_items[self.tes_buys_neg_masks[start_end]]
        users = self.trained_users[start_end]
        shp0, shp2 = users.shape        # shape=(sub_n_user, n_hidden)
        # 利用性质：(n_user, 1, n_hidden) * (n_user, len, n_hidden) = (n_user, len, n_hidden)，即broadcast
        # 利用性质：np.sum((n_user, len, n_hidden), axis=2) = (n_user, len)，
        #         即得到各用户对test里正负样本的偏好值
        all_upqs = T.sum(users.reshape((shp0, 1, shp2)) * (tes_items - tes_items_neg), axis=2)
        all_upqs *= self.tes_masks[start_end]       # 只保留原先test items对应有效位置的偏好值
        return all_upqs.eval() > 0                  # 将>0的标为True, 也就是1

    def get_corrupted_input_whole(self, inp, corruption_prob):
        # 处理2D矩阵：randomly set whole feature to zero. Matrix.shape=(n, m)
        # denoising方式0：随机将某些图、文特征整体性置为0
        # 比如原先一条序列的图像特征是(num, 1024); 那么0/1概率矩阵是(num, 1), T.Rebroadcast，再相乘
        # if corruption_prob < 0. or corruption_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        retain_prob = 1. - corruption_prob
        randoms = self.thea_rng.binomial(
            size=(inp.shape[0], 1),     # shape=(num, 1)
            n=1,
            p=retain_prob,             # p是得1的概率。
            dtype=theano.config.floatX)
        randoms = T.Rebroadcast((1, True))(randoms)
        return inp * randoms            # shape=(num, 1024)

    def get_corrupted_input_whole_minibatch(self, inp, corruption_prob):
        # 处理3D矩阵
        retain_prob = 1. - corruption_prob
        randoms = self.thea_rng.binomial(
            size=(inp.shape[0], inp.shape[1], 1),     # shape=(seq_length, batch_size, 1)
            n=1,
            p=retain_prob,             # p是得1的概率。
            dtype=theano.config.floatX)
        randoms = T.Rebroadcast((2, True))(randoms)
        return inp * randoms            # shape=(seq_length, batch_size, 1024)

    def dropout(self, inp, drop_prob):
        # 处理向量：randomly set some positions to zero. Vector.shape=(n, )
        # 例如一个向量20维，就有20个位置，也就是有20个神经元。
        # train时做dropout，test时还是全连接。
        # if drop_prob < 0. or drop_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        retain_prob = 1. - drop_prob      # 取0.5就可以了。
        randoms = self.thea_rng.binomial(
            size=inp.shape,     # 生成与向量inp同样维度的0、1向量
            n=1,                # 每个神经元实验一次
            p=retain_prob)     # 每个神经元*1的概率为p/retain_prob。*0的概率为drop_prob
        inp *= randoms          # 屏蔽某些神经元，重置为0
        inp /= retain_prob     # drop完后需要rescale，以维持inp在dropout前后的数值和(sum)不变。
        return inp              # 直接本地修改inp，所以调用时'self.dropout(x, 0.5)'即可直接本地修改输入x。


# 不再用
# ======================================================================================================================
class MFonebyone(MfBasic):
    def __init__(self, n_user, n_item, n_in):
        super(MFonebyone, self).__init__(n_user, n_item, n_in)
        self.params = [self.ur, self.lt]
        self.L2_sqr = (
            T.sum(self.ur ** 2) +
            T.sum(self.lt ** 2))
        self.__theano_train__()

    def __theano_train__(self, ):
        """
        训练阶段跑一遍训练序列
        """
        uidx, pidx = T.iscalar(), T.iscalar()
        us = self.ur[uidx]     # shape=(n_in, )
        xp = self.lt[pidx]

        """
        输入t时刻正样本，计算用户表达 * 正样本的得分，与目标分数做差，算平方. 公式里省略了时刻t
        # 根据性质：T.dot((n, ), (n, ))得到(1, 1)
        """
        pre = T.dot(us, xp)
        err = 5.0 - pre
        loss = err * err

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, L2_reg = T.scalar(), T.scalar()
        bpr_L2_sqr = (
            T.sum(us ** 2) +
            T.sum(xp ** 2))
        costs = (
            loss +
            0.5 * L2_reg * bpr_L2_sqr)
        # SGD
        update_us = T.set_subtensor(us, us - lr * T.grad(costs, us))    # sub_tensor 求导更新
        update_xp = T.set_subtensor(xp, xp - lr * T.grad(costs, xp))

        # ----------------------------------------------------------------------------
        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.train = theano.function(
            inputs=[uidx, pidx, lr, L2_reg],
            outputs=loss,
            updates=[(self.ur, update_us),     # 把 sub_tensor 的更新返回到原参数里
                     (self.lt, update_xp)])

    def train(self, u_idx, p_idx, lr, L2_reg):
        # 某用户的某次购买
        return self.train(u_idx, p_idx, lr, L2_reg)


# ======================================================================================================================
class OboBpr(MfBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OboBpr, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ux, self.lt]    # 两个都是单独进行更新。
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()

    def __theano_train__(self, ):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        uidx, pqidx = T.iscalar(), T.ivector()
        usr = self.ux[uidx]     # shape=(n_in, )
        xpq = self.lt[pqidx]

        """
        输入t时刻正负样本，计算当前损失并更新user/正负样本. 公式里省略了时刻t
        # 根据性质：T.dot((n, ), (n, ))得到(1, 1)
            uij  = user * (xp - xq)
            upq = log(sigmoid(uij))
        """
        uij = T.dot(usr, xpq[0] - xpq[1])
        upq = T.log(sigmoid(uij))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [usr, xpq]]))
        costs = (
            - upq +
            0.5 * l2 * bpr_l2_sqr)
        # 1个user，2个items，这种更新求导是最快的。
        pars_subs = [(self.ux, usr), (self.lt, xpq)]
        seq_updates = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, sub)))
                       for par, sub in pars_subs]
        # ----------------------------------------------------------------------------

        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.bpr_train = theano.function(
            inputs=[uidx, pqidx],
            outputs=-upq,
            updates=seq_updates)

    def train(self, u_idx, pq_idx):
        # 某用户的某次购买
        return self.bpr_train(u_idx, pq_idx)


# ======================================================================================================================
class OboVBpr(MfBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, fea_img):
        super(OboVBpr, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        # 初始化参数
        rang = 0.5
        mi = uniform(-rang, rang, (n_item + 1, n_in))   # 低维的多模态融合特征，和 lt 一一对应。
        ue = uniform(-rang, rang, (n_user, n_in))
        ei = uniform(-rang, rang, (n_in, n_img))       # image, shape=(20, 1024)
        self.mi = theano.shared(borrow=True, value=mi.astype(theano.config.floatX))
        self.ue = theano.shared(borrow=True, value=ue.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.params = [self.ux, self.lt, self.ue, self.ei]
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params[:3]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[3:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__()

    def __theano_train__(self):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev', 'fea_random_zero']
        ei = self.ei

        uidx, pqidx = T.iscalar(), T.ivector()
        usr = self.ux[uidx]     # shape=(n_in, )
        xpq = self.lt[pqidx]
        use = self.ue[uidx]
        ipq = self.fi[pqidx]

        """
        输入t时刻正负样本，计算当前损失并更新user/正负样本. 公式里省略了时刻t
        # 根据性质：T.dot((n, ), (n, ))得到(1, 1)
            uij  = user * (xp - xq)
            upq = log(sigmoid(uij))
        """
        uij = (
            T.dot(usr, xpq[0] - xpq[1]) +
            T.dot(use, T.dot(ei, ipq[0] - ipq[1])))
        upq = T.log(sigmoid(uij))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [usr, xpq, use]]))
        bpr_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei]]))
        costs = (
            - upq +
            0.5 * l2 * bpr_l2_sqr +
            0.5 * l2_ev * bpr_l2_ev)
        pars_subs = [(self.ux, usr), (self.lt, xpq), (self.ue, use)]    # 这个的前提是sub得直接用于loss计算。
        seq_updates = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, sub)))
                       for par, sub in pars_subs]
        pars_alls = [self.ei]
        seq_updates.extend([(par, par - lr * T.grad(costs, par)) for par in pars_alls])
        # ----------------------------------------------------------------------------

        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.bpr_train = theano.function(
            inputs=[uidx, pqidx],
            outputs=-upq,
            updates=seq_updates)

    def train(self, u_idx, pq_idx):
        # 某用户的某次购买
        return self.bpr_train(u_idx, pq_idx)

    def update_trained_items(self):
        # 获取低维的图像表达
        mi = T.dot(self.fi, self.ei.T)     # shape=(n, 20)
        mi = mi.eval()
        self.mi.set_value(np.asarray(mi, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mi), axis=1)   # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_users(self):
        # 更新最终的users表达
        users = T.concatenate((self.ux, self.ue), axis=1)   # shape=(n, 40)
        users = users.eval()
        self.trained_users.set_value(np.asarray(users, dtype=theano.config.floatX), borrow=True)


# 不用mini-batch了，速度太慢。
# 因为不像RNN那样，多条序列全部算完后用BPTT更新一次就行，BPR是每次购买都要更新，结果是self.lt的求导次数瞬间暴涨。
# ======================================================================================================================
class Bpr(MfBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Bpr, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ux, self.lt]    # 两个都是单独进行更新。
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()

    def __theano_train__(self):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        pidxs_t, qidxs_t = T.ivector(), T.ivector()
        mask_t, uidxs = T.ivector(), T.ivector()
        users = self.ux[uidxs]  # shape=(n, 20)
        xps = self.lt[pidxs_t]  # shape=(n, 20)
        xqs = self.lt[qidxs_t]

        pqs = T.concatenate((pidxs_t, qidxs_t))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本，计算当前损失并更新user/正负样本. 公式里省略了时刻t
        # 根据性质：T.dot((n, ), (n, ))得到(1, 1)
            uij  = user * (xp - xq)
            upq = log(sigmoid(uij))
        """
        upq_t = T.sum(users * (xps - xqs), axis=1)
        loss_t = T.log(sigmoid(upq_t))      # shape=(n, )
        loss_t *= mask_t                    # 只在损失这里乘一下0/1向量就可以了

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [users, xps, xqs]]))
        upq = T.sum(loss_t)
        costs = (
            - upq +
            0.5 * l2 * bpr_l2_sqr)
        pars_subs = [(self.ux, users, uidxs), (self.lt, uiq_x, uiq_pqs)]
        bpr_updates = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, par)[idxs]))
                       for par, sub, idxs in pars_subs]
        # ----------------------------------------------------------------------------

        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.bpr_train = theano.function(
            inputs=[pidxs_t, qidxs_t, mask_t, uidxs],
            outputs=-upq,
            updates=bpr_updates)

    def train(self, pidxs_t, qidxs_t, mask_t, uidxs):
        return self.bpr_train(pidxs_t, qidxs_t, mask_t, uidxs)


@exe_time  # 放到待调用函数的定义的上一行
def main():
    pass


if '__main__' == __name__:
    main()
