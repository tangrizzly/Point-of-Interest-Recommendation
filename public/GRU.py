#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.shared_randomstreams import RandomStreams
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
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


# 输出时：h*x → sigmoid(T.sum(h*(xp-xq), axis=1))
# 预测时：h*x → np.dot(h, x.T)
# ======================================================================================================================
class GruBasic(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        """
        构建 模型参数
        :param train: 添加mask后的
        :param test: 添加mask后的
        :param n_user: 用户的真实数目
        :param n_item: 商品items的真正数目，init()里补全一个商品作为填充符
        :param n_in: rnn输入向量的维度
        :param n_hidden: rnn隐层向量的维度
        :return:
        """
        # 来自于theano官网的dAE部分。
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
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        ui = uniform(-rang, rang, (3, n_hidden, n_in))
        wh = uniform(-rang, rang, (3, n_hidden, n_hidden))
        h0 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bi = np.zeros((3, n_hidden), dtype=theano.config.floatX)
        # 建立参数。
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.h0 = theano.shared(borrow=True, value=h0)
        self.bi = theano.shared(borrow=True, value=bi)
        # 存放训练好的users、items表达。用于计算所有users对所有items的评分：users * items
        trained_items = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_users = uniform(-rang, rang, (n_user, n_hidden))
        self.trained_items = theano.shared(borrow=True, value=trained_items.astype(theano.config.floatX))
        self.trained_users = theano.shared(borrow=True, value=trained_users.astype(theano.config.floatX))
        # 内建predict函数。不要写在这里，写在子类里，否则子类里会无法覆盖掉重写。
        # self.__theano_predict__(n_in, n_hidden)

    def update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks):
        # 每个epoch都更新负样本
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def update_trained_items(self):
        # 更新最终的items表达
        lt = self.lt.get_value(borrow=True)    # self.lt是shared，用get_value()。
        self.trained_items.set_value(np.asarray(lt, dtype=theano.config.floatX), borrow=True)     # update

    def update_trained_users(self, all_hus):
        # 外部先计算好，传进来直接更新
        self.trained_users.set_value(np.asarray(all_hus, dtype=theano.config.floatX), borrow=True)  # update

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
        # 亲测表明：在序列前做data的corruption，效果更好更稳定。
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

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # 隐层是1个GRU Unit：都可以用这个统一的格式。
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return h_t
        h, _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
            outputs_info=h0,
            n_steps=seq_length)

        # 得到batch_hts.shape=(n, 20)，就是这个batch里每个用户的表达ht。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def predict(self, idxs):
        return self.seq_predict(idxs)


# 输出时：h*x → sigmoid(T.sum([hx, hm]*([xp, mp] - [xq, mq])))
# 预测是：h*x → np.dot([hx, hm], [x, m].T)
# ======================================================================================================================
class GruBasic2Units(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(GruBasic2Units, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        mm = uniform(-rang, rang, (n_item + 1, n_in))   # 多模态融合特征，和 lt 一一对应。
        uix = uniform(-rang, rang, (3, n_hidden, n_hidden))
        uim = uniform(-rang, rang, (3, n_hidden, n_hidden))
        whx = uniform(-rang, rang, (3, n_hidden, n_hidden))
        whm = uniform(-rang, rang, (3, n_hidden, n_hidden))
        h0x = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        h0m = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bix = np.zeros((3, n_hidden), dtype=theano.config.floatX)
        bim = np.zeros((3, n_hidden), dtype=theano.config.floatX)
        # 建立参数。
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.mm = theano.shared(borrow=True, value=mm.astype(theano.config.floatX))
        self.uix = theano.shared(borrow=True, value=uix.astype(theano.config.floatX))
        self.uim = theano.shared(borrow=True, value=uim.astype(theano.config.floatX))
        self.whx = theano.shared(borrow=True, value=whx.astype(theano.config.floatX))
        self.whm = theano.shared(borrow=True, value=whm.astype(theano.config.floatX))
        self.h0x = theano.shared(borrow=True, value=h0x)
        self.h0m = theano.shared(borrow=True, value=h0m)
        self.bix = theano.shared(borrow=True, value=bix)
        self.bim = theano.shared(borrow=True, value=bim)
        # 内建predict函数。不要写在这里，写在子类里，否则会无法继承、覆盖掉重写。
        # self.__theano_predict__(n_in, n_hidden)

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        uix, whx = self.uix, self.whx
        uim, whm = self.uim, self.whm

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)
        h0m = T.alloc(self.h0m, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 3, n_hidden)     # shape=(n, 3, 20), 原维度放在后边
        bim = T.alloc(self.bim, actual_batch_size, 3, n_hidden)     # shape=(n, 3, 20), 原维度放在后边
        bix = bix.dimshuffle(1, 2, 0)                               # shape=(3, 20, n)
        bim = bim.dimshuffle(1, 2, 0)                               # shape=(3, 20, n)

        # 隐层是2个GRU Units：当然也有2部分输入。
        pidxs = T.imatrix()
        xps, mps = self.lt[pidxs], self.mm[pidxs]                   # shape((actual_batch_size, seq_length, n_in))
        xps, mps = xps.dimshuffle(1, 0, 2), mps.dimshuffle(1, 0, 2) # shape=(seq_length, batch_size, n_in)

        def recurrence(xp_t, mp_t, hx_t_pre1, hm_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # 隐层计算
            z_rx = sigmoid(T.dot(uix[:2], xp_t.T) + T.dot(whx[:2], hx_t_pre1.T) + bix[:2])    # shape=(2, 20, n)
            z_rm = sigmoid(T.dot(uim[:2], mp_t.T) + T.dot(whm[:2], hm_t_pre1.T) + bim[:2])    # shape=(2, 20, n)
            zx, rx = z_rx[0].T, z_rx[1].T                   # shape=(n, 20)
            zm, rm = z_rm[0].T, z_rm[1].T                   # shape=(n, 20)
            cx = tanh(T.dot(uix[2], xp_t.T) + T.dot(whx[2], (rx * hx_t_pre1).T) + bix[2])    # shape=(20, n)
            cm = tanh(T.dot(uim[2], mp_t.T) + T.dot(whm[2], (rm * hm_t_pre1).T) + bim[2])    # shape=(20, n)
            hx_t = (T.ones_like(zx) - zx) * hx_t_pre1 + zx * cx.T        # shape=(n, 20)
            hm_t = (T.ones_like(zm) - zm) * hm_t_pre1 + zm * cm.T        # shape=(n, 20)
            return [hx_t, hm_t]
        [hx, hm], _ = theano.scan(              # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=[xps, mps],
            outputs_info=[h0x, h0m],
            n_steps=seq_length)
        h = T.concatenate((hx, hm), axis=2)     # h.shape=(157, n, 40)

        # 得到batch_hus.shape=(n, 20)，就是这个batch里每个用户的表达hu。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})


# 'Obo' is one by one. 逐条训练。
# ======================================================================================================================
class OboGru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OboGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)                # 有效长度

        h0 = self.h0
        bi = self.bi

        pidxs, qidxs = T.ivector(), T.ivector()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((seq_length, n_in))

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, xq_t, h_t_pre1):
            z_r = sigmoid(T.dot(ui[:2], xp_t) +
                          T.dot(wh[:2], h_t_pre1) + bi[:2])
            z, r = z_r[0], z_r[1]
            c = tanh(T.dot(ui[2], xp_t) +
                     T.dot(wh[2], (r * h_t_pre1)) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
            upq_t = T.dot(h_t_pre1, xp_t - xq_t)
            loss_t = T.log(sigmoid(upq_t))
            return [h_t, loss_t]
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs],
            outputs_info=[h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi]])
        upq = T.sum(loss)
        seq_costs = (
            - upq +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        uidx = T.iscalar()                              # T.iscalar()类型是 TensorType(int32, )
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def train(self, idx):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        return self.seq_train(idx)


# 只是train相关的要做成mini-batch形式，其它的都和 Gru/MvGru 是一样的。
# 要做梯度归一化,即算出来的梯度除以batch size. 不除T.sum(tra_mask)，除以batch_size.
# ======================================================================================================================
class Gru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Gru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, xq_t, mask_t, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])   # shape=(2, 20, n)
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (xp_t - xq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # 只在损失这里乘一下0/1向量就可以了
            return [h_t, loss_t]                        # shape=(n, 20), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)       # 10/30 * l2，不行。
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

        # 这个要不要？
        self.normalize = theano.function(
            inputs=[],
            updates={
                self.lt: self.lt / T.sqrt(T.sum(self.lt ** 2, axis=1).dimshuffle(0, 'x'))
            })

    def train(self, idxs):
        return self.seq_train(idxs)


# ======================================================================================================================
class Lstm(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Lstm, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        ui = uniform(-rang, rang, (4, n_hidden, n_hidden))
        wh = uniform(-rang, rang, (4, n_hidden, n_hidden))
        c0 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bi = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        # 建立参数。
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.c0 = theano.shared(borrow=True, value=c0)
        self.bi = theano.shared(borrow=True, value=bi)
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        c0 = T.alloc(self.c0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, xq_t, mask_t, c_t_pre1, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gates = T.dot(ui, xp_t.T) + T.dot(wh, h_t_pre1.T) + bi  # shape=(4, 20, n)
            i, f, g, o = sigmoid(gates[0]).T, sigmoid(gates[1]).T, tanh(gates[2]).T, sigmoid(gates[3]).T
            c_t = f * c_t_pre1 + i * g
            h_t = o * tanh(c_t)   # shape=(n, 20)
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (xp_t - xq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # 只在损失这里乘一下0/1向量就可以了
            return [c_t, h_t, loss_t]                        # shape=(n, 20), (n, )
        [c, h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, mask],
            outputs_info=[c0, h0, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        c0 = T.alloc(self.c0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # 隐层是1个GRU Unit：都可以用这个统一的格式。
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, c_t_pre1, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gates = T.dot(ui, p_t.T) + T.dot(wh, h_t_pre1.T) + bi  # shape=(4, 20, n)
            i, f, g, o = sigmoid(gates[0]).T, sigmoid(gates[1]).T, tanh(gates[2]).T, sigmoid(gates[3]).T
            c_t = f * c_t_pre1 + i * g
            h_t = o * tanh(c_t)   # shape=(n, 20)
            return [c_t, h_t]
        [c, h], _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
            outputs_info=[c0, h0],
            n_steps=seq_length)

        # 得到batch_hts.shape=(n, 20)，就是这个batch里每个用户的表达ht。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})


# ======================================================================================================================
class Rnn(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Rnn, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        ui = uniform(-rang, rang, (n_hidden, n_hidden))
        wh = uniform(-rang, rang, (n_hidden, n_hidden))
        bi = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        # 建立参数。
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.bi = theano.shared(borrow=True, value=bi)
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, n_hidden)   # shape=(n, 20), n_hidden放在最后
        bi = bi.T   # shape=(20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_length, batch_size, n_in)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, xq_t, mask_t, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            h_t = sigmoid(T.dot(ui, xp_t.T) +
                          T.dot(wh, h_t_pre1.T) + bi)     # shape=(n, 20), 计算当前隐层值
            h_t = h_t.T
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (xp_t - xq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                            # 只在损失这里乘一下0/1向量就可以了
            return [h_t, loss_t]                        # shape=(n, 20), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()     # int32
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.T   # shape=(20, n)

        # 隐层是1个GRU Unit：都可以用这个统一的格式。
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, h_t_pre1):
            h_t = sigmoid(T.dot(ui, p_t.T) +
                          T.dot(wh, h_t_pre1.T) + bi)   # shape=(20, n)
            h_t = h_t.T                                     # shape=(n, 20)。这里注意
            return h_t
        h, _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
            outputs_info=h0,
            n_steps=seq_length)

        # 得到batch_hts.shape=(n, 20)，就是这个batch里每个用户的表达ht。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the class: GRU')


if '__main__' == __name__:
    main()

