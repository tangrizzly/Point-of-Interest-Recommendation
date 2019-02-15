#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 13/03/2018 3:30 PM

@author: Tangrizzly
"""
from __future__ import print_function
import time

import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse

from public.Load_Data_prme import cal_dis

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
class PrmeBasic(object):
    def __init__(self, train, test, alpha_lambda, threshold, component_weight, cordi, n_user, n_item, n_size):
        """
        构建 模型参数
        :param train: 添加mask后的
        :param test: 添加mask后的
        :param n_user: 用户的真实数目
        :param n_item: 商品items的真正数目，init()里补全一个商品作为填充符
        :param n_dist: distance的个数
        :param n_size: 维度
        :return:
        """
        self.cordi = theano.shared(borrow=True, value=np.asarray(cordi, dtype=theano.config.floatX))
        self.thd = theano.shared(borrow=True, value=np.asscalar(np.asarray(threshold, dtype='int32')))
        self.cw = theano.shared(borrow=True, value=np.asscalar(np.asarray(component_weight, dtype=theano.config.floatX)))
        self.size = n_size
        # 来自于theano官网的dAE部分。
        rng = np.random.RandomState(123)
        self.thea_rng = RandomStreams(rng.randint(2 ** 30))  # 旗下随机函数可在GPU下运行。
        # 用mask进行补全后的train/test
        tra_pois_masks, tra_all_times, tra_all_dists, tra_masks, tra_pois_neg_masks = train
        tes_pois_masks, tes_all_times, tes_all_dists, tes_masks, tes_pois_neg_masks = test
        self.tra_pois_masks = theano.shared(borrow=True, value=np.asarray(tra_pois_masks, dtype='int32'))
        self.tes_pois_masks = theano.shared(borrow=True, value=np.asarray(tes_pois_masks, dtype='int32'))
        self.tra_all_times = theano.shared(borrow=True, value=np.asarray(tra_all_times, dtype='int32'))
        self.tes_all_times = theano.shared(borrow=True, value=np.asarray(tes_all_times, dtype='int32'))
        self.tra_all_dists = theano.shared(borrow=True, value=np.asarray(tra_all_dists, dtype=theano.config.floatX))
        self.tes_all_dists = theano.shared(borrow=True, value=np.asarray(tes_all_dists, dtype=theano.config.floatX))
        self.tra_masks = theano.shared(borrow=True, value=np.asarray(tra_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.tra_pois_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_pois_neg_masks, dtype='int32'))
        self.tes_pois_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_pois_neg_masks, dtype='int32'))
        # 把超参数shared
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        rang = 0.5

        # 初始化，先定义局部变量，再self.修饰成实例变量
        ds = uniform(-rang, rang, (n_item + 1, n_size))  # sequential transition
        dp = uniform(-rang, rang, (n_item + 1, n_size))  # item representation
        du = uniform(-rang, rang, (n_user, n_size))  # user representation

        # 建立参数。
        self.ds = theano.shared(borrow=True, value=ds.astype(theano.config.floatX))
        self.dp = theano.shared(borrow=True, value=dp.astype(theano.config.floatX))
        self.du = theano.shared(borrow=True, value=du.astype(theano.config.floatX))

        # 存放训练好的users、items表达。用于计算所有users对所有items的评分：users * items
        trained_ds = uniform(-rang, rang, (n_item, n_size))
        trained_dp = uniform(-rang, rang, (n_item, n_size))
        trained_du = uniform(-rang, rang, (n_user, n_size))
        self.trained_ds = theano.shared(borrow=True, value=trained_ds.astype(theano.config.floatX))
        self.trained_dp = theano.shared(borrow=True, value=trained_dp.astype(theano.config.floatX))
        self.trained_du = theano.shared(borrow=True, value=trained_du.astype(theano.config.floatX))
        # 内建predict函数。不要写在这里，写在子类里，否则子类里会无法覆盖掉重写。
        # self.__theano_predict__(n_size, n_size)

    def update_neg_masks(self, tra_pois_neg_masks, tes_pois_neg_masks):
        # 每个epoch都更新负样本
        self.tra_pois_neg_masks.set_value(np.asarray(tra_pois_neg_masks, dtype='int32'), borrow=True)
        self.tes_pois_neg_masks.set_value(np.asarray(tes_pois_neg_masks, dtype='int32'), borrow=True)

    def update_trained_items(self):
        # 更新最终的items表达
        ds = self.ds.get_value(borrow=True)
        self.trained_ds.set_value(np.asarray(ds, dtype=theano.config.floatX), borrow=True)
        dp = self.dp.get_value(borrow=True)
        self.trained_dp.set_value(np.asarray(dp[:-1], dtype=theano.config.floatX), borrow=True)
        du = self.du.get_value(borrow=True)
        self.trained_du.set_value(np.asarray(du, dtype=theano.config.floatX), borrow=True)

    def compute_sub_all_scores(self, start_end):
        shp0 = len(start_end)
        tra_ls = self.tra_pois_masks[start_end, T.sum(self.tra_masks[start_end], axis=1) - 1]
        ls = T.concatenate([tra_ls.reshape((shp0, 1)), self.tes_pois_masks[start_end, :T.max(T.sum(self.tes_masks[start_end], axis=1))-1]], axis=1)
        dsl = self.trained_ds[ls]
        # n_item+1 x letent_size
        du = self.trained_du[start_end]
        dp = self.trained_dp
        ds = self.trained_ds
        _, shp1, shp3 = dsl.shape
        shp2, shp3 = dp.shape

        wl = T.pow(1 + cal_dis(self.cordi[ls][:, :, 0].reshape((shp0, shp1, 1)), self.cordi[ls][:, :, 1].reshape((shp0, shp1, 1)),
                                 self.cordi[:, 0].reshape((1, 1, shp2+1)), self.cordi[:, 1].reshape((1, 1, shp2+1))), 0.25)
        sub_all_scores = - wl[:, :, :-1] * (self.cw * T.sum(T.pow(du.reshape((shp0, 1, shp3)) - dp.reshape((1, shp2, shp3)), 2), axis=2).reshape((shp0, 1, shp2)) +
                                 (1 - self.cw) * T.sum(T.pow((dsl.reshape((shp0, shp1, 1, shp3)) -
                                                              ds[:-1].reshape((1, 1, shp2, shp3))), 2), axis=3))

        # sub_all_scores = - (self.cw * T.sum(T.pow(du.reshape((shp0, 1, shp3)) -
        #                                           dp.reshape((1, shp2, shp3)), 2), axis=2).reshape((shp0, 1, shp2)) +
        #                     (1 - self.cw) * T.sum(T.pow((dsl.reshape((shp0, shp1, 1, shp3)) -
        #                                                  ds[:-1].reshape((1, 1, shp2, shp3))), 2), axis=3))

        return T.reshape(sub_all_scores, (shp0 * shp1, shp2)).eval()

    def compute_sub_auc_preference(self, start_end):
        # ls = self.tra_pois_masks[start_end, T.sum(self.tra_masks[start_end], axis=1) - 1]
        # dsl = self.trained_ds[ls]
        # tes_dp = self.trained_dp[self.tes_pois_masks[start_end]]
        # tes_ds = self.trained_ds[self.tes_pois_masks[start_end]]
        # tes_dp_neg = self.trained_dp[self.tes_pois_neg_masks[start_end]]
        # tes_ds_neg = self.trained_ds[self.tes_pois_neg_masks[start_end]]
        # du = self.trained_du[start_end]
        # shp0, shp2 = du.shape
        # Dp = - self.wls[start_end, self.tes_pois_masks[start_end]] * (self.cw * T.sum((du.reshape((shp0, 1, shp2))
        #                                                - tes_dp) ** 2, axis=2)
        #                               + (1 - self.cw) * T.sum((dsl.reshape((shp0, 1, shp2))
        #                                                        - tes_ds) ** 2, axis=2))
        # Dq = - self.wls[start_end, self.tes_pois_masks[start_end]] * (self.cw * T.sum((du.reshape((shp0, 1, shp2))
        #                                                - tes_dp_neg) ** 2, axis=2)
        #                               + (1 - self.cw) * T.sum((dsl.reshape((shp0, 1, shp2))
        #                                                        - tes_ds_neg) ** 2, axis=2))
        #
        # all_upqs = Dq - Dp
        # all_upqs *= self.tes_masks[start_end]  # 只保留原先test items对应有效位置的偏好值
        # return all_upqs.eval() > 0  # 将>0的标为True, 也就是1
        return np.array([[0 for _ in np.arange(len(self.tes_masks[0].eval()))]])


# 'Obo' is one by one. 逐条训练。
# ======================================================================================================================
class OboPrme(PrmeBasic):
    def __init__(self, train, test, alpha_lambda, threshold, component_weight, cordi, n_user, n_item, n_size):
        super(OboPrme, self).__init__(train, test, alpha_lambda, threshold, component_weight, cordi, n_user,
                                      n_item, n_size)
        self.params = [self.dp, self.ds, self.du]

        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()

    def __theano_train__(self, ):
        """
        训练阶段跑一遍训练序列
        """
        uidx, pqidx = T.iscalar(), T.ivector()
        adidx, tidx = T.dscalar(), T.iscalar()
        du = self.du[uidx]
        dppq = self.dp[pqidx]
        dspq = self.ds[pqidx]

        """
            # PRME
            D_u,lc,c = D^P_u,l if delta(l, lc) > threshold
                     = alpha * D^P_u,l + (1 - alpha) * D^S_lc,l otherwise
        """
        Dp_p = T.sum(T.pow(du - dppq[0], 2))
        Dp_q = T.sum(T.pow(du - dppq[1], 2))
        Ds_p = T.sum(T.pow(dspq[0] - dspq[2], 2))
        Ds_q = T.sum(T.pow(dspq[1] - dspq[2], 2))
        w = T.pow((1 + adidx), 0.25)
        Dp = ifelse(T.gt(tidx, self.thd), Dp_p, w * (self.cw * Dp_p + (1 - self.cw) * Ds_p))
        Dq = ifelse(T.gt(tidx, self.thd), Dp_q, w * (self.cw * Dp_q + (1 - self.cw) * Ds_q))

        upq = T.sum(T.log(sigmoid(- Dp + Dq)))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        bpr_l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [du, dppq, dspq]]))
        costs = (
            upq -
            0.5 * l2 * bpr_l2_sqr)
        # 1个user，2个items，这种更新求导是最快的。
        pars_subs = [(self.du, du), (self.dp, dppq), (self.ds, dspq)]
        seq_updates = [(par, T.set_subtensor(sub, sub + lr * T.grad(costs, sub)))
                       for par, sub in pars_subs]
        # ----------------------------------------------------------------------------

        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.prme_train = theano.function(
            inputs=[uidx, pqidx, adidx, tidx],
            outputs=upq,
            updates=seq_updates)

    def train(self, u_idx, pq_idx, ad_idx, t_idx):
        # 某用户的某次购买
        return self.prme_train(u_idx, pq_idx, ad_idx, t_idx)


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the class: GRU')


if '__main__' == __name__:
    main()
