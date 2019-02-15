#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 2018/10/6 9:33 AM

@author: Tangrizzly
"""

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import exp
from theano.tensor.shared_randomstreams import RandomStreams
# from theano.tensor.nnet.nnet import softmax     # 作用于2d-matrix，按行处理。
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


def softmax(x):
    # 竖直方向取softmax。
    # theano里的是作用于2d-matrix，按行处理。我文中scan里有一步是处理(n,)，直接用会报错，所以要重写。
    # 按axis=0处理(n, )，会超级方便。
    e_x = exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out


# 'Obo' is one by one. 逐条训练。
# ======================================================================================================================
class GeoIE:
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden, ulptai):
        # 来自于theano官网的dAE部分。
        rng = np.random.RandomState(123)
        self.n_hidden = n_hidden
        self.thea_rng = RandomStreams(rng.randint(2 ** 30))  # 旗下随机函数可在GPU下运行。
        # 用mask进行补全后的train/test
        self.ulptai = ulptai
        tra_buys_masks, tra_buys_neg_masks, tra_count, tra_masks = train
        tes_buys_masks, tes_buys_neg_masks = test
        self.tra_masks = theano.shared(borrow=True, value=np.asarray(tra_masks, dtype='int32'))
        self.tra_count = theano.shared(borrow=True, value=np.asarray(tra_count, dtype='int32'))
        self.tra_buys_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_masks, dtype='int32'))
        self.tes_buys_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_masks, dtype='int32'))
        self.tra_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_neg_masks, dtype='int32'))
        self.tes_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_neg_masks, dtype='int32'))
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        rang = 0.5

        g = uniform(-rang, rang, (n_item + 1, n_hidden))  # geo-influence
        h = uniform(-rang, rang, (n_item + 1, n_hidden))  # geo-susceptibility
        t = uniform(-rang, rang, (n_user, n_hidden))      # user preference
        z = uniform(-rang, rang, (n_item + 1, n_hidden))  # poi preference
        self.g = theano.shared(borrow=True, value=g.astype(theano.config.floatX))
        self.h = theano.shared(borrow=True, value=h.astype(theano.config.floatX))
        self.t = theano.shared(borrow=True, value=t.astype(theano.config.floatX))
        self.z = theano.shared(borrow=True, value=z.astype(theano.config.floatX))

        a = uniform(-rang, rang)
        b = uniform(-rang, rang)
        # c = uniform(0, rang)
        self.a = theano.shared(borrow=True, value=a)
        self.b = theano.shared(borrow=True, value=b)
        # self.c = theano.shared(borrow=True, value=c)

        trained_g = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_h = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_t = uniform(-rang, rang, (n_user, n_hidden))
        trained_z = uniform(-rang, rang, (n_item + 1, n_hidden))
        self.trained_g = theano.shared(borrow=True, value=trained_g.astype(theano.config.floatX))
        self.trained_h = theano.shared(borrow=True, value=trained_h.astype(theano.config.floatX))
        self.trained_t = theano.shared(borrow=True, value=trained_t.astype(theano.config.floatX))
        self.trained_z = theano.shared(borrow=True, value=trained_z.astype(theano.config.floatX))

        # params：-----------------------------------------------------------------
        self.params = [self.a, self.b]  # self.g, self.h, self.t, self.z,
        self.l2_sqr = (
            T.sum(self.g ** 2) +
            T.sum(self.h ** 2) +
            T.sum(self.t ** 2) +
            T.sum(self.z ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)

    def f_d(self, d):
        return self.a * (d ** self.b)# * exp(self.c * d)

    def update_trained(self):
        g = self.g.get_value(borrow=True)
        h = self.h.get_value(borrow=True)
        t = self.t.get_value(borrow=True)
        z = self.z.get_value(borrow=True)
        self.trained_g.set_value(np.asarray(g, dtype=theano.config.floatX), borrow=True)
        self.trained_h.set_value(np.asarray(h, dtype=theano.config.floatX), borrow=True)
        self.trained_t.set_value(np.asarray(t, dtype=theano.config.floatX), borrow=True)
        self.trained_z.set_value(np.asarray(z, dtype=theano.config.floatX), borrow=True)

    def compute_sub_auc_preference(self, start_end):
        return [[0] for _ in np.arange(self.tes_buys_masks.eval().shape[0])]

    def compute_sub_all_scores(self, start_end):
        d = self.ulptai[start_end[0]: start_end[-1]]
        n_H = T.sum(self.tra_buys_masks[start_end], 1)  # (32, )
        tz = T.sum(self.trained_t[start_end].reshape((-1, 1, self.n_hidden)) * self.trained_z[:-1].reshape((1, -1, self.n_hidden)), 2)  # (32, 5528)
        gi = self.trained_g[self.tra_buys_masks[start_end]]  # (32 1264 20)
        gi = gi.reshape((gi.shape[0], gi.shape[1], 1, -1)) * self.tra_masks[start_end].reshape((gi.shape[0], gi.shape[1], 1, 1))  # (32, 1264, 1, 20), (32, 1264, 1, 1)
        hj = self.trained_h[:-1].reshape((1, 1, -1, self.n_hidden))  # (1 1 5528 20)
        # gh = T.sum(gi * hj, 3) * self.f_d(self.trained_f, d) / n_H
        gh = T.sum(T.sum(gi * hj, 3), 1) / n_H.reshape((-1, 1))
        s = tz + gh
        return s.eval()

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """

        uidx = T.iscalar()
        msk = T.imatrix()
        dist_pos = T.fmatrix()
        dist_neg = T.fmatrix()

        seq_n, seq_len = msk.shape  # 315 x 315
        tu = self.t[uidx]           # (20, )
        xpidxs = self.tra_buys_masks[uidx]  # (1264, )
        xqidxs = self.tra_buys_neg_masks[uidx]  # (1264, )
        gps = self.g[xpidxs[:seq_len]]  # (315, 20)
        hps, hqs = self.h[xpidxs[1: seq_len + 1]], self.h[xqidxs[1: seq_len + 1]]  # (315, 20)
        zps, zqs = self.z[xpidxs[1: seq_len + 1]], self.z[xqidxs[1: seq_len + 1]]

        guiq_pqs = Unique(False, False, False)(xpidxs)
        uiq_g = self.g[guiq_pqs]

        pqs = T.concatenate((xpidxs, xqidxs))
        uiq_pqs = Unique(False, False, False)(pqs)
        uiq_h = self.h[uiq_pqs]
        uiq_z = self.z[uiq_pqs]

        t_z = T.sum(tu * zps, 1)  # (315, )
        n_h = T.sum(msk, 1)  # (315, )
        expand_g = gps.reshape((1, seq_len, n_hidden)) * msk.reshape((seq_n, seq_len, 1))  # (315, 315, 20)
        sp = T.sum(T.sum(expand_g * hps.reshape((seq_n, 1, n_hidden)), 2) * self.f_d(dist_pos), 1) / n_h + t_z  # [(315, 315) * (315, 315)] -> (315, ) / (315, ) + (315, )
        sq = T.sum(T.sum(expand_g * hqs.reshape((seq_n, 1, n_hidden)), 2) * self.f_d(dist_neg), 1) / n_h + t_z

        # sp = T.sum(T.sum(expand_g * hps.reshape((seq_n, 1, n_hidden)), 2), 1) / n_h + t_z
        # sq = T.sum(T.sum(expand_g * hqs.reshape((seq_n, 1, n_hidden)), 2), 1) / n_h + t_z

        loss = T.sum(T.log(sigmoid(sp - sq)))
        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [gps, hps, hqs, zps, zqs]])
        seq_costs = (
            - loss +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_g = T.set_subtensor(uiq_g, uiq_g - lr * T.grad(seq_costs, self.g)[guiq_pqs])
        update_h = T.set_subtensor(uiq_h, uiq_h - lr * T.grad(seq_costs, self.h)[uiq_pqs])
        update_t = T.set_subtensor(tu, tu - lr * T.grad(seq_costs, self.t)[uidx])
        update_z = T.set_subtensor(uiq_z, uiq_z - lr * T.grad(seq_costs, self.z)[uiq_pqs])
        seq_updates.append((self.g, update_g))
        seq_updates.append((self.h, update_h))
        seq_updates.append((self.t, update_t))
        seq_updates.append((self.z, update_z))
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        self.seq_train = theano.function(
            inputs=[uidx, dist_pos, dist_neg, msk],
            outputs=loss,
            updates=seq_updates)

    def train(self, uidx, dist_pos, dist_neg, msk):
        return self.seq_train(uidx,
                              np.asarray(dist_pos, dtype=theano.config.floatX),
                              np.asarray(dist_neg, dtype=theano.config.floatX),
                              msk)


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the class: GRU')


if '__main__' == __name__:
    main()
