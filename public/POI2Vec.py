#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 23/03/2018 10:15 AM

@author: Tangrizzly
"""

from __future__ import print_function

from theano.tensor.nnet import sigmoid

import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
class Poi2vecBasic(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_node, n_size, probs, routes, lrs):
        """
        构建 模型参数
        :param n_item: 商品items的数目
        :param n_in: rnn输入向量的维度
        :return:
        """
        rng = np.random.RandomState(123)
        self.thea_rng = RandomStreams(rng.randint(2 ** 30))  # 旗下随机函数可在GPU下运行。
        # 用mask进行补全后的train/test
        tra_target_masks, tra_context_masks, tra_masks, tra_masks_cot, tra_accum_lens = train
        tes_target_masks, tes_context_masks, tes_masks, tes_masks_cot, tes_accum_lens = test
        self.tra_target_masks = theano.shared(borrow=True, value=np.asarray(tra_target_masks, dtype='int32'))
        self.tes_target_masks = theano.shared(borrow=True, value=np.asarray(tes_target_masks, dtype='int32'))
        self.tra_context_masks = theano.shared(borrow=True, value=np.asarray(tra_context_masks, dtype='int32'))
        self.tes_context_masks = theano.shared(borrow=True, value=np.asarray(tes_context_masks, dtype='int32'))
        self.tra_masks_cot = theano.shared(borrow=True, value=np.asarray(tra_masks_cot, dtype='int32'))
        self.tes_masks_cot = theano.shared(borrow=True, value=np.asarray(tes_masks_cot, dtype='int32'))
        self.tra_accum_lens = theano.shared(borrow=True, value=np.asarray(tra_accum_lens, dtype='int32'))
        self.tes_accum_lens = theano.shared(borrow=True, value=np.asarray(tes_accum_lens, dtype='int32'))
        self.tra_masks = theano.shared(borrow=True, value=np.asarray(tra_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        self.probs = theano.shared(borrow=True, value=np.asarray(probs, dtype=theano.config.floatX))
        self.routes = theano.shared(borrow=True, value=np.asarray(routes, dtype='int32'))
        self.lrs = theano.shared(borrow=True, value=np.asarray(lrs, dtype='int32'))
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        xu = uniform(-rang, rang, (n_user, n_size))
        wl = uniform(-rang, rang, (n_item, n_size))
        pb = uniform(-rang, rang, (n_node, n_size))
        wl_m = np.zeros((1, n_size))
        self.xu = theano.shared(borrow=True, value=xu.astype(theano.config.floatX))
        self.wl = theano.shared(borrow=True, value=wl.astype(theano.config.floatX))
        self.pb = theano.shared(borrow=True, value=pb.astype(theano.config.floatX))
        self.wl_m = theano.shared(borrow=True, value=wl_m)
        # 存放训练好的users、items表达。用于计算所有users对所有items的评分：users * items
        trained_items = uniform(-rang, rang, (n_item + 1, n_size))
        trained_users = uniform(-rang, rang, (n_user, n_size))
        trained_branch = uniform(-rang, rang, (n_node, n_size))
        self.trained_items = theano.shared(borrow=True, value=trained_items.astype(theano.config.floatX))
        self.trained_users = theano.shared(borrow=True, value=trained_users.astype(theano.config.floatX))
        self.trained_branch = theano.shared(borrow=True, value=trained_branch.astype(theano.config.floatX))

    def update_trained_params(self):
        xu = self.xu.get_value(borrow=True)
        self.trained_users.set_value(np.asarray(xu, dtype=theano.config.floatX), borrow=True)
        wl = self.wl.get_value(borrow=True)
        wl_m = self.wl_m.get_value(borrow=True)
        wl_f = np.concatenate((wl, wl_m))
        self.trained_items.set_value(np.asarray(wl_f, dtype=theano.config.floatX), borrow=True)
        pb = self.pb.get_value(borrow=True)
        self.trained_branch.set_value(np.asarray(pb, dtype=theano.config.floatX), borrow=True)

    def compute_sub_all_scores(self, start_end):
        plu = softmax(T.dot(self.trained_users[start_end], self.trained_items.T))[:, :-1]  # (n_batch, n_item)
        length = T.max(T.sum(self.tes_masks[start_end], axis=1))  # 253
        cidx = T.arange(length).reshape((1, length)) + self.tra_accum_lens[start_end][:, 0].reshape((len(start_end), 1))
        cl = T.sum(self.trained_items[self.tra_context_masks[cidx]], axis=2)  # n_batch x seq_length x n_size
        cl = cl.dimshuffle(1, 2, 0)
        pb = self.trained_branch[self.routes]  # (n_item x 4 x tree_depth x n_size)
        shp0, shp1, shp2 = self.lrs.shape
        lrs = self.lrs.reshape((shp0, shp1, shp2, 1, 1))
        pr_bc = T.dot(pb, cl)
        br = sigmoid(pr_bc * lrs) * T.ceil(abs(pr_bc))  # (n_item x 4 x tree_depth x seq_length x n_batch)
        path = T.prod(br, axis=2) * self.probs.reshape((shp0, shp1, 1, 1))
        del cl, pb, br, lrs
        # paths = T.prod((T.floor(1 - path) + path), axis=1)  # (n_item x seq_length x n_batch)
        paths = T.sum(path, axis=1)
        paths = T.floor(1 - paths) + paths
        p = paths[:-1].T * plu.reshape((plu.shape[0], 1, plu.shape[1]))  # (n_batch x n_item)
        # p = plu.reshape((plu.shape[0], 1, plu.shape[1])) * T.ones((plu.shape[0], length, plu.shape[1]))
        return T.reshape(p, (p.shape[0] * p.shape[1], p.shape[2])).eval()

    def compute_sub_auc_preference(self, start_end):
        return np.array([[0 for _ in np.arange(len(self.tes_masks[0].eval()))]])


class Poi2vec(Poi2vecBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_node, n_size, probs, routes, lrs):
        super(Poi2vec, self).__init__(train, test, alpha_lambda, n_user, n_item, n_node, n_size, probs, routes, lrs)
        self.params = [self.wl]
        self.l2_sqr = (
            T.sum(self.xu ** 2) +
            T.sum(self.pb ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_size)

    def __theano_train__(self, n_size):
        """
        Pr(l|u, C(l)) = Pr(l|u) * Pr(l|C(l))
        Pr(u, l, t) = Pr(l|u, C(l))     if C(l) exists,
                      Pr(l|u)           otherwise.
        $Theta$ = argmax Pr(u, l, t)
        """
        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)  # 有效长度
        wl = T.concatenate((self.wl, self.wl_m))
        tidx, cidx, bidx, userid = T.ivector(), T.imatrix(), T.itensor3(), T.iscalar()
        pb = self.pb[bidx]  # (seq_length x 4 x depth x n_size)
        lrs = self.lrs[tidx]  # (seq_length x 4 x depth)
        # user preference
        xu = self.xu[userid]
        plu = softmax(T.dot(xu, self.wl.T))
        # geographical influence
        cl = T.sum(wl[cidx], axis=1)  # (seq_length x n_size)
        cl = cl.reshape((cl.shape[0], 1, 1, cl.shape[1]))
        br = sigmoid(T.sum(pb[:seq_length] * cl, axis=3) * lrs[:seq_length]) * T.ceil(abs(T.mean(cl, axis=3)))
        path = T.prod(br, axis=2) * self.probs[tidx][:seq_length]
        # paths = T.prod((T.floor(1-path) + path), axis=1)
        paths = T.sum(path, axis=1)
        paths = T.floor(1 - paths) + paths
        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xu, self.wl]])
        upq = - 1 * T.sum(T.log(plu[tidx[:seq_length]] * paths)) / seq_length
        seq_costs = (
            upq +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        pars_subs = [(self.xu, xu), (self.pb, pb)]
        seq_updates.extend([(par, T.set_subtensor(sub, sub - lr * T.grad(seq_costs, sub)))
                            for par, sub in pars_subs])
        # ----------------------------------------------------------------------------
        uidx = T.iscalar()  # T.iscalar()类型是 TensorType(int32, )
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=upq,
            updates=seq_updates,
            givens={
                userid: uidx,
                tidx: self.tra_target_masks[uidx],
                cidx: self.tra_context_masks[T.arange(self.tra_accum_lens[uidx][0], self.tra_accum_lens[uidx][1])],
                bidx: self.routes[self.tra_target_masks[uidx]],
                tra_mask: self.tra_masks[uidx]
                # tra_mask_cot: self.tra_masks_cot[T.arange(self.tra_accum_lens[uidx][0], self.tra_accum_lens[uidx][1])]
            })

    def train(self, idx):
        return self.seq_train(idx)


def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out
