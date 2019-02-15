#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
import random
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
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


# 'Obo' is one by one. 逐条训练。
# ======================================================================================================================
class OboFpmc_lr(object):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_size):
        """
        构建 模型参数
        :param train: 添加mask后的
        :param test: 添加mask后的
        :param n_user: 用户的真实数目
        :param n_item: 商品items的真正数目，init()里补全一个商品作为填充符
        :param n_size: 向量的维度
        :return:
        """
        # train
        tra_buys, tra_buys_negs, tra_last_poi = train
        self.tra_buys = tra_buys
        self.tra_buys_negs = tra_buys_negs
        self.tra_last_poi = theano.shared(borrow=True, value=np.asarray(tra_last_poi, dtype='int32'))
        # test
        tes_buys_masks, tes_masks, tes_buys_neg_masks = test
        self.tes_buys_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.tes_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_neg_masks, dtype='int32'))
        # 把超参数shared
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        # ua = uniform(-rang, rang, (n_user, n_size))         # 这2个实际上用不到。t-1时刻的。
        # au = uniform(-rang, rang, (n_item + 1, n_size))
        ui = uniform(-rang, rang, (n_user, n_size))
        iu = uniform(-rang, rang, (n_item + 1, n_size))
        ia = uniform(-rang, rang, (n_item + 1, n_size))
        ai = uniform(-rang, rang, (n_item + 1, n_size))
        # 建立参数。t-1时刻的a。t时刻的正样本i，负样本j也用i矩阵表示。
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.iu = theano.shared(borrow=True, value=iu.astype(theano.config.floatX))
        self.ai = theano.shared(borrow=True, value=ai.astype(theano.config.floatX))
        self.ia = theano.shared(borrow=True, value=ia.astype(theano.config.floatX))
        # 存放训练好的users、items表达。用于计算所有users对所有items的评分：users * items
        self.params = [self.ui, self.iu, self.ai, self.ia]
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_size)

    def update_neg_masks(self, tes_buys_neg_masks):
        # 每个epoch都更新负样本
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def compute_sub_all_scores(self, start_end):    # 其实可以直接传过来实数参数
        # 计算users * items，每个用户对所有商品的评分(需去掉填充符)
        # 基于训练集中最后一个poi(t-1时刻，用a)，计算跳转到其它所有pois的得分(t时刻，用i)。
        # start_end之间的usr、poi(t-1时刻)，对所有poi的得分(t时刻)。ui*iu + ai*ia
        sub_all_scores = T.dot(self.ui[start_end], self.iu[:-1].T) + \
                         T.dot(self.ai[self.tra_last_poi[start_end]], self.ia[:-1].T)
        return sub_all_scores.eval()                # shape=(sub_n_user, n_item)

    def compute_sub_auc_preference(self, start_end):
        # items.shape=(n_item+1, 20)，因为是mask形式，所以需要填充符。(ui*iu + ai*ia) - (ui*ju + ai*ja)
        # 注意矩阵形式的索引方式。
        # 1. usr对t时刻的poi(就是test里的)
        users = self.ui[start_end]
        tes_usrs = self.iu[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_usrs_neg = self.iu[self.tes_buys_neg_masks[start_end]]
        # 2. t-1时刻的poi对t时刻的poi
        pois_pre1 = self.ai[self.tra_last_poi[start_end]]
        tes_items = self.ia[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_items_neg = self.ia[self.tes_buys_neg_masks[start_end]]
        shp0, shp2 = users.shape        # shape=(sub_n_user, n_hidden)
        # 利用性质：(n_user, 1, n_hidden) * (n_user, len, n_hidden) = (n_user, len, n_hidden)，即broadcast
        # 利用性质：np.sum((n_user, len, n_hidden), axis=2) = (n_user, len)，
        #         即得到各用户对test里正负样本的偏好值
        all_upqs = T.sum(users.reshape((shp0, 1, shp2)) * (tes_usrs - tes_usrs_neg), axis=2) + \
            T.sum(pois_pre1.reshape((shp0, 1, shp2)) * (tes_items - tes_items_neg), axis=2)
        all_upqs *= self.tes_masks[start_end]       # 只保留原先test items对应有效位置的偏好值
        return all_upqs.eval() > 0                  # 将>0的标为True, 也就是1

    def __theano_train__(self, n_size):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        uidx = T.iscalar()
        aidx = T.iscalar()
        tidxs = T.ivector()     # t时刻的正负样本。

        ui = self.ui[uidx]
        ai = self.ai[aidx]
        tu = self.iu[tidxs]
        ta = self.ia[tidxs]
        """
        t-1时刻样本a，t时刻正样本i，t时刻负样本j。
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # FPMC
            x(uti)=x(uai)= ui*iu + ai*ia    # ua、au实际上用不到。
            x(utj)=x(uaj)= uj*ju + aj*ja    # j与i采用完全相同的矩阵。差别只是j与i的idx不同。
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        upq = T.dot(tu[0]-tu[1:], ui) + T.dot(ta[0]-ta[1:], ai)   # shape=(le, )
        los = T.sum(T.log(sigmoid(upq)))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [ui, ai, tu, ta]])
        costs = (
            los -
            0.5 * l2 * seq_l2_sq)
        pars_subs = [(self.ui, ui), (self.ai, ai),
                     (self.iu, tu), (self.ia, ta)]
        updates = [(par, T.set_subtensor(sub, sub + lr * T.grad(costs, sub)))
                   for par, sub in pars_subs]
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        self.seq_train = theano.function(
            inputs=[uidx, aidx, tidxs],
            outputs=los,
            updates=updates)

    def train(self, uidx, aidx, iidx, jidxs):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        """
        uidx: usr_id,
        aidx: t-1时刻的poi
        iidx: t时刻正样本poi
        jidxs: t时刻负样本pois，就是正样本的邻居。
        """
        return self.seq_train(uidx, aidx, [iidx] + jidxs)


@exe_time
def main():
    print('... construct the class: FPMC_LR')


if '__main__' == __name__:
    main()

