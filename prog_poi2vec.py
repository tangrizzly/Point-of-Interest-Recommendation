#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 23/03/2018 10:15 AM

@author: Tangrizzly
"""

from __future__ import print_function
import random
from collections import OrderedDict  # 按输入的顺序构建字典
from public.POI2Vec import *
from public.Global_Best import *
from public.Load_Data_Poi2vec import *
from public.Valuate import *

__docformat__ = 'restructedtext en'

WHOLE = './poidata/'
PATH_f = os.path.join(WHOLE, 'Foursquare/sequence')
PATH_g = os.path.join(WHOLE, 'Gowalla/sequence')
PATH_f_nyc = os.path.join(WHOLE, 'Foursquare2014_NYC/sequence')
PATH = PATH_g


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh" % (name, total, total / 3600.0))
        return back

    return new_func


class Params(object):
    def __init__(self, p=None):
        # 1. 建立各参数。要调整的地方都在 p 这了，其它函数都给写死。
        if not p:
            t = 't'  # 写1就是valid, 写0就是test
            assert 't' == t or 'v' == t  # no other case
            p = OrderedDict(
                [
                    # ('dataset',             'Foursquare.txt'),
                    ('dataset',           'Gowalla.txt'),
                    # ('dataset',           'Foursquare2014_NYC.txt'),
                    ('mode',                'test' if 't' == t else 'valid'),
                    ('regionThreshold',     0.1),  # 11 km
                    ('timeThreshold',       360),  # 6 * 60 min
                    ('split',               [0.8, 1.0] if 't' == t else [0.6, 0.8]),  # no third case
                    ('at_nums',             [5, 10, 15, 20]),
                    ('epochs',              100),
                    ('latent_size',         20),
                    ('initial_alpha',       1),
                    ('initial_loss',        25000),
                    ('alpha',               0.01),
                    ('lambda',              0.001),
                    ('mini_batch',          0),  # 0:one_by_one, 1:mini_batch
                    ('poi2vec',             0),
                    ('batch_size_train',    4),  #
                    ('batch_size_test',     25),  # user * item 矩阵太大了，分成多次计算。 768
                ])
            for i in p.items():
                print(i)

        (user_num, item_num, node_num), (tra_target, tes_target), (tra_context, tes_context), \
        (probs, routes, lrs) = load_data()

        # masks
        tra_target_masks, tra_context_masks, tra_masks, tra_masks_cot, tra_accum_lens = \
            fun_data_masks(tra_target, tra_context, [item_num])
        tes_target_masks, tes_context_masks, tes_masks, tes_masks_cot, tes_accum_lens = \
            fun_data_masks(tes_target, tes_context, [item_num])

        # 3. 创建类变量
        self.tes_target = tes_target
        self.p = p
        self.user_num, self.item_num, self.node_num, self.probs, self.routes, self.lrs = user_num, item_num, node_num, probs, routes, lrs
        self.tra_target_masks, self.tra_context_masks, self.tra_masks, self.tra_masks_cot, self.tra_accum_lens = tra_target_masks, tra_context_masks, tra_masks, tra_masks_cot, tra_accum_lens
        self.tes_target_masks, self.tes_context_masks, self.tes_masks, self.tes_masks_cot, self.tes_accum_lens = tes_target_masks, tes_context_masks, tes_masks, tes_masks_cot, tes_accum_lens

    def build_model_one_by_one(self, flag):
        """
        建立模型对象
        :param flag: 参数变量、数据
        :return:
        """
        print('Building the model one_by_one ...')  # mask只是test计算用户表达时用。
        p = self.p
        size = p['latent_size']
        model = Poi2vec(
            train=[self.tra_target_masks, self.tra_context_masks, self.tra_masks, self.tra_masks_cot, self.tra_accum_lens],
            test=[self.tes_target_masks, self.tes_context_masks, self.tes_masks, self.tes_masks_cot, self.tes_accum_lens],
            alpha_lambda=[p['initial_alpha'], p['lambda']],
            n_user=self.user_num,
            n_item=self.item_num,
            n_node=self.node_num,
            n_size=size,
            probs=self.probs,
            routes=self.routes,
            lrs=self.lrs)
        model_name = model.__class__.__name__
        print('\t the current Class name is: {val}'.format(val=model_name))
        return model, model_name, size

    def compute_start_end(self, flag):
        """
        获取mini-batch的各个start_end(np.array类型，一组连续的数值)
        :param flag: 'train', 'test'
        :return: 各个start_end组成的list
        """
        assert flag in ['train', 'test', 'test_auc']
        if 'train' == flag:
            size = self.p['batch_size_train']
        elif 'test' == flag:
            size = self.p['batch_size_test']  # test: top-k and acquire user vector
        else:
            size = self.p['batch_size_test'] * 10  # test: auc
        user_num = self.user_num
        rest = (user_num % size) > 0  # 能整除：rest=0。不能整除：rest=1，则多出来一个小的batch
        n_batches = np.minimum(user_num // size + rest, user_num)
        batch_idxs = np.arange(n_batches, dtype=np.int32)
        starts_ends = []
        for bidx in batch_idxs:
            start = bidx * size
            end = np.minimum(start + size, user_num)  # 限制标号索引不能超过user_num
            start_end = np.arange(start, end, dtype=np.int32)
            starts_ends.append(start_end)
        return batch_idxs, starts_ends


def train_valid_or_test():
    """
    主程序
    :return:
    """
    # 建立参数、数据、模型、模型最佳值
    pas = Params()
    p = pas.p
    model, model_name, size = pas.build_model_one_by_one(flag=p['poi2vec'])
    best = GlobalBest(at_nums=p['at_nums'])
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='test_auc')

    # 直接取出来部分变量，后边就不用加'pas.'了。
    user_num, item_num = pas.user_num, pas.item_num
    tes_target = pas.tes_target_masks
    tes_masks = pas.tes_masks

    del pas

    # 主循环
    losses = []
    pre_loss, lr_min = p['initial_loss'], p['alpha']
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        # 每次epoch，都要重新选择负样本。都要把数据打乱重排，这样会以随机方式选择样本计算梯度，可得到精确结果
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        random.seed(str(123 + epoch))
        user_idxs_tra = np.arange(user_num, dtype=np.int32)
        random.shuffle(user_idxs_tra)  # 每个epoch都打乱user_id输入顺序
        for uidx in user_idxs_tra:
            uloss = model.train(uidx)
            loss += uloss
        rnn_l2_sqr = model.l2.eval()
        print('\t\tsum_loss = {val} = {v1} + {v2}'.format(val=loss + rnn_l2_sqr, v1=loss, v2=rnn_l2_sqr))
        losses.append('{v1}'.format(v1=int(loss + rnn_l2_sqr)))
        t1 = time.time()
        times0.append(t1 - t0)

        # ----------------------------------------------------------------------------------------------------------
        print("\tPredicting ...")
        # 计算：所有用户、商品的表达
        model.update_trained_params()  # 对于MV-GRU，这里会先算出来图文融合特征。
        t2 = time.time()
        times1.append(t2 - t1)

        # 计算各种指标，并输出当前最优值。
        fun_predict_auc_recall_map_ndcg(
            p, model, best, epoch, starts_ends_auc, starts_ends_tes, tes_target, tes_masks)
        best.fun_print_best(epoch)  # 每次都只输出当前最优的结果
        t3 = time.time()
        times2.append(t3 - t2)
        print('\tavg. time (train, user, test): %0.0fs,' % np.average(times0),
              '%0.0fs,' % np.average(times1), '%0.0fs' % np.average(times2),
              '| alpha, lam: {v1}'.format(v1=', '.join([str(lam) for lam in [model.alpha_lambda[0].eval(), p['lambda']]])),
              '| model: {v1}'.format(v1=model_name))

        # ----------------------------------------------------------------------------------------------------------
        lr = model.alpha_lambda[0].eval()
        if pre_loss < loss and lr >= lr_min * 10:
            model.alpha_lambda.set_value(np.asarray([lr / 10, p['lambda']], dtype=theano.config.floatX))
        pre_loss = loss
        if epoch == p['epochs'] - 1:
            # 保存最优值、所有的损失值。
            print("\tBest and losses saving ...")
            path = os.path.join(os.path.split(__file__)[0], '..', 'Results_best_and_losses', PATH.split('/')[-2])
            fun_save_best_and_losses(path, model_name, epoch, p, best, losses)

    for i in p.items():
        print(i)
    print('\t the current Class name is: {val}'.format(val=model_name))


@exe_time
def main():
    train_valid_or_test()


if '__main__' == __name__:
    main()
