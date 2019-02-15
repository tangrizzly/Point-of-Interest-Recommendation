#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 13/03/2018 3:30 PM

@author: Tangrizzly
"""
from __future__ import print_function
from collections import OrderedDict     # 按输入的顺序构建字典
from public.PRME import *
from public.Global_Best import *
from public.Load_Data_prme import *
from public.PRPRM import OboPRPRM
from public.Valuate import *

__docformat__ = 'restructedtext en'

WHOLE = './poidata/'
PATH_f = os.path.join(WHOLE, 'Foursquare/sequence')
PATH_g = os.path.join(WHOLE, 'Gowalla/sequence')
PATH = PATH_f


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
        if not p:
            t = 't'
            assert 't' == t or 'v' == t
            p = OrderedDict(
                [
                    ('dataset',             'Foursquare.txt'),
                    # ('dataset',             'Gowalla.txt'),
                    ('mode',                'test' if 't' == t else 'valid'),
                    ('split',               [0.8, 1.0] if 't' == t else [0.6, 0.8]),   # no third case
                    ('at_nums',             [5, 10, 15, 20]),
                    ('epochs',              100),
                    ('threshold',           360),  # 6 * 60 min
                    ('component_weight',    0.2),
                    ('latent_size',         20),
                    ('alpha',               0.01),
                    ('lambda',              0.001),

                    ('mini_batch',          0),     # 0:one_by_one, 1:mini_batch
                    ('prme',                0),     # 0:prme, 1:pnprm

                    ('batch_size_train',    1),     #
                    ('batch_size_test',     20),   # user * item 矩阵太大了，分成多次计算。 768
                ])
            for i in p.items():
                print(i)

        [(user_num, item_num, cordi), (tra_pois, tes_pois), (tra_all_times, tes_all_times), (tra_all_dists, tes_all_dists)] = \
            load_data(os.path.join(PATH, p['dataset']), p['mode'], p['split'])
        # 正样本加masks
        tra_pois_masks, tra_all_times, tra_all_dists, tra_masks = \
            fun_data_pois_masks(tra_pois, tra_all_times, tra_all_dists, [item_num])
        tes_pois_masks, tes_all_times, tes_all_dists, tes_masks = \
            fun_data_pois_masks(tes_pois, tes_all_times, tes_all_dists, [item_num])

        # 负样本加masks
        tra_pois_neg_masks = fun_random_neg_masks_tra(item_num, tra_pois_masks)  # 训练时用（逐条、mini-batch均可）
        tes_pois_neg_masks = fun_random_neg_masks_tes(item_num, tra_pois_masks, tes_pois_masks)  # 预测时用

        # 3. 创建类变量
        self.p = p
        self.cordi = cordi
        self.tes_pois = tes_pois
        self.user_num, self.item_num = user_num, item_num
        self.tra_pois_masks, self.tra_all_times, self.tra_all_dist, self.tra_masks, self.tra_pois_neg_masks = tra_pois_masks, tra_all_times, tra_all_dists, tra_masks, tra_pois_neg_masks
        self.tes_pois_masks, self.tes_all_times, self.tes_all_dist, self.tes_masks, self.tes_pois_neg_masks = tes_pois_masks, tes_all_times, tes_all_dists, tes_masks, tes_pois_neg_masks

    def build_model_one_by_one(self, flag):
        """
        建立模型对象
        :param flag: 参数变量、数据
        :return:
        """
        print('Building the model one_by_one ...')      # mask只是test计算用户表达时用。
        p = self.p
        size = p['latent_size']
        if flag == 0:
            model = OboPrme(
                train=[self.tra_pois_masks, self.tra_all_times, self.tra_all_dist, self.tra_masks, self.tra_pois_neg_masks],
                test=[self.tes_pois_masks, self.tes_all_times, self.tra_all_dist, self.tes_masks, self.tes_pois_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                threshold=p['threshold'],
                component_weight=p['component_weight'],
                cordi=self.cordi,
                n_user=self.user_num,
                n_item=self.item_num,
                n_size=size)
        else:
            model = OboPRPRM(
                train=[self.tra_pois_masks, self.tra_all_times, self.tra_all_dist, self.tra_masks,
                       self.tra_pois_neg_masks],
                test=[self.tes_pois_masks, self.tes_all_times, self.tra_all_dist, self.tes_masks,
                      self.tes_pois_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                threshold=p['threshold'],
                component_weight=p['component_weight'],
                cordi=self.cordi,
                n_user=self.user_num,
                n_item=self.item_num,
                n_size=size)

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
            size = self.p['batch_size_test']        # test: top-k and acquire user vector
        else:
            size = self.p['batch_size_test'] * 10   # test: auc
        user_num = self.user_num
        rest = (user_num % size) > 0   # 能整除：rest=0。不能整除：rest=1，则多出来一个小的batch
        n_batches = np.minimum(user_num // size + rest, user_num)
        batch_idxs = np.arange(n_batches, dtype=np.int32)
        starts_ends = []
        for bidx in batch_idxs:
            start = bidx * size
            end = np.minimum(start + size, user_num)   # 限制标号索引不能超过user_num
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
    model, model_name, size = pas.build_model_one_by_one(flag=p['prme'])
    best = GlobalBest(at_nums=p['at_nums'])
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='test_auc')

    # 直接取出来部分变量，后边就不用加'pas.'了。
    user_num, item_num = pas.user_num, pas.item_num
    tra_pois_masks, tra_masks, tra_pois_neg_masks = pas.tra_pois_masks, pas.tra_masks, pas.tra_pois_neg_masks
    tes_pois_masks, tes_masks, tes_pois_neg_masks = pas.tes_pois_masks, pas.tes_masks, pas.tes_pois_neg_masks
    tra_all_dist, tra_all_times = pas.tra_all_dist, pas.tra_all_times
    tes_all_dist, tes_all_times = pas.tes_all_dist, pas.tes_all_times

    del pas

    # 主循环
    losses = []
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        # 每次epoch，都要重新选择负样本。都要把数据打乱重排，这样会以随机方式选择样本计算梯度，可得到精确结果
        if epoch > 0:       # epoch=0的负样本已在循环前生成，且已用于类的初始化
            tra_pois_neg_masks = fun_random_neg_masks_tra(item_num, tra_pois_masks)
            tes_pois_neg_masks = fun_random_neg_masks_tes(item_num, tra_pois_masks, tes_pois_masks)
            model.update_neg_masks(tra_pois_neg_masks, tes_pois_neg_masks)

        # ----------------------------------------------------------------------------------------------------------
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        random.seed(str(123 + epoch))
        user_idxs_tra = np.arange(user_num, dtype=np.int32)
        random.shuffle(user_idxs_tra)  # 每个epoch都打乱user_id输入顺序
        for uidx in user_idxs_tra:
            tra = tra_pois_masks[uidx]
            neg = tra_pois_neg_masks[uidx]
            adidx = tra_all_dist[uidx]
            tidx = tra_all_times[uidx]
            for i in np.arange(1, sum(tra_masks[uidx])):
                loss += model.train(uidx, [tra[i], neg[i], tra[i-1]], adidx[i], tidx[i])
                # u_idx, pq_idx, pre_idx, ad_idx, t_idx
        rnn_l2_sqr = model.l2.eval()
        print('\t\tsum_loss = {val} = {v1} + {v2}'.format(val=loss + rnn_l2_sqr, v1=loss, v2=rnn_l2_sqr))
        losses.append('{v1}'.format(v1=int(loss + rnn_l2_sqr)))
        t1 = time.time()
        times0.append(t1 - t0)

        # ----------------------------------------------------------------------------------------------------------
        print("\tPredicting ...")
        # 计算：所有用户、商品的表达
        model.update_trained_items()    # 对于MV-GRU，这里会先算出来图文融合特征。
        t2 = time.time()
        times1.append(t2 - t1)

        # 计算各种指标，并输出当前最优值。
        fun_predict_auc_recall_map_ndcg(
            p, model, best, epoch, starts_ends_auc, starts_ends_tes, tes_pois_masks, tes_masks)
        best.fun_print_best(epoch)   # 每次都只输出当前最优的结果
        t3 = time.time()
        times2.append(t3-t2)
        print('\tavg. time (train, user, test): %0.0fs,' % np.average(times0),
              '%0.0fs,' % np.average(times1), '%0.0fs' % np.average(times2),
              '| alpha, lam: {v1}'.format(v1=', '.join([str(lam) for lam in [p['alpha'], p['lambda']]])),
              '| model: {v1}'.format(v1=model_name))

        # ----------------------------------------------------------------------------------------------------------
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
