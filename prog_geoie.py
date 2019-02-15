#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 2018/10/6 11:48 AM

@author: Tangrizzly
"""

from __future__ import print_function
from collections import OrderedDict
import datetime
import cPickle
import os
from public.GeoIE import GeoIE
from public.Global_Best import GlobalBest
from public.Load_Data_GeoIE import *
from public.Valuate import fun_predict_auc_recall_map_ndcg, fun_save_best_and_losses
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
            assert 't' == t or 'v' == t or 's' == t  # no other case
            p = OrderedDict(
                [
                    ('dataset',             'Foursquare.txt'),
                    # ('dataset',             'Gowalla.txt'),
                    ('mode',                'test' if 't' == t else 'valid' if 'v' == t else 's'),
                    ('load_epoch',          0),
                    ('save_per_epoch',      100),
                    ('split',               -2 if 'v' == t else -1),
                    ('at_nums',             [5, 10, 15, 20]),
                    ('epochs',              101),

                    ('latent_size',         20),
                    ('alpha',               0.01),
                    ('lambda',              0.001),

                    ('mini_batch',          0),     # 0:one_by_one, 1:mini_batch. 全都用逐条。
                    ('GeoIE',               1),     # 1:(ijcai18)GeoIE

                    ('batch_size_train',    1),     #
                    ('batch_size_test',     5),
                ])
            for i in p.items():
                print(i)

        [(user_num, item_num), pois_cordis, (tra_buys, tes_buys), (tra_dist, tes_dist), tra_count] = \
            load_data(os.path.join(PATH, p['dataset']), p['mode'], p['split'])

        tra_buys_masks, tra_dist_masks, tra_masks, tra_count = fun_data_buys_masks(tra_buys, tra_dist, [item_num], [0], tra_count)
        tes_buys_masks, tes_dist_masks, tes_masks = fun_data_buys_masks(tes_buys, tes_dist, [item_num], [0])
        tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)
        tes_buys_neg_masks = fun_random_neg_masks_tes(item_num, tra_buys_masks, tes_buys_masks)
        tra_dist_pos_masks, tra_dist_neg_masks, tra_dist_masks = fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis)
        usrs_last_poi_to_all_intervals = fun_compute_distance(tra_buys, tra_masks, pois_cordis, p['batch_size_test'])

        self.p = p
        self.user_num, self.item_num = user_num, item_num
        self.pois_cordis = pois_cordis
        self.tra_count = tra_count
        self.tra_masks, self.tes_masks = tra_masks, tes_masks
        self.tra_buys_masks, self.tes_buys_masks = tra_buys_masks, tes_buys_masks
        self.tra_buys_neg_masks, self.tes_buys_neg_masks = tra_buys_neg_masks, tes_buys_neg_masks
        self.tra_dist_pos_masks, self.tra_dist_neg_masks, self.tra_dist_masks = tra_dist_pos_masks, tra_dist_neg_masks, tra_dist_masks
        self.ulptai = usrs_last_poi_to_all_intervals

    def build_model_one_by_one(self, flag=0):
        """
        建立模型对象
        :param flag: 参数变量、数据
        :return:
        """
        print('Building the model one_by_one ...')
        p = self.p
        size = p['latent_size']
        model = GeoIE(
            train=[self.tra_buys_masks, self.tra_buys_neg_masks, self.tra_count, self.tra_masks],
            test=[self.tes_buys_masks, self.tes_buys_neg_masks],
            alpha_lambda=[p['alpha'], p['lambda']],
            n_user=self.user_num,
            n_item=self.item_num,
            n_in=size,
            n_hidden=size,
            ulptai=self.ulptai)
        model_name = model.__class__.__name__
        print('\t the current Class name is: {val}'.format(val=model_name))
        return model, model_name

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
            size = self.p['batch_size_test']
        else:
            size = self.p['batch_size_test'] * 10
        user_num = self.user_num
        rest = (user_num % size) > 0
        n_batches = np.minimum(user_num // size + rest, user_num)
        batch_idxs = np.arange(n_batches, dtype=np.int32)
        starts_ends = []
        for bidx in batch_idxs:
            start = bidx * size
            end = np.minimum(start + size, user_num)
            start_end = np.arange(start, end, dtype=np.int32)
            starts_ends.append(start_end)
        return batch_idxs, starts_ends


def train_valid_or_test(pas):
    """
    主程序
    :return:
    """
    p = pas.p
    model, model_name = pas.build_model_one_by_one(flag=p['GeoIE'])
    best = GlobalBest(at_nums=p['at_nums'])
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='test_auc')

    user_num, item_num = pas.user_num, pas.item_num
    tra_masks, tes_masks = pas.tra_masks, pas.tes_masks
    tra_buys_masks, tes_buys_masks = pas.tra_buys_masks, pas.tes_buys_masks
    tra_dist_pos_masks, tra_dist_neg_masks, tra_dist_masks = pas.tra_dist_pos_masks, pas.tra_dist_neg_masks, pas.tra_dist_masks

    pois_cordis = pas.pois_cordis
    del pas

    # 主循环
    losses = []
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(0, p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        if epoch > 0:
            tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)

            tra_dist_pos_masks, tra_dist_neg_masks, tra_dist_masks = fun_compute_dist_neg(tra_buys_masks, tra_masks,
                                                                                          tra_buys_neg_masks,
                                                                                          pois_cordis)

        # ----------------------------------------------------------------------------------------------------------
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        ls = [0, 0]
        total_ls = []
        random.seed(str(123 + epoch))
        user_idxs_tra = np.arange(user_num, dtype=np.int32)
        random.shuffle(user_idxs_tra)
        for uidx in user_idxs_tra:
            print(model.a.eval(), model.b.eval())
            dist_pos = tra_dist_pos_masks[uidx]
            dist_neg = tra_dist_neg_masks[uidx]
            msk = tra_dist_masks[uidx]
            tmp = model.train(uidx, dist_pos, dist_neg, msk)
            loss += tmp
            print(tmp)
        rnn_l2_sqr = model.l2.eval()

        def cut2(x):
            return '%0.2f' % x

        print('\t\tsum_loss = {val} = {v1} + {v2}'.format(val=loss + rnn_l2_sqr, v1=loss, v2=rnn_l2_sqr))
        losses.append('{v1}'.format(v1=int(loss + rnn_l2_sqr)))
        # ls = model.loss_weight
        print('\t\tloss_weight = {v1}, {v2}'.format(v1=ls[0], v2=ls[1]))
        t1 = time.time()
        times0.append(t1 - t0)

        # ----------------------------------------------------------------------------------------------------------
        print("\tPredicting ...")
        model.update_trained()
        t2 = time.time()
        times1.append(t2 - t1)

        fun_predict_auc_recall_map_ndcg(
            p, model, best, epoch, starts_ends_auc, starts_ends_tes, tes_buys_masks, tes_masks)
        best.fun_print_best(epoch)
        t3 = time.time()
        times2.append(t3-t2)
        print('\tavg. time (train, user, test): %0.0fs,' % np.average(times0),
              '%0.0fs,' % np.average(times1), '%0.0fs' % np.average(times2),
              '| alpha, lam: {v1}'.format(v1=', '.join([str(lam) for lam in [p['alpha'], p['lambda']]])),
              '| model: {v1}'.format(v1=model_name))

        # ----------------------------------------------------------------------------------------------------------
        if epoch == p['epochs'] - 1:
            print("\tBest and losses saving ...")
            path = os.path.join(os.path.split(__file__)[0], '..', 'Results_best_and_losses', PATH.split('/')[-2])
            fun_save_best_and_losses(path, model_name, epoch, p, best, losses)
            if 2 == p['gru']:
                size = p['latent_size']
                fil_name = 'size' + str(size) + 'UD' + str(p['UD']) + 'dd' + str(p['dd']) + 'loss.txt'
                fil = os.path.join(path, fil_name)
                np.savetxt(fil, total_ls)

        if 2 == p['gru'] and epoch % p['save_per_epoch'] == 0 and epoch != 0:
            m_path = './model/' + p['dataset'] + '/' + model_name + '_size' + \
                     str(p['latent_size']) + '_UD' + str(p['UD']) + '_dd' + str(p['dd']) + '_epoch' + str(epoch)
            with open(m_path, 'wb') as file:
                save_model = [model.loss_weight.get_value(), model.wd.get_value(), model.lt.get_value(), model.di.get_value(),
                              model.ui.get_value(), model.wh.get_value(), model.bi.get_value(), model.vs.get_value(),
                              model.bs.get_value()]
                cPickle.dump(save_model, file, protocol=cPickle.HIGHEST_PROTOCOL)

    for i in p.items():
        print(i)
    print('\t the current Class name is: {val}'.format(val=model_name))


@exe_time
def main():
    pas = Params()
    train_valid_or_test(pas)


if '__main__' == __name__:
    main()
