#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 13/03/2018 3:30 PM

@author: Tangrizzly
"""

from __future__ import print_function
from collections import OrderedDict
import time
import datetime
import cPickle
import numpy as np
import os
import random
from public.BPR import OboBpr
from public.GRU import OboGru
from public.CA_RNN import OboCARNN
from public.GRU_Spatial import OboSpatialGru
from public.Global_Best import GlobalBest
from public.Load_Data_by_length import load_data, fun_data_buys_masks
from public.Load_Data_by_length import fun_random_neg_masks_tra, fun_random_neg_masks_tes, fun_compute_dist_neg
from public.Load_Data_by_length import fun_compute_distance, fun_acquire_prob
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
            assert 't' == t or 'v' == t or 's' == t
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
                    ('loss_weight',         [0.5, 0.5]),

                    ('dd',                  200),
                    ('UD',                  40),

                    ('mini_batch',          0),
                    ('gru',                 0),

                    ('batch_size_train',    1),
                    ('batch_size_test',     32),
                ])
            for i in p.items():
                print(i)
        dist_num = int(p['UD'] * 1000 / p['dd'])    # 1520 = 38*1000/25。idx=[0, 1519+1]

        [(user_num, item_num), pois_cordis, (tra_buys, tes_buys), (tra_dist, tes_dist)] = \
            load_data(os.path.join(PATH, p['dataset']), p['mode'], p['split'], p['dd'], dist_num)
        tra_buys_masks, tra_dist_masks, tra_masks = fun_data_buys_masks(tra_buys, tra_dist, [item_num], [dist_num])
        tes_buys_masks, tes_dist_masks, tes_masks = fun_data_buys_masks(tes_buys, tes_dist, [item_num], [dist_num])
        tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)
        tes_buys_neg_masks = fun_random_neg_masks_tes(item_num, tra_buys_masks, tes_buys_masks)
        tra_dist_neg_masks = fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis, p['dd'], dist_num)
        usrs_last_poi_to_all_intervals = fun_compute_distance(tra_buys_masks, tra_masks, pois_cordis, p['dd'], dist_num)

        self.p = p
        self.user_num, self.item_num, self.dist_num = user_num, item_num, dist_num
        self.pois_cordis = pois_cordis
        self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks = tra_buys_masks, tra_masks, tra_buys_neg_masks
        self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks = tes_buys_masks, tes_masks, tes_buys_neg_masks
        self.tra_dist_masks = tra_dist_masks
        self.tes_dist_masks = tes_dist_masks
        self.tra_dist_neg_masks = tra_dist_neg_masks
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
        if 0 == flag:
            model = OboBpr(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_in=size,
                n_hidden=size)
        elif 1 == flag:
            model = OboGru(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_in=size,
                n_hidden=size)
        elif 2 == flag:
            model = OboSpatialGru(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                dist= [self.tra_dist_masks, self.tes_dist_masks, self.tra_dist_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_dists=[self.dist_num, 1.0*p['dd']/1000],
                n_in=size,              # 总的input是2 * size。
                n_hidden=size)
        else:
            model = OboCARNN(
                train=[self.tra_buys_masks, self.tra_masks, self.tra_buys_neg_masks],
                test= [self.tes_buys_masks, self.tes_masks, self.tes_buys_neg_masks],
                dist= [self.tra_dist_masks, self.tes_dist_masks, self.tra_dist_neg_masks],
                alpha_lambda=[p['alpha'], p['lambda']],
                n_user=self.user_num,
                n_item=self.item_num,
                n_dists=[self.dist_num, 1.0*p['dd']/1000],
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
            size = self.p['batch_size_test']        # test: top-k and acquire user vector
        else:
            size = self.p['batch_size_test'] * 10   # test: auc
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
    # 建立参数、数据、模型、模型最佳值
    p = pas.p
    model, model_name = pas.build_model_one_by_one(flag=p['gru'])
    best = GlobalBest(at_nums=p['at_nums'])   # 存放最优数据
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    _, starts_ends_auc = pas.compute_start_end(flag='test_auc')

    # 直接取出来部分变量，后边就不用加'pas.'了。
    user_num, item_num, dist_num = pas.user_num, pas.item_num, pas.dist_num
    tra_buys_masks, tra_masks, tra_buys_neg_masks = pas.tra_buys_masks, pas.tra_masks, pas.tra_buys_neg_masks
    tes_buys_masks, tes_masks, tes_buys_neg_masks = pas.tes_buys_masks, pas.tes_masks, pas.tes_buys_neg_masks
    dd = p['dd']
    pois_cordis = pas.pois_cordis
    ulptai = pas.ulptai
    del pas

    ini_epoch = 0
    if 2 == p['gru']:
        ini_epoch = p['load_epoch']
        if p['load_epoch'] != 0:
            print('Loading model ...')
            m_path = './model/' + p['dataset'] + '/' + model_name + '_size' + \
                     str(p['latent_size']) + '_UD' + str(p['UD']) + '_dd' + str(p['dd']) + '_epoch' + str(
                p['load_epoch'])
            with open(m_path, 'rb') as f:
                loaded_objects = cPickle.load(f)
            model.load_params(loaded_objects)
            ini_epoch = p['load_epoch'] + 1

    # 主循环
    losses = []
    times0, times1, times2, times3 = [], [], [], []
    for epoch in np.arange(ini_epoch, p['epochs']):
        print("Epoch {val} ==================================".format(val=epoch))
        if epoch > 0:
            tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)
            tes_buys_neg_masks = fun_random_neg_masks_tes(item_num, tra_buys_masks, tes_buys_masks)
            if p['gru'] in [0, 1]:
                model.update_neg_masks(tra_buys_neg_masks, tes_buys_neg_masks)
            else:
                tra_dist_neg_masks = fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis, dd, dist_num)
                model.s_update_neg_masks(tra_buys_neg_masks, tes_buys_neg_masks, tra_dist_neg_masks)

        # ----------------------------------------------------------------------------------------------------------
        print("\tTraining ...")
        t0 = time.time()
        loss = 0.
        ls = [0, 0]
        total_ls = []
        random.seed(str(123 + epoch))
        user_idxs_tra = np.arange(user_num, dtype=np.int32)
        random.shuffle(user_idxs_tra)
        if 0 == p['gru']:
            for uidx in user_idxs_tra:
                tra = tra_buys_masks[uidx]
                neg = tra_buys_neg_masks[uidx]
                for i in np.arange(sum(tra_masks[uidx])):
                    loss += model.train(uidx, [tra[i], neg[i]])
        elif 1 == p['gru'] or 3 == p['gru']:
            for uidx in user_idxs_tra:
                loss += model.train(uidx)
        else:
            for uidx in user_idxs_tra:
                los, a, b, ls = model.train(uidx)
                loss += los
                a_b = [a, b]
                a_b.extend(ls)
                total_ls.append(a_b)
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
        if 0 == p['gru']:
            model.update_trained_items()
            model.update_trained_users()
        elif 1 == p['gru']:
            model.update_trained_items()
            all_hus = []
            for start_end in starts_ends_tes:
                sub_all_hus = model.predict(start_end)
                all_hus.extend(sub_all_hus)
            model.update_trained_users(all_hus)
        elif 2 == p['gru']:
            model.update_trained_items()
            model.update_trained_dists()
            all_hus = []
            all_sus = []
            for start_end in starts_ends_tes:
                [sub_all_hus, sub_all_sus] = model.predict(start_end)
                all_hus.extend(sub_all_hus)
                all_sus.extend(sub_all_sus)
            probs = fun_acquire_prob(all_sus, ulptai, dist_num)
            model.update_trained_users(all_hus)
            model.update_prob(probs)
        else:
            model.update_trained_items()
            model.update_trained_dists()
            all_hus = []
            for start_end in starts_ends_tes:
                sub_all_hus = model.predict(start_end)
                all_hus.extend(sub_all_hus)
            model.update_trained_users(all_hus)
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


def cal_s(pas):
    p = pas.p
    model, model_name = pas.build_model_one_by_one(flag=p['gru'])
    _, starts_ends_tes = pas.compute_start_end(flag='test')
    user_num, dist_num = pas.user_num, pas.dist_num
    ulptai = pas.ulptai

    del pas

    m_path = './model/' + p['dataset'] + '/' + model_name + '_size' +\
             str(p['latent_size']) + '_UD' + str(p['UD']) + '_dd' + str(p['dd']) + '_epoch' + str(p['load_epoch'])
    print('Loading model ...')
    with open(m_path, 'rb') as f:
        loaded_objects = cPickle.load(f)
    model.load_params(loaded_objects)

    print("\tPredicting ...")
    model.update_trained_items()
    model.update_trained_dists()
    all_sus = []
    for start_end in starts_ends_tes:
        [_, sub_all_sus] = model.predict(start_end)
        all_sus.extend(sub_all_sus)

    np.save('./Lmdd/' + p['dataset'] + '_size' + str(p['latent_size']) + '_UD' + str(p['UD']) + '_dd' + str(p['dd']) + '_epoch' + str(p['load_epoch']) + 'last1',
            all_sus)


@exe_time
def main():
    pas = Params()
    if pas.p['mode'] == 's':
        cal_s(pas)
    else:
        train_valid_or_test(pas)


if '__main__' == __name__:
    main()
