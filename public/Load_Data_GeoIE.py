#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 2018/10/6 11:50 AM

@author: Tangrizzly
"""

from __future__ import print_function
import time
import numpy as np
import pandas as pd
import random
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


def cal_dis(lat1, lon1, lat2, lon2):
    """
    Haversine公式: 计算两个latitude-longitude点之间的距离. [ http://www.cnblogs.com/softidea/p/6925673.html ]
    二倍角公式：cos(2a) = 1 - 2sin(a)sin(a)，即sin(a/2)*sin(a/2) = (1 - cos(a))/2
    dd = 25m，距离间隔。
    """
    d = 12742                           # 地球的平均直径。
    p = 0.017453292519943295            # math.pi / 180, x*p由度转换为弧度。
    a = (lat1 - lat2) * p
    b = (lon1 - lon2) * p
    # c = pow(sin(a / 2), 2) + cos(lat1 * p) * cos(lat2 * p) * pow(sin(b / 2), 2)     # a/b别混了。
    c = (1.0 - np.cos(a)) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1.0 - np.cos(b)) / 2     # 二者等价，但这个更快。
    dist = d * np.arcsin(np.sqrt(c))            # 球面上的弧线距离(km)

    return dist


def load_data(dataset, mode, split):
    # 用户购买历史记录，原纪录. 嵌套列表, 元素为一个用户的购买记录(小列表)
    print('Original data ...')
    pois = pd.read_csv(dataset, sep=' ')
    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_user_cods = [[i.split(',') for i in upois.split('/')] for upois in pois['u_coordinates']]       # string
    all_user_cods = [[[float(ucod[0]), float(ucod[1])] for ucod in ucods] for ucods in all_user_cods]   # float
    all_trans = [item for upois in all_user_pois for item in upois]
    all_cordi = [ucod for ucods in all_user_cods for ucod in ucods]
    poi_cordi = dict(zip(all_trans, all_cordi))  # 每个poi都有一个对应的的坐标。
    tran_num, user_num, item_num = len(all_trans), len(all_user_pois), len(set(all_trans))
    print('\tusers, items, trans:  = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user check:      = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. poi checked:     = {val}'.format(val=1.0 * tran_num / item_num))

    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))    # 将poi转换为[0, n)标号。
    all_user_pois = [[aliases_dict[i] for i in u] for u in all_user_pois]
    # 根据别名对应关系，更新poi-坐标的表示，以list表示。
    cordi_new = dict()
    for poi in poi_cordi.keys():
        cordi_new[aliases_dict[poi]] = poi_cordi[poi]       # 将poi和坐标转换为：poi的[0, n)标号、坐标。
    pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

    # 选取训练集、验证集(测试集)，并对test去重。不管是valid还是test模式，统一用train，test表示。
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_count, tra_pois, tra_targ, tes_pois, tes_targ = [], [], [], [], []
    tra_dist, tes_dist = [], []
    for upois, ucods in zip(all_user_pois, all_user_cods):
        left, right = upois[:split], [upois[split]]
        count_dict = dict(zip(*np.unique(left, return_counts=True)))
        count = [count_dict[l] for l in left]
        dist = []
        ucods = np.asarray(ucods)
        for i in range(1, len(upois)):
            dist.append(cal_dis(ucods[:i][:, 0], ucods[:i][:, 1], ucods[i][0], ucods[i][1]).tolist())

        tra_count.append(count)
        tra_pois.append(left)
        tes_pois.append(right)
        tra_dist.append(dist[:split])
        tes_dist.append(dist[split])

    return [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), (tra_dist, tes_dist), tra_count]


def fun_data_buys_masks(all_usr_pois, all_usr_dist, item_tail, dist_tail, tra_count=None):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_dist = [udist + dist_tail * (len_max - le) for udist, le in zip(all_usr_dist, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    if tra_count is not None:
        us_count = [ucount + [0] * (len_max - le) for ucount, le in zip(tra_count, us_lens)]
        return us_pois, us_dist, us_msks, us_count
    else:
        return us_pois, us_dist, us_msks


def fun_random_neg_masks_tra(item_num, tras_mask):
    """
    从num件商品里随机抽取与每个用户的购买序列等长且不在已购买商品里的标号。后边补全的负样本用虚拟商品[item_num]
    """
    us_negs = []
    for utra in tras_mask:     # 每条用户序列
        unegs = []
        for i, e in enumerate(utra):
            if item_num == e:                        # 表示该购买以及之后的，都是用虚拟商品[item_num]来补全的
                unegs += [item_num] * (len(utra) - i)   # 购买序列里对应补全商品的负样本也用补全商品表示
                break
            j = random.randint(0, item_num - 1)      # 负样本在商品矩阵里的标号
            while j in utra:                     # 抽到的不是用户训练集里的。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_random_neg_masks_tes(item_num, tras_mask, tess_mask):
    """
    从num件商品里随机抽取与测试序列等长且不在训练序列、也不再测试序列里的标号
    """
    us_negs = []
    for utra, utes in zip(tras_mask, tess_mask):
        unegs = []
        for i, e in enumerate(utes):
            if item_num == e:                   # 尾部补全mask
                unegs += [item_num] * (len(utes) - i)
                break
            j = random.randint(0, item_num - 1)
            while j in utra or j in utes:         # 不在训练序列，也不在预测序列里。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis):
    pdist, qdist, m = [], [], []
    pois_cordis = np.asarray(pois_cordis)
    for p, q, mask in zip(tra_buys_masks, tra_buys_neg_masks, tra_masks):
        ipdist, iqdist, im = [], [], []
        len = sum(mask)
        for i in range(1, len):
            im.append([1] * i + [0] * (len - i - 1))
            ipdist.append(cal_dis(pois_cordis[p[:i]][:, 0], pois_cordis[p[:i]][:, 1], pois_cordis[p[i]][0], pois_cordis[p[i]][1]).tolist() + (len - i - 1) * [0])
            iqdist.append(cal_dis(pois_cordis[p[:i]][:, 0], pois_cordis[p[:i]][:, 1], pois_cordis[q[i]][0], pois_cordis[q[i]][1]).tolist() + (len - i - 1) * [0])
        pdist.append(ipdist)
        qdist.append(iqdist)
        m.append(im)
    return pdist, qdist, m


def fun_compute_distance(tra_pois, tra_masks, pois_cordis, test_batch):
    pois_cordis = np.asarray(pois_cordis)
    tra_masks = np.asarray(tra_masks)

    def fun_poi_to_all_intervals(upois):
        udists = []
        for upoi in upois:
            udist = cal_dis(pois_cordis[upoi][0], pois_cordis[upoi][1], pois_cordis[:, 0], pois_cordis[:, 1])
            udists.append(udist)
        return udists

    n = len(tra_pois)
    dists = []
    for i in range(n / test_batch):
        max_len = max(np.sum(tra_masks[i * test_batch: (i + 1) * test_batch], 1))
        for j in range(i * test_batch, (i + 1) * test_batch):
            dist = fun_poi_to_all_intervals(tra_pois[j]) + np.zeros((max_len - len(tra_pois[j]), len(pois_cordis))).tolist()
            dists.append(dist)
    if n % test_batch != 0:
        max_len = max(np.sum(tra_masks[- (n % test_batch):], 1))
        for j in range(n - (n % test_batch), n):
            dist = fun_poi_to_all_intervals(tra_pois[j]) + np.zeros((max_len - len(tra_pois[j]), len(pois_cordis))).tolist()
            dists.append(dist)
    return dists


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... load the dataset, and  no need to set shared.')


if '__main__' == __name__:
    main()
