#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from math import sin, cos, sqrt, asin
import numpy as np
import pandas as pd
import random
from collections import defaultdict
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
    c = (1.0 - cos(a)) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos(b)) / 2     # 二者等价，但这个更快。
    dist = d * asin(sqrt(c))            # 球面上的弧线距离(km)
    return dist


def load_data(dataset, mode, split):
    """
    加载购买记录文件，生成数据。
    """
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

    # 选取训练集、验证集(测试集)，并对test去重。不管是valid还是test模式，统一用train，test表示。
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_pois, tes_pois = [], []
    for upois, ucods in zip(all_user_pois, all_user_cods):
        left, right = upois[:split], [upois[split]]   # 预测最后一个。此时right只有一个idx，加[]变成list。
        # 保存
        tra_pois.append(left)
        tes_pois.append(right)

    # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名。
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))    # 将poi转换为[0, n)标号。
    tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
    tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
    tra_last_poi = [utra[-1] for utra in tra_pois]          # test时用，基于这个计算对最后一个poi的喜好程度。
    # 根据别名对应关系，更新poi-坐标的表示
    cordi_new = dict()
    for poi in poi_cordi.keys():
        cordi_new[aliases_dict[poi]] = poi_cordi[poi]       # 将poi和坐标转换为：poi的[0, n)标号、坐标。
    pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

    return [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), tra_last_poi]


def fun_data_buys_masks(all_usr_pois, item_tail):
    # 将train/test中序列补全为最大长度，补的idx值=item_num. 为了能在theano里对其进行shared。
    # tail, 添加的。商品索引是0~item_num-1，所以该值[item_num]是没有对应商品实物的。
    # TIP: 把超过最大时间的都改成最大时间
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks


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


def fun_acquire_neighbors_for_each_poi(pois_cordis, max_dist):
    """
    计算：每个poi和所有pois的距离，并选出小于截断距离内的作为邻居。
    :param pois_cordis: 每个poi的idx及其坐标
    :param max_dist:    截断距离
    :return:            dict，key=poi idx, val=set(截断距离内的pois)
    """
    def fun_poi_to_all_pois(poi_idx):
        """
        计算poi和所有poi的距离, 并选取截断距离内的作为邻居。
        :param poi_idx: 每个poi idx
        """
        poi_idx = poi_idx[0]
        poi_cor = pois_cordis[poi_idx]
        all_dists = []
        for each_cor in pois_cordis:
            dist = cal_dis(poi_cor[0], poi_cor[1], each_cor[0], each_cor[1])
            all_dists.append(dist)
        all_dists = np.asarray(all_dists) <= max_dist
        all_neibs = np.nonzero(all_dists)[0]
        poi_neighbors[poi_idx] = list(set(all_neibs) - {poi_idx})     # 邻居不包含自己。
        return [1]

    poi_neighbors = defaultdict(list)
    # 获得邻居
    _ = np.apply_along_axis(    # 该函数需要有返回值，不然在ubuntu下可能会报错。
        func1d=fun_poi_to_all_pois,
        axis=1,
        arr=np.arange(len(pois_cordis))[:, np.newaxis])     # shape=(n_item, 1)
    return poi_neighbors


def fun_acquire_negs_tra(tra_pois, all_pois_neighbors):
    """
    根据每个poi邻居所组成的字典，顺序检索tra，从字典里取出某poi的邻居它自己的负样本。
    """
    # item_num = len(all_pois_neighbors)
    all_negs = []
    for utras in tra_pois:
        unegs = []
        for i, utra in enumerate(utras):
            negs = all_pois_neighbors[utra]
            # if not negs:      # 如果没有邻居，就随机选一个。不写这个了，因为截断距离内，肯定有不少的。
            #     negs = [random.randint(0, item_num - 1)]
            unegs.append(negs)
        all_negs.append(unegs)
    return all_negs


@exe_time  # 放到待调用函数的定义的上一行
def main():
    pass


if '__main__' == __name__:
    main()
