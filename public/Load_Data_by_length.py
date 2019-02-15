#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from math import cos, sqrt, asin
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


def cal_dis(lat1, lon1, lat2, lon2, dd, dist_num):
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

    interval = int(dist * 1000 / dd)    # 该距离落在哪个距离间隔区间里。
    interval = min(interval, dist_num)
    # 间隔区间范围是[0, 379+1]。即额外添加一个idx=380, 表示两点间距>=38km。
    # 对应的生成分析计算出来的各区间概率，也添加一个位置来表示1520的概率，就是0.
    return interval


def load_data(dataset, mode, split, dd, dist_num):
    """
    加载购买记录文件，生成数据。
    dd = 25m, dt = 60min,
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
    print('\tdistance interval     = [0, {val}]'.format(val=dist_num))

    # 选取训练集、验证集(测试集)，并对test去重。不管是valid还是test模式，统一用train，test表示。
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_pois, tes_pois = [], []
    tra_dist, tes_dist = [], []                 # idx0 = max_dist, idx1 = 0/1的间距并划分到距离区间里。
    for upois, ucods in zip(all_user_pois, all_user_cods):
        left, right = upois[:split], [upois[split]]   # 预测最后一个。此时right只有一个idx，加[]变成list。

        # 两个POI之间距离间隔落在哪个区间。
        dist = []
        for i, cord in enumerate(ucods[1:]):    # 从idx=1和idx=0的距离间隔开始算。
            pre = ucods[i]
            dist.append(cal_dis(cord[0], cord[1], pre[0], pre[1], dd, dist_num))
        dist = [dist_num] + dist                # idx=0的距离间隔，就用最大的。
        dist_lf, dist_rt = dist[:split], [dist[split]]

        # 保存
        tra_pois.append(left)
        tes_pois.append(right)
        tra_dist.append(dist_lf)
        tes_dist.append(dist_rt)

    # # 去重后的基本信息。只预测最后一个，这个用不到。
    # all_trans = []
    # for utra, utes in zip(tra_pois, tes_pois):
    #     all_trans.extend(utra)
    #     all_trans.extend(utes)
    # tran_num, user_num, item_num = len(all_trans), len(tra_pois), len(set(all_trans))
    # temp = tra_dist
    # temp.extend(tes_dist)
    # all_dists = [item for upois in temp for item in upois]
    # print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    # print('\tavg. user poi:          = {val}'.format(val=1.0 * tran_num / user_num))
    # print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))
    # print('\tdistance interval     = [0, {val}]'.format(val=max_dist))

    # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名。
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))    # 将poi转换为[0, n)标号。
    tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
    tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
    # 根据别名对应关系，更新poi-坐标的表示，以list表示。
    cordi_new = dict()
    for poi in poi_cordi.keys():
        cordi_new[aliases_dict[poi]] = poi_cordi[poi]       # 将poi和坐标转换为：poi的[0, n)标号、坐标。
    pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

    return [(user_num, item_num), pois_cordis, (tra_pois, tes_pois), (tra_dist, tes_dist)]


def fun_data_buys_masks(all_usr_pois, all_usr_dist, item_tail, dist_tail):
    # 将train/test中序列补全为最大长度，补的idx值=item_num. 为了能在theano里对其进行shared。
    # tail, 添加的。商品索引是0~item_num-1，所以该值[item_num]是没有对应商品实物的。
    # TIP: 把超过最大时间的都改成最大时间
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_dist = [udist + dist_tail * (len_max - le) for udist, le in zip(all_usr_dist, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
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


def fun_compute_dist_neg(tra_buys_masks, tra_masks, tra_buys_neg_masks, pois_cordis, dd, dist_num):
    """
    用户序列里已计算出了相邻poi(t-1, t)的距离间隔，现在需要计算负样本(t)与正样本(t-1)的距离间隔。
    """
    tra_dist_neg_masks = []
    for upois, umasks, upois_neg in zip(tra_buys_masks, tra_masks, tra_buys_neg_masks):
        # 两个POI之间距离间隔落在哪个区间。
        dist = []
        for i in range(1, sum(umasks)):     # 采用mask，防止坐标字典检索到补全的那个poi，因为它没有坐标。
            pre = pois_cordis[upois[i-1]]
            cur_neg = pois_cordis[upois_neg[i]]
            dist.append(cal_dis(cur_neg[0], cur_neg[1], pre[0], pre[1], dd, dist_num))
        # idx=0的距离间隔，就用最大的。后边补全的那些，也用最大的。
        dist = [dist_num] + dist + [dist_num] * (len(upois) - sum(umasks))
        tra_dist_neg_masks.append(dist)
    return tra_dist_neg_masks


def fun_compute_distance(tra_pois_masks, tra_masks, pois_cordis, dd, dist_num):
    """
    计算：每个user最后一个poi和all pois的距离落在哪个区间里。
    :param tra_pois_masks:  用于获得tra的最后一个poi
    :param tra_masks:       用于找到最后一个poi在该usr序列的哪个位置。
    :param pois_cordis:     各个poi的坐标。按poi的idx顺序排列的list。
    :param dd:              25m，距离间隔。
    :return:                shape=(n_usr, n_item)的矩阵。
    """
    def fun_poi_to_all_intervals(poi):
        """
        计算last poi和所有poi的距离落在哪个区间里
        :param poi:     每个usr的最后一个poi idx
        """
        last_poi_cordi = pois_cordis[poi[0]]
        all_inters = []
        for cordi in pois_cordis:
            inter = cal_dis(last_poi_cordi[0], last_poi_cordi[1], cordi[0], cordi[1], dd, dist_num)
            all_inters.append(inter)
        return all_inters

    le = len(tra_pois_masks)
    # 获得各usr_tra的最后一个poi idx。
    usrs_last_pois = np.asarray(tra_pois_masks)[
        np.arange(le),
        np.sum(tra_masks, axis=1) - 1]

    # 获得该poi与所有poi的距离并划分到距离间隔区间里。
    usrs_last_poi_to_all_intervals = np.apply_along_axis(
        func1d=fun_poi_to_all_intervals,
        axis=1,
        arr=usrs_last_pois[:, np.newaxis])  # shape=(n_item, 1)
    return usrs_last_poi_to_all_intervals


def fun_acquire_prob(all_sus, ulptai, dist_num):
    """
    把一个usr对所有间隔区间的概率，按照usr的last poi对all pois的间隔区间重新索引一遍。
    """
    def fun_uprob_uinterval(prob_inter):
        # prob.shape=(n_usr, 380), inter.shape=(n_usr, n_item)
        # 距离区间超过38km，概率则是0，就是不推荐这个距离以外的。
        prob, interval, mask = prob_inter       # 各区间概率(380,)。all items落在哪个区间(5528,)、且区间>=1520的标为了0.
        usr_probs_to_all_pois = prob[interval]  # all items落在哪个距离区间，就给它分配该区间的概率。
        usr_probs_to_all_pois *= mask           # 再通过mask把>=1520的区间概率直接置为0
        return usr_probs_to_all_pois

    probs_mask = np.asarray(ulptai) < dist_num  # 反之如果>=380，也就是38km，那些太远，不做推荐，mask里会标为0。
    usrs_probs_to_all_pois = np.apply_along_axis(
        func1d=fun_uprob_uinterval,
        axis=1,
        arr=np.array(zip(all_sus, ulptai, probs_mask)))
    return usrs_probs_to_all_pois


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... load the dataset, and  no need to set shared.')


if '__main__' == __name__:
    main()
