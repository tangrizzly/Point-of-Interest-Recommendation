#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 23/03/2018 10:15 AM

@author: Tangrizzly
"""

from __future__ import print_function

import time
import pandas as pd
import numpy as np

__docformat__ = 'restructedtext en'

PATH = '../poidata/Gowalla/sequence/'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


def rad(x):
    return np.multiply(x, np.pi) / 180.0


def preprocess(dataset, mode, split, time_threshold, region_threshold):
    print('Original data ...')
    pois = pd.read_csv(dataset, sep=' ')
    all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
    all_user_times = [[int(i) for i in upois.split('/')] for upois in pois['u_times']]
    all_user_cordi = [[i.split(',') for i in upois.split('/')] for upois in pois['u_coordinates']]
    all_trans = [item for upois in all_user_pois for item in upois]
    all_cordi = [cordi for ucordi in all_user_cordi for cordi in ucordi]
    cordi = dict(zip(all_trans, all_cordi))
    tran_num, user_num, item_num = len(all_trans), len(all_user_pois), len(set(all_trans))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user check:        = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. poi checked:       = {val}'.format(val=1.0 * tran_num / item_num))

    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    tra_target, tes_target = [], []
    tra_context, tes_context = [], []
    for i in range(len(pois)):
        utarget = all_user_pois[i][:]
        utime = all_user_times[i][:]
        le = len(utarget)
        split0, split1, split2 = 0, int(le * split[1]-1), int(le * split[1])
        utarget.reverse()
        utime.reverse()
        ucontext = []
        for j in range(0, le):
            flag = utime[j]
            ucontext.append([])
            for k in range(j+1, le):
                if flag - utime[k] < time_threshold:
                    ucontext[j].append(utarget[k])
                else:
                    ucontext[j].reverse()
                    break
        utarget.reverse()
        ucontext.reverse()
        tra_target.append(utarget[split0:split1])
        tes_target.append(utarget[split1:split2])
        tra_context.append(ucontext[split0:split1])
        tes_context.append(ucontext[split1:split2])

    # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名
    print('Use aliases to represent pois ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))
    tra_target = [[aliases_dict[i] for i in utra] for utra in tra_target]
    tes_target = [[aliases_dict[i] for i in utes] for utes in tes_target]
    tra_context = [[[aliases_dict[j] for j in i] for i in utra] for utra in tra_context]
    tes_context = [[[aliases_dict[j] for j in i] for i in utes] for utes in tes_context]
    cordi_new = dict()
    for i in cordi:
        cordi_new[aliases_dict[i]] = cordi[i]
    location = []
    for i in range(len(cordi)):
        location.append(cordi_new[i])
    location = np.asarray(location, dtype='float')
    del cordi, cordi_new

    print('\tBuilding huffman tree ... ')
    tree = Treenode(min(location[:, 0]), max(location[:, 0]), max(location[:, 1]), min(location[:, 1]),
                    region_threshold)
    tree.build()

    routes, lrs, probs, uniqs = [], [], [], []
    for lat, lon in location:
        parea, proutes, plrs, puniq = [], [], [], []
        p_n = [(lat - 0.5 * tree.theta, lon - 0.5 * tree.theta),
               (lat - 0.5 * tree.theta, lon + 0.5 * tree.theta),
               (lat + 0.5 * tree.theta, lon - 0.5 * tree.theta),
               (lat + 0.5 * tree.theta, lon + 0.5 * tree.theta)]
        for p in p_n:
            node, route, lr = tree.route(p)
            if route not in proutes:
                parea.append(node.overlap((lat, lon)))
            else:
                parea.append(0)
            proutes.append(route)
            plrs.append(lr)

        prob = np.asarray(parea) / np.sum(parea)
        routes.append(proutes)
        lrs.append(plrs)
        probs.append(prob)

    # mask
    routes.append(routes[0])
    lrs.append(lrs[0])
    probs.append([0, 0, 0, 0])

    np.save(PATH + 'tra_target', tra_target)
    np.save(PATH + 'tra_context', tra_context)
    np.save(PATH + 'tes_target', tes_target)
    np.save(PATH + 'tes_context', tes_context)
    np.save(PATH + 'location', [[user_num, item_num, Treenode.count+1], location])
    np.save(PATH + 'probs', probs)
    np.save(PATH + 'routes', routes)
    np.save(PATH + 'lrs', lrs)

    # return [(user_num, item_num, location), (tra_target, tes_target), (tra_context, tes_context)]


def load_data():
    Path = './poidata/Foursquare/sequence/'
    tra_target = np.load(Path + 'tra_target.npy')
    tra_context = np.load(Path + 'tra_context.npy')
    tes_target = np.load(Path + 'tes_target.npy')
    tes_context = np.load(Path + 'tes_context.npy')
    [user_num, item_num, node_count], location = np.load(Path + 'location.npy')
    probs = np.load(Path + 'probs.npy')
    routes = np.load(Path + 'routes.npy')
    lrs = np.load(Path + 'lrs.npy')
    # return [(user_num, item_num, location), (tra_target, tes_target), (tra_context, tes_context),
    # (areas, routes, lrs)]
    return [(user_num, item_num, node_count), (tra_target, tes_target), (tra_context, tes_context),
            (probs, routes, lrs)]


def fun_data_masks(targets, contexts, item_tail):
    us_lens = [len(upois) for upois in targets]
    len_max = max(us_lens)
    if not len_max == 1:
        us_target = [upois + item_tail * (len_max - le) for upois, le in zip(targets, us_lens)]
    else:
        us_target = targets
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]

    us_accum_lens = []
    pre = 0
    for lens in us_lens:
        aft = pre + lens
        us_accum_lens.append([pre, aft])
        pre = aft

    context_flatten = [j for i in contexts for j in i]
    us_lens_cot = [len(upois) for upois in context_flatten]
    len_max_cot = max(us_lens_cot)
    us_context = [upois + item_tail * (len_max_cot - le) for upois, le in zip(context_flatten, us_lens_cot)]
    us_msks_cot = [[1] * le + [0] * (len_max_cot - le) for le in us_lens_cot]
    return us_target, us_context, us_msks, us_msks_cot, us_accum_lens


class Treenode:
    count = -1
    theta = None
    leaves = []

    def __init__(self, left, right, up, down, theta=None):
        self.splt = None
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.left_child = None
        self.right_child = None
        Treenode.count += 1
        self.count = Treenode.count
        if not theta is None:
            Treenode.theta = theta

    def overlap(self, (latitude, longitude)):
        left = max(latitude - 0.5 * Treenode.theta, self.left)
        right = min(latitude + 0.5 * Treenode.theta, self.right)
        up = min(longitude + 0.5 * Treenode.theta, self.up)
        down = max(longitude - 0.5 * Treenode.theta, self.down)
        return (right - left) * (up - down)

    def build(self):
        if (self.right - self.left) > (self.up - self.down):
            # 纵向划分
            if (self.right - self.left) > 2 * Treenode.theta:
                self.splt = 1
                self.left_child = Treenode(self.left, (self.left + self.right) / 2, self.up, self.down)
                self.right_child = Treenode((self.right + self.left) / 2, self.right, self.up, self.down)
                self.left_child.build()
                self.right_child.build()
            else:
                Treenode.leaves.append(self)
        else:
            # 横向划分
            if (self.up - self.down) > 2 * Treenode.theta:
                self.splt = 0
                self.left_child = Treenode(self.left, self.right, self.up, (self.up + self.down) / 2)
                self.right_child = Treenode(self.left, self.right, (self.up + self.down) / 2, self.down)
                self.left_child.build()
                self.right_child.build()
            else:
                Treenode.leaves.append(self)

    def route(self, (latitude, longitude)):
        prev_route = None
        if self.splt is None:
                prev_route = [self.count]
                prev_lr = [1]
                return self, prev_route, prev_lr

        if self.splt == 1:
            # left is true and right is false
            if self.left_child.right < latitude:
                node, prev_route, prev_lr = self.right_child.route((latitude, longitude))
                prev_lr.append(-1)
            else:
                node, prev_route, prev_lr = self.left_child.route((latitude, longitude))
                prev_lr.append(1)
        else:
            # up is true and down is false
            if self.left_child.down > longitude:
                node, prev_route, prev_lr = self.right_child.route((latitude, longitude))
                prev_lr.append(-1)
            else:
                node, prev_route, prev_lr = self.left_child.route((latitude, longitude))
                prev_lr.append(1)
        prev_route.append(self.count)
        return node, prev_route, prev_lr


@exe_time
def main():
    preprocess(PATH + 'Gowalla.txt', 'test', [0.8, 1.0], 360, 0.1)
    print('... load the dataset, and  no need to set shared.')

if '__main__' == __name__:
    main()
