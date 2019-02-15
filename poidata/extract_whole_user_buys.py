import os
import pandas as pd
from pandas import DataFrame
from collections import Counter
import datetime
data_name = 'Foursquare'
# data_name = 'Gowalla'
PATH_FROM = './' + data_name + '/'
PATH_TO = PATH_FROM + 'sequence/'


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


def transform_hist_to_buys(
        hist_file,
        buys_file):
    print('\t*** reformat history data to sequential data ***')

    # dataset format: user_ID	POI_ID	coordinate	checkin_time(hour:min)	date_id
    header = ['user_ID', 'POI_ID', 'coordinate', 'checkin_time', 'date_id']
    hist = pd.read_table(hist_file, sep='\t', names=header)
    hist[['hour', 'minuties']] = hist['checkin_time'].str.split(':', expand=True)
    hist['time'] = pd.to_numeric(hist['date_id']) * 24 * 60 + pd.to_numeric(hist['hour']) * 60 + pd.to_numeric(hist['minuties'])
    hist = hist.drop(hist[hist['POI_ID'] == 'LOC_null'].index)
    gp = hist.groupby('user_ID')
    print('\tthe number of history records:', len(hist))
    print('\tthe number of usrs:           ', len(gp))

    i, length = 0, len(gp)
    check_times, pois_different, u_ids, u_pois, u_times, u_coordinates = [], [], [], [], [], []

    while 1:
        len_hist = len(hist)
        len_usr = len(gp)
        apr = dict()
        for user_id, hist_group in gp:
            ulength = len(hist_group)
            if ulength < 5:
                hist = hist.drop(hist_group.index)
                continue
            for poi in hist_group['POI_ID'].unique():
                try:
                    apr[poi] += 1
                except:
                    apr[poi] = 1
        hist = hist.drop(hist.loc[hist['POI_ID'].apply(lambda x: apr[x] < 5)].index)
        gp = hist.groupby('user_ID')
        if (len_hist == len(hist)) & (len_usr == len(gp)):
            break
        print('\t*** removing unqualified users and pois ***')
        print('\tthe number of history records:', len(hist))
        print('\tthe number of usrs:           ', len(gp))

    for user_id, hist_group in gp:
        i += 1
        if 0 == i % 10000:
            print('\t{v1} / {v2} = {v3}'.format(v1=i, v2=length, v3=1.0 * i / length))
        u_ids.append(user_id)
        g = hist_group.sort_values(by='time')  # sort_values(by=)
        pois, coordinate, times = list(g['POI_ID']), g['coordinate'], list(g['time'])
        check_times.append(len(pois))
        pois_different.append('%0.2f' % (1.0 * len(set(pois)) / len(pois)))
        u_pois.append('/'.join([str(j) for j in pois]))
        u_times.append('/'.join([str(k) for k in times]))
        u_coordinates.append('/'.join([str(l) for l in coordinate]))

    # 准备保存
    cols = ['check_times', 'pois_different', 'u_id', 'u_pois', 'u_times', 'u_coordinates']
    df = DataFrame({cols[0]: check_times,
                    cols[1]: pois_different,
                    cols[2]: u_ids,
                    cols[3]: u_pois,
                    cols[4]: u_times,
                    cols[5]: u_coordinates})
    del check_times, pois_different, u_ids, u_pois, u_times, u_coordinates
    df.sort_values(by=['check_times', 'pois_different'], ascending=False, inplace=True)
    df.to_csv(buys_file, sep=' ', index=False, columns=cols)
    print('\t*** reformat history data to sequential data - the end ***')


def buys_details(
        buys_file,
        u_cou_file,
        i_cou_file,
        u_cou_cou_file,
        i_cou_cou_file):

    pois = pd.read_csv(buys_file, sep=' ')

    # 基本计数信息统计: 三个数
    item_ids = [i for poi in pois['u_pois'] for i in str(poi).split('/')]
    a, b, c = len(pois), len(item_ids), len(set(item_ids))
    print('\tn_user：{val}'.format(val=a))           # 619332
    print('\tn_checkin：{val}'.format(val=b))       # 10245725
    print('\tn_poi：{val}'.format(val=c))  # 449970
    print('\tsparsity：{val}'.format(val=1.0 - 1.0 * b / (a * c)))
    print('\taverage checkin of each user：{val}'.format(val=1.0 * b / a))
    print('\taverage checkin of each item：{val}'.format(val=1.0 * b / c))

    # 提取 _cou, _cou_cou
    u_cou = pois[['check_times', 'u_id']]
    u_cou_cou = dict(Counter(u_cou['check_times'].values))
    i_cou = dict(Counter(item_ids))
    i_cou_cou = dict(Counter(i_cou.values()))
    # 保存 _cou
    i_cou = DataFrame(data={'item_id': list(i_cou.keys()), 'check_times': list(i_cou.values())})
    i_cou.sort_values(by=['check_times'], ascending=False, inplace=True)
    u_cou.to_csv(u_cou_file, sep=' ', index=False)
    i_cou.to_csv(i_cou_file, sep=' ', index=False)
    # 保存 _cou_cou
    u_cou_cou = DataFrame(data={'check_times': list(u_cou_cou.keys()), 'nums': list(u_cou_cou.values())})
    u_cou_cou['times*nums'] = u_cou_cou['check_times'] * u_cou_cou['nums']
    i_cou_cou = DataFrame(data={'check_times': list(i_cou_cou.keys()), 'nums': list(i_cou_cou.values())})
    i_cou_cou['times*nums'] = i_cou_cou['check_times'] * i_cou_cou['nums']
    u_cou_cou.sort_values(by=['check_times'], ascending=False, inplace=True)
    i_cou_cou.sort_values(by=['check_times'], ascending=False, inplace=True)
    u_cou_cou.to_csv(u_cou_cou_file, sep=' ', index=False)
    i_cou_cou.to_csv(i_cou_cou_file, sep=' ', index=False)


def process_whole():
    global PATH_FROM, PATH_TO
    pafr, pato = PATH_FROM, PATH_TO

    hf = os.path.join(pafr, data_name + '.txt')
    pf = os.path.join(pato, data_name + '.txt')
    transform_hist_to_buys(hist_file=hf, buys_file=pf)

    ucf = os.path.join(pato, 'history_user_count.txt')
    icf = os.path.join(pato, 'history_poi_count.txt')
    uccf = os.path.join(pato, 'history_user_count_count.txt')
    iccf = os.path.join(pato, 'history_poi_count_count.txt')
    buys_details(buys_file=pf,
                 u_cou_file=ucf,
                 i_cou_file=icf,
                 u_cou_cou_file=uccf,
                 i_cou_cou_file=iccf)

@exe_time
def main():
    process_whole()


if __name__ == '__main__':
    main()
