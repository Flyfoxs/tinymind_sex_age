import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import lru_cache
import numpy as np



def get_brand():
    brand = pd.read_csv('input/deviceid_brand.tsv', sep='\t', header=None)

    brand.columns = ['device', 'brand', 'phone_type']
    return brand


def get_package(limit=None):
    cache_file = './output/deviceid_package.tsv'
    if os.path.exists(cache_file):
        print('load package from cache:%s' % cache_file)
        return pd.read_csv(cache_file, sep=',')
    else:
        tmp = pd.read_csv('input/deviceid_packages.tsv', sep='\t', header=None, nrows=limit)
        tmp.columns = ['device', 'package_list']

        tmp = tmp[tmp.device.isin(get_test().iloc[:, 0])]

        package_list_all = frozenset.union(*tmp.iloc[:, 1].apply(lambda v: frozenset(v.split(','))))

        print(len(package_list_all))

        i = 1
        for package in package_list_all:
            i += 1
            print(f'{i}/{len(package_list_all)}')
            tmp[package] = tmp.apply(lambda _: int(package in _.package_list), axis=1)

        tmp.to_csv('./output/deviceid_package.tsv', index=False)

        return tmp


def get_package_label():
    package = pd.read_csv('input/package_label.tsv', sep='\t', header=None, )
    package.columns = ['package', 'type', 'sub_type']
    return package


def get_train():
    pass


def get_test():
    test = pd.read_csv('input/deviceid_test.tsv', sep='\t', header=None)
    return test



def get_test():
    test = pd.read_csv('input/deviceid_test.tsv', sep='\t',header=None)
    return test


def extend_cols(tmp):
    package = get_package_label()
    tmp = tmp.merge(package, how='left')

    brand = get_brand()
    tmp = tmp.merge(brand, how='left')

    return tmp


def split_days(tmp):
    max_duration = 60
    # 超长记录,截取后面的数据
    tmp.loc[tmp.duration > max_duration, 'start'] = \
        tmp.loc[tmp.duration > max_duration, 'close'].dt.date - pd.DateOffset(max_duration)
    tmp.loc[tmp.duration > max_duration, 'duration'] = max_duration

    # 创建新记录,截取后面的时间段
    tmp_new = tmp[tmp.duration > 1]
    tmp_new.start = tmp_new.start.dt.date + pd.DateOffset(1)
    tmp.start = tmp.start.astype('datetime64[ns]')
    tmp_new.duration = tmp_new.duration - 1

    # 旧记录,保留前面的时间段
    tmp.close = tmp.start.dt.date + pd.DateOffset(1)
    tmp.duration = 0

    # print(len(tmp))
    tmp = tmp.append(tmp_new)
    # print(len(tmp))

    tmp.duration = (tmp.close - tmp.start) / np.timedelta64(1, 'D')
    # tmp['duration'] = (tmp.close - tmp.start).apply(lambda value: value.days)

    tmp = tmp.sort_values('duration', ascending=False)

    tmp.reset_index(drop=True, inplace=True)

    return tmp, tmp_new


def get_start_closed(type='long'):
    if type == 'long':
        tmp = pd.read_csv('./output/tem_long_duration.csv', parse_dates=['start', 'close'])
        # tmp = tmp[tmp.device == '225f189c7c214711d483eb3e55743e73']
        # tmp.groupby('device').agg({'duration':['count', 'sum']}) .sort_values(('duration','count'), ascending=False)
        del tmp['start_d']
        del tmp['close_d']

        tmp[tmp.start.dt.date != tmp.close.dt.date].duration = 0

        tmp['duration'] = tmp.close - tmp.start

        # tmp[tmp.duration>17]
        # tmp.duration = tmp.duration.astype('timedelta64[D]')
        tmp.duration = tmp.duration / np.timedelta64(1, 'D')
        return tmp

    if type == 'full':
        limit = None
    else:
        limit = 10000

    start_close = pd.read_csv('input/deviceid_package_start_close.tsv', sep='\t',
                              # index_col=0 ,
                              nrows=limit,
                              header=None )

    start_close.columns = ['device', 'package', 'start_t', 'close_t']

    # start_close.index.name = 'device'

    start_close['start'] = pd.to_datetime(start_close.iloc[:, 2], unit='ms')
    start_close['close'] = pd.to_datetime(start_close.iloc[:, 3], unit='ms')

    #     start_close['start_d'] = pd.to_datetime(start_close.iloc[:,2], unit='ms').dt.date
    #     start_close['close_d'] = pd.to_datetime(start_close.iloc[:,3], unit='ms').dt.date


    # start_close.groupby('device')[['package']].count()
    start_close['duration'] = start_close.close - start_close.start

    start_close.duration = start_close.duration / np.timedelta64(1, 'D')

    return start_close

    # package = get_package()
    # package[package['package']=='225f189c7c214711d483eb3e55743e73']