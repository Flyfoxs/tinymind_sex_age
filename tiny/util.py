import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import lru_cache
import numpy as np
from utils_.util_log import *
from utils_.util_date import *
from utils_.util_cache_file import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


@timed()
def get_brand():
    brand = pd.read_csv('input/deviceid_brand.tsv', sep='\t', header=None)

    brand.columns = ['device', 'brand', 'phone_type']
    return brand


# Performance issue
@timed()
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


@timed()
def get_package_label():
    package = pd.read_csv('input/package_label.tsv', sep='\t', header=None, )
    package.columns = ['package', 'p_type', 'p_sub_type']
    return package

@timed()
def get_max_week(df):
    """
    到到每个Device使用的最多的那一周
    :param df:
    :return:
    """
    df = df.groupby(['device', 'weekbegin']).agg({'weekday':'nunique','package':'nunique'}).reset_index()

    # .to_frame().reset_index()
    # df.sort_values(['device', 'package'])

    df = df.sort_values(by=['device', 'weekday','package', 'weekbegin'], ascending=False).groupby('device').nth(0)
    df = df.reset_index()
    df.rename({'package':'package_count'}, axis=1, inplace=True)

    return df


@timed()
def get_percent_duration(df, groupby=['device', 'weekday'], prefix=None):
    prefix = groupby[-1] if prefix is None else prefix
    columns = [key for key in df.columns if 'span_' in key]
    gp_map = [(key, 'sum') for key in columns if 'span_' in key]
    gp_map = dict(gp_map)
    # print(gp_map)

    gp_map['package'] = 'nunique'
    gp_map['start_base'] = 'nunique'
    df = df.groupby(groupby).agg(gp_map)
    #     df = df.groupby(groupby).agg({'span_0':'sum','span_1':'sum','span_2':'sum',
    #                                                  'span_3':'sum','package':'nunique' })
    df['total'] = df[[key for key in columns if 'span_' in key]].sum(axis=1)

    for col in columns:
        df[f'{col}_p'] = round(df[col] / df['total'], 3)

    df.rename({'package':'pkg_cnt', 'start_base':'day_cnt'}, axis=1, inplace=True)

    df.columns = [f'{prefix}_{key}' for key in df.columns]
    return df


def get_train():
    columns_1  =[ #sum and percentage
        'total_day_00_0_du',
        'total_day_00_1_du',
        'total_day_00_2_du',
        'total_day_00_3_du',
        '......'
        'total_day_07_3_du',


        'maxmax_day_00_0_du',
        'max_day_00_1_du',
        'max_day_00_2_du',
        'max_day_00_3_du',
        '......'
        'max_day_07_3_du',
    ]

    columns_2 =    [

        #指定时间段,打开pak的统计时长
        'max_package_du_1',  'max_package_du_1_type', 'max_package_du_1_sub_type',
        'max_package_du_2',  'max_package_du_2_type', 'max_package_du_2_sub_type',
        'max_package_du_3',  'max_package_du_3_type', 'max_package_du_3_sub_type',

        #指定时间段,打开pak的次数
        'max_package_cnt_1',  'max_package_cnt_1_type', 'max_package_cnt_1_sub_type',
        'max_package_cnt_2',  'max_package_cnt_2_type', 'max_package_cnt_2_sub_type',
        'max_package_cnt_3',  'max_package_cnt_3_type', 'max_package_cnt_3_sub_type',

        'total_used_package_count', 'per_total_install',
        'weekly_used_package_count', 'per_weekly_install',
        'total_install_package_count',

        'package_top1 ,,,, package_top10',
        'device_brand', 'device_type'

    ]
    pass




def get_test():
    test = pd.read_csv('input/deviceid_test.tsv', sep='\t',header=None)
    return test

@timed()
def extend_cols(tmp):
    package = get_package_label()
    tmp = tmp.merge(package, how='left')

    brand = get_brand()
    tmp = tmp.merge(brand, how='left')

    return tmp

@timed()
@file_cache()
def extend_time(df, span_no=4):
     # mini.start = pd.to_datetime(mini.start)
    #df['dayname'] = df.start.dt.weekday_name

    span_len = 24//span_no

    #把一天分为4个时间段
    for sn in range(0, span_no):
        # df[f'span_{sn}'] = df.apply(lambda row: get_duration(row['start'], row['close'], sn), axis=1)

        print(f'Try to cal for range#{sn}')
        # df['start_base']  = df['start'].dt.date
        df['check_start'] = df['start'].dt.date + pd.DateOffset(hours=span_len * (sn))
        df['check_close'] = df['start'].dt.date + pd.DateOffset(hours=span_len * (sn + 1))

        df['merge_begin'] = df[['check_start', 'start']].max(axis=1)
        df['merge_end'] = df[['check_close', 'close']].min(axis=1)

        df[f'span_{sn}'] = (df['merge_end'] - df['merge_begin']) / np.timedelta64(1, 's')

        df[f'span_{sn}'][df[f'span_{sn}'] <= 0] = 0
        df

    df.drop(columns = ['check_start', 'check_close', 'merge_begin','merge_end'], inplace=True)

    return df

def extend_percent(df):
    total = get_percent_duration(df, ['device'], 'total')

    max_week = get_max_week(df)

    merge = df.merge(max_week, on=['device', 'weekbegin'])

    max = get_percent_duration(merge, ['device'], prefix='max')

    return pd.concat( [total, max], axis=1 )


@file_cache()
def split_days_all(tmp):
    max_duration = 31
    # 超长记录,截取后面的数据, 截断为最后N天
    tmp.loc[tmp.duration > max_duration, 'start'] = \
        tmp.loc[tmp.duration > max_duration, 'close'].dt.date - pd.DateOffset(max_duration)
    tmp.loc[tmp.duration > max_duration, 'duration'] = max_duration

    print(f'Out loop: The original Df size is {len(tmp)}')
    tmp = split_days(tmp)
    print(f'Out loop: The new Df size is {len(tmp)}')

    tmp['start_base'] = tmp['start'].dt.date

    tmp.duration = round(tmp.duration, 6)

    tmp = tmp.sort_values(by = ['device','package','start'])

    return tmp

@timed()
def split_days(tmp):
    print(f'The original df#{len(tmp)} before split')

    # 创建新记录,截取后面的时间段
    tmp_new_big = tmp[tmp.duration > 1]
    if len(tmp_new_big) == 0:
        return tmp

    tmp_new_big.start = tmp_new_big.start.dt.date + pd.DateOffset(1)
    tmp_new_big.duration = tmp_new_big.duration - 1

    # 旧记录,保留前面的时间段
    tmp             = tmp[tmp.duration <= 1]
    tmp_new_small   = tmp[tmp.duration > 1]
    tmp_new_small.close    = tmp_new_small.start.dt.date + pd.DateOffset(1)
    tmp_new_small.duration = (tmp_new_small.close - tmp_new_small.start) / np.timedelta64(1, 'D')

    tmp = tmp.append(tmp_new_small)

    tmp_new_big = split_days(tmp_new_big)
    tmp = tmp.append(tmp_new_big)

    #tmp = tmp.sort_values('duration', ascending=False)
    tmp.reset_index(drop=True, inplace=True)

    print(f'The new df#{len(tmp)} after split')

    return tmp

@timed()
@file_cache()
def get_start_closed(type='long'):
    if type == 'long':
        tmp = pd.read_csv('./output/tem_long_duration.csv', parse_dates=['start', 'close'])
        # tmp = tmp[tmp.device == '225f189c7c214711d483eb3e55743e73']
        # tmp.groupby('device').agg({'duration':['count', 'sum']}) .sort_values(('duration','count'), ascending=False)
        del tmp['start_d']
        del tmp['close_d']

        tmp[tmp.start.dt.date != tmp.close.dt.date].duration = 0

        tmp['duration'] = tmp.close - tmp.start

        tmp['weekday'] = tmp.start.dt.weekday
        tmp['weekbegin'] = (tmp['start'] - tmp['start'].dt.weekday.astype('timedelta64[D]')).dt.date

        # tmp[tmp.duration>17]
        # tmp.duration = tmp.duration.astype('timedelta64[D]')
        tmp.duration = tmp.duration / np.timedelta64(1, 'D')
        return tmp
    if type == 'mini':
        tmp = pd.read_csv('./output/mini_start_closed_2.csv', parse_dates=['start', 'close'])
        # tmp = tmp[tmp.device == '225f189c7c214711d483eb3e55743e73']
        # tmp.groupby('device').agg({'duration':['count', 'sum']}) .sort_values(('duration','count'), ascending=False)

        tmp[tmp.start.dt.date != tmp.close.dt.date].duration = 0


        tmp['duration'] = tmp.close - tmp.start
        df = tmp
        df['weekday'] = df.start.dt.weekday
        df['weekbegin'] = (df['start'] - df['start'].dt.weekday.astype('timedelta64[D]')).dt.date

        # tmp[tmp.duration>17]
        # tmp.duration = tmp.duration.astype('timedelta64[D]')
        tmp.duration = tmp.duration / np.timedelta64(1, 'D')
        return tmp

    if type == 'all':
        limit = None
    else:
        limit = type

    start_close = pd.read_csv('input/deviceid_package_start_close.tsv', sep='\t',
                              # index_col=0 ,
                              nrows=limit,
                              header=None )

    start_close.columns = ['device', 'package', 'start_t', 'close_t']

    print(f'Sort the df#{len(start_close)} by device(begin)')
    start_close.sort_values('device', inplace=True)
    print(f'Sort the df by device(end)')

    # start_close.index.name = 'device'


    start_close['start'] = pd.to_datetime(start_close.loc[:, 'start_t'], unit='ms')
    start_close['close'] = pd.to_datetime(start_close.loc[:, 'close_t'], unit='ms')

    len_original = len(start_close)
    start_close = start_close[start_close.start < pd.to_datetime('now')]
    #去除部分异常数据
    print(f'Remove {len_original-len(start_close)} records data')

    # start_close.groupby('device')[['package']].count()
    start_close['duration'] = start_close.close - start_close.start

    #df = start_close
    start_close['weekday'] = start_close.start.dt.weekday
    start_close['weekbegin'] = (start_close['start'] -
                                start_close['start'].dt.weekday.astype('timedelta64[D]')).dt.date

    start_close.duration = start_close.duration / np.timedelta64(1, 's')

    return start_close

def split_df(df, split_no=10, prefix='default'):
    df.reset_index(drop=True, inplace=True)
    begin=0
    end=len(df)

    if len(df)%split_no == 0:
        size = len(df) //split_no
    else:
        size = len(df) // (split_no -1)

    begin_adjust = 0
    end_adjust = 0
    for i in range(0,split_no):
        begin_adjust = end_adjust
        end_adjust = adjust_split_end(df,  size*(i+1))
        file = f'output/{prefix}_{split_no}_{str(i).rjust(2,"0")}_{begin_adjust}_{end_adjust}.csv'
        print(f'Split df to file#{i}:{file}')
        df[begin_adjust : end_adjust].to_csv(file, index=None)

def adjust_split_end(df,  end):
    id = df.iat[end, 0]
    end_adjust = 1 + df[df.iloc[:,0]==id].index.max()
    print(f'Adjust original end from {end} to {end_adjust} for id:{id}')
    return end_adjust



# @timed()
@DeprecationWarning
def get_duration(start, close, sn=0):
    check_point_start = pd.to_datetime(start.date()) + pd.DateOffset(hours=6 * sn)
    check_point_close = pd.to_datetime(start.date()) + pd.DateOffset(hours=6 * (sn + 1))

    # print(check_point_start, start,  close  ,check_point_close )

    # print(min(check_point_close, close), max(check_point_start,start) )

    gap = (min(check_point_close, close) - max(check_point_start, start)) / np.timedelta64(1, 'D')
    return max(gap, 0)

    # package = get_package()
    # package[package['package']=='225f189c7c214711d483eb3e55743e73']