import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import lru_cache
import numpy as np
from utils_.util_log import *
from utils_.util_date import *
from utils_.util_cache_file import *
from pandas.tseries.offsets import Week
from utils_.util_pandas import *
try:
    from tiny.conf import *
except :
    mini=True
    version=1


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


@timed()
def get_brand():
    brand = pd.read_csv('input/deviceid_brand.tsv', sep='\t', header=None)

    brand.columns = ['device', 'brand', 'phone_type']
    return convert_label_encode(brand, ['device'])

#
# # Performance issue
# @timed()
# def get_package(limit=None):
#     cache_file = './output/deviceid_package.tsv'
#     if os.path.exists(cache_file):
#         print('load package from cache:%s' % cache_file)
#         return pd.read_csv(cache_file, sep=',')
#     else:
#         tmp = pd.read_csv('input/deviceid_packages.tsv', sep='\t', header=None, nrows=limit)
#         tmp.columns = ['device', 'package_list']
#
#         tmp = tmp[tmp.device.isin(get_test().iloc[:, 0])]
#
#         package_list_all = frozenset.union(*tmp.iloc[:, 1].apply(lambda v: frozenset(v.split(','))))
#
#         print(len(package_list_all))
#
#         i = 1
#         for package in package_list_all:
#             i += 1
#             print(f'{i}/{len(package_list_all)}')
#             tmp[package] = tmp.apply(lambda _: int(package in _.package_list), axis=1)
#
#         tmp.to_csv('./output/deviceid_package.tsv', index=False)
#
#         return tmp


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


# @timed()
# def get_percent_duration(df, groupby=['device', 'weekday'], prefix=None, span_no=6):
#     prefix = groupby[-1] if prefix is None else prefix
#     sum_duration = get_sum_duration(df, groupby, prefix)
#
#     sum_duration = reduce_time_span(sum_duration, prefix, span_no)
#
#     for col in [item for item in sum_duration.columns if f'{prefix}_span_' in item]:
#         df[f'{col}_p'] = round(df[col] / df[f'{prefix}_total'], 3)
#
#     return df

def reduce_time_span(df, prefix, span_no=4):
    span_len = 24//span_no
    print(f'columns before reduce:{df.columns}')
    for sn in range(0, span_no):
        for type in ['sum', 'count']:
            col_list = [f'{prefix}_span24_{sn}_{type}' for sn in range(span_len*sn, span_len*(sn+1))]
            df[f'{prefix}_{span_no}span_{sn}_{type}'] = df[col_list].sum(axis=1)
            # col_list.remove(f'{prefix}_span_{sn}')
            df.drop(columns=col_list, inplace=True)
    return df

def get_summary_weekday(df):

    #按照每个星期去统计
    gp = df.groupby(['device', 'weekday']).agg({'package': 'nunique', 'day_duration': 'sum'})
    gp.reset_index(inplace=True)

    gp1 = gp.pivot(index='device', columns='weekday', values='package')
    gp1.columns = [f'package_{col}' for col in gp1.columns]

    gp2 = gp.pivot(index='device', columns='weekday', values='day_duration')
    gp2.columns = [f'duration_{col}' for col in gp2.columns]


    #区分周末和工作日
    df['weekend'] = df.weekday // 5
    gp3 = df.groupby(['device', 'weekend']).agg({'package': 'nunique', 'day_duration': 'sum'})
    gp3.reset_index(inplace=True)

    wk_end1 = gp3.pivot(index='device', columns='weekend', values='package')
    wk_end1.columns = [f'package_wk_{col}' for col in wk_end1.columns]

    wk_end2 = gp3.pivot(index='device', columns='weekend', values='day_duration')
    wk_end2.columns = [f'duration_wk_{col}' for col in wk_end2.columns]

    wk = pd.concat([wk_end1, wk_end2], axis=1)
    wk.head()

    #计算总数
    total = df.groupby(['device']).agg({'package': 'nunique', 'duration': 'sum'})


    merge = pd.concat([gp1, gp2, wk, total], axis=1)

    # 转换为Percentage
    type_ = 'package'
    for col in [col for col in merge.columns if f'{type_}_' in col]:
        # print(col)
        merge[col] = merge[col] / merge[type_]

    type_ = 'duration'
    for col in [col for col in merge.columns if f'{type_}_' in col]:
        # print(col)
        merge[col] = merge[col] / merge[type_]

    return merge


@timed()
#@file_cache()
def get_summary_span24(df, prefix):
    #prefix = groupby[-1] if prefix is None else prefix

    columns = [key for key in df.columns if 'span24_' in key]
    gp_map = [(key, ['sum', 'count']) for key in columns if 'span24_' in key]
    gp_map = dict(gp_map)
    # print(gp_map)

    gp_map['package'] = 'nunique'
    gp_map['start_base'] = ['min','max','nunique']

    df = df.groupby('device').agg(gp_map)

    df['total_sum']   = df[[key for key in df.columns if 'sum' in key]].sum(axis=1)
    df['total_count'] = df[[key for key in df.columns if 'count' in key]].sum(axis=1)

    df.rename({'package': 'pkg', 'start_base': 'day'}, axis=1, inplace=True)

    print(type(df.columns[0]))
    print('_'.join(df.columns[0]))
    print(f'The latest colums:{df.columns}')
    df.columns = [f"sum_{'_'.join(key)}" for key in df.columns]

    print(f'The latest colums:{df.columns}')
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
def extend_brand_pkg(tmp):
    if 'package' in tmp:
        package = get_package_label()
        print(f'input df col:{tmp.columns}')
        print(f'packae col:{package.columns}')
        tmp = tmp.merge(package, how='left')

    brand = get_brand()
    print(f'column list:{tmp.columns}')
    tmp = tmp.merge(brand, how='left')

    return tmp




@timed()
#@file_cache()
def cal_duration_for_span(df, span_no=24):
    # mini.start = pd.to_datetime(mini.start)
    #df['dayname'] = df.start.dt.weekday_name

    span_len = 24//span_no

    df[f'day_duration'] = (df['close'] - df['start']) / np.timedelta64(1, 'D')

    #把一天分为4个时间段
    for sn in range(0, span_no):
        # df[f'span_{sn}'] = df.apply(lambda row: get_duration(row['start'], row['close'], sn), axis=1)

        print(f'Try to cal for range#{sn} with span_len: {span_len}')
        # df['start_base']  = df['start'].dt.date
        df['check_start'] = df['start'].dt.date + pd.DateOffset(hours=span_len * (sn))
        df['check_close'] = df['start'].dt.date + pd.DateOffset(hours=span_len * (sn + 1))

        df['merge_begin'] = df[['check_start', 'start']].max(axis=1)
        df['merge_end'] = df[['check_close', 'close']].min(axis=1)

        df[f'span24_{sn}'] = (df['merge_end'] - df['merge_begin']) / np.timedelta64(1, 'D')

        #去除负值
        df[f'span24_{sn}'][df[f'span24_{sn}'] <= 0] = np.nan
        df

    df.drop(columns = ['check_start', 'check_close', 'merge_begin','merge_end'], inplace=True)
    print(f'Output columns for extend_time is {df.columns}')
    return df

# def extend_sum_duration_df(df, groupby=['device', 'weekday'], prefix=None):
#     total = get_sum_duration(df, ['device'], 'total')
#
#     max_week = get_max_week(df)
#
#     merge = df.merge(max_week, on=['device', 'weekbegin'])
#
#     max = get_sum_duration(merge, ['device'], 'max')
#
#     return pd.concat( [total, max], axis=1 ).reset_index()


def extend_feature( span_no=6, input=None, trunc_long_time=False, mini=False):
    prefix='tol'
    df = extend_time_span(version=version, trunc_long_time=trunc_long_time, mini=mini)
    # df = reduce_time_span(df, prefix, span_no)
    df = extend_percent(df, prefix)

    df = extend_brand_pkg(df)
    if input is not None:
        if 'device' not in list(input.columns):
            input.index.name = 'device'
            input = input.reset_index()
        print(f'({list(input.columns)}')

        print(f'({list(df.columns)}')

        df = input.merge(df, on='device', how='left')

    drop_list = ['tol_day_cnt_min', 'tol_day_cnt_max', 'tol_day_min', 'tol_day_max']
    drop_list = [ col for col in df.columns if col in drop_list]
    df.drop(columns=drop_list, inplace=True)

    return df

def extend_percent(df, prefix):
    for col in [item for item in df.columns if f'_sum' in item]:
        df[f'{col}_p'] = round(df[col] / df[f'{prefix}_total_sum_'], 5)

    for col in [item for item in df.columns if f'_count' in item]:
        df[f'{col}_p'] = round(df[col] / df[f'{prefix}_total_count_'], 5)

    drop_col = [col for col in df.columns if str(col).endswith('_count') or str(col).endswith('_sum')]
    print(f'=========After percent, will drop:{drop_col}')
    df.drop(columns=drop_col, inplace=True)

    return df

@timed()
@file_cache()
def extend_time_span(version, trunc_long_time=False, mini=False):
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    list = sorted(list, reverse=True)

    if mini:
        list =  list[:3]

    duration_list = []
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path) and 'csv' in path:
            print(f"Try to summary file:{path}")
            df = get_start_closed(path)

            if mini:
                print('Return mini result for testing')
                df = df[0:1000]

            df = split_days_all(df, trunc_long_time)
            df = cal_duration_for_span(df, span_no=24)
            df_weekday = get_summary_weekday(df)
            df_span    = get_summary_span24(df)
            df = pd.concat([df_weekday, df_span], axis=1)
            if len(df) > 0 :
                duration_list.append(df)
            else:
                print(f'The df is None for file:{path}')
    all = pd.concat(duration_list)
    return all.reset_index()




@timed()
#@file_cache()
def split_days_all(tmp, trunc_long_time=None):
    if trunc_long_time == True:
        trunc_week_sn =2
        # 超长记录,截取后面的数据, 最多保留2个星期,最少保留一个完整的星期
        tmp['start_tmp'] = (tmp.close - Week(trunc_week_sn, weekday=0)).dt.date.astype('datetime64[ns]')
        tmp.start = tmp[['start', 'start_tmp']].max(axis=1)
        tmp.drop( columns=['start_tmp'] , inplace=True )

    tmp.duration = (tmp.close - tmp.start) / np.timedelta64(1, 'D')

    old_len = len(tmp)
    print(f'Out loop: The original Df size is {old_len}')
    tmp = split_days(tmp, 50)
    tmp = split_days(tmp, 1)
    print(f'Out loop: The new Df size is {len(tmp)}, old df size is {old_len}')

    tmp['start_base'] = tmp['start'].dt.date
    tmp['weekday'] = tmp.start.dt.weekday
    tmp['weekbegin'] = (tmp['start'] -
                        tmp['start'].dt.weekday.astype('timedelta64[D]')).dt.date


    tmp.duration = round(tmp.duration, 6)

    tmp = tmp.sort_values(by = ['device','package','start'])

    return tmp

#@timed()
def split_days(tmp, threshold_days = 100):
    threshold_days = max(1,threshold_days)
    # print(f'The input df#{len(tmp)} before split, with max duration:{tmp.duration.max()} '
    #                              f'and  threshold_days@{threshold_days}')

    # 检查是否有需要截断的数据, 如果没有直接Return, 或者进入小循环
    tmp_todo_big = tmp[tmp.duration > threshold_days]
    if len(tmp_todo_big) == 0 and tmp.duration.max() <= threshold_days:
        print(f'Final return with para:{threshold_days}:{len(tmp)}')
        return tmp


    # 创建新记录,截取最近的时间段(大段)
    tmp_todo_big.start = tmp_todo_big.start.dt.date + pd.DateOffset(threshold_days)
    tmp_todo_big.duration = (tmp_todo_big.close - tmp_todo_big.start) / np.timedelta64(1, 'D')
    tmp_big = split_days(tmp_todo_big, threshold_days)

    # inpu中,已经小于阀值天的
    tmp_small_p1         = tmp[tmp.duration <= threshold_days]
    # 旧记录,保留早期的时间段(小段)
    tmp_small_p2   = tmp[tmp.duration > threshold_days]
    tmp_small_p2.close    = tmp_small_p2.start.dt.date + pd.DateOffset(threshold_days)
    #tmp_small_p2.duration = (tmp_small_p2.close - tmp_small_p2.start) / np.timedelta64(1, 'D')
    tmp_small = pd.concat([tmp_small_p1, tmp_small_p2])
    tmp_small.duration = (tmp_small.close - tmp_small.start) / np.timedelta64(1, 'D')

    # print(f'max duration:{tmp_small_p2.duration.max()} with small threshold:{threshold_days}')

    tmp = tmp_big.append(tmp_small)

    #tmp = tmp.sort_values('duration', ascending=False)
    tmp.reset_index(drop=True, inplace=True)

    # print(f'The output df#{len(tmp)} after split')

    return tmp

@timed()
#@file_cache()
def get_start_closed(file=None):

    file = file if file is not None else './output/start_close/deviceid_package_start_close_40_38_35780089_36720940.csv'

    start_close = pd.read_csv(file,
                              # index_col=0 ,
                              nrows=None,
                              header=None )

    if len(start_close) == 0 :
        return pd.DataFrame()

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
    start_close['duration'] = (start_close.close - start_close.start) / np.timedelta64(1, 'D')

    return start_close

