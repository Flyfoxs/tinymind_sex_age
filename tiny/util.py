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
    mini=False
    version=4


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

    brand = brand.sort_values('brand')
    return brand

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
#
# @timed()
# def get_max_week(df):
#     """
#     到到每个Device使用的最多的那一周
#     :param df:
#     :return:
#     """
#     df = df.groupby(['device', 'weekbegin']).agg({'weekday':'nunique','package':'nunique'}).reset_index()
#
#     # .to_frame().reset_index()
#     # df.sort_values(['device', 'package'])
#
#     df = df.sort_values(by=['device', 'weekday','package', 'weekbegin'], ascending=False).groupby('device').nth(0)
#     df = df.reset_index()
#     df.rename({'package':'package_count'}, axis=1, inplace=True)
#
#     return df


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

def extend_pkg_label(df=None):

    pkg_label = get_package_label()
    #pkg_label.set_index('package', inplace=True)

    pkg_label['combine_type'] = pkg_label.apply(lambda row: f'{row.p_type}_{row.p_sub_type}', axis=1)
    if df is None:
        return pkg_label
    else:
        df = pd.merge(df, pkg_label, on='package', how='left')
        df.fillna('Unknown', inplace=True)
        return df

@timed()
def extend_device_brand(tmp):

    brand = get_brand()
    print(f'column list:{tmp.columns}')
    if tmp is None:
        return brand
    else:
        if 'device' not in tmp:
            tmp.index.name = 'device'
            tmp.reset_index(inplace=True)
        tmp = tmp.merge(brand, how='left')
        tmp[['brand', 'phone_type']] = tmp[['brand', 'phone_type']].fillna('Other', inplace=True)

        tmp = convert_label_encode(tmp,['brand', 'phone_type'])
        return tmp




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




@timed()
#@file_cache()
def split_days_all(tmp):

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

def replace_invalid_filename_char(filename):
    invalid_characaters = '\':"<>|'
    for char in invalid_characaters:
        filename = filename.replace(char, '')
    return filename


def attach_device_train_label(df):

    deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])
    deviceid_train = pd.read_csv('./input/deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])

    deviceid_train = pd.concat([deviceid_train, deviceid_test])

    deviceid_train['sex'] = deviceid_train['sex'].apply(lambda x: str(x))
    deviceid_train['age'] = deviceid_train['age'].apply(lambda x: str(x))

    def tool(x):
        if x == 'nan':
            return x
        else:
            return str(int(float(x)))

    deviceid_train['sex'] = deviceid_train['sex'].apply(tool)
    deviceid_train['age'] = deviceid_train['age'].apply(tool)
    deviceid_train['sex_age'] = deviceid_train['sex'] + '-' + deviceid_train['age']
    deviceid_train = deviceid_train.replace({'nan': np.NaN, 'nan-nan': np.NaN})
    if df is not None:
        df = pd.merge(df, deviceid_train, on='device', how='left')
        df.sort_values('device', inplace=True)

        return df
    else :
        return deviceid_train


@timed()
@file_cache(overwrite=True)
def get_stable_feature():
    from tiny.lda import get_lda_from_usage
    from tiny.usage import extend_feature

    lda_feature = get_lda_from_usage(n_topics=5)

    feature = extend_feature(span_no=24, input=lda_feature,
                             drop_useless_pkg=True, drop_long=0.3)

    feature=  extend_device_brand(feature)

    check = check_exception(feature, 'device')
    if not check.empty:
        print(f"Error Return feature NAN/INF:\n{check}" )
        exit(1)

    feature_label = attach_device_train_label(feature)
    return feature_label

# if __name__ == '__main__':
#     for drop_useless_pkg in [True, False]:
#         for drop_long in [1, 0.9, 0.7, 0.5, 0.3, 0.1]:
#             summary_time_trend_on_usage(version=version,
#                                         drop_useless_pkg=drop_useless_pkg,
#                                         drop_long=drop_long)
