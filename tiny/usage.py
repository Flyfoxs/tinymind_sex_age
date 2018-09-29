from tiny.group_label import summary_top_on_usage
from tiny.util import *
from functools import partial
from tiny.group_label import *

@file_cache()
@timed()
def cal_duration_for_partition(path='./output/start_close/deviceid_package_start_close_40_09_8474059_9415641.csv'):
    df = get_start_closed(path)
    df = split_days_all(df)
    df = cal_duration_for_span(df, span_no=24)
    return df

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

        print(f'Try to cal for range#{sn} with span_len: {span_len} hours')
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

def convert_count_to_percent(df):
    #需要挑选怎么样的百分比更合适
    # sum_pkg_nunique: 全部时间段,总共用了多少pkg
    # sum_day_nunique: 全部时间段,总共有多少天的统计数据
    # sum_day_min-sum_day_max 全部时间段,统计数据跨度多少天
    # sum_total_count_ 多个时间段的 count算术相加
    # dur_sum_daily 每天平均多久
    #

    try:
        import re
        pattern = re.compile(r'.*span.*_sum')
        columns = [item for item in df.columns if pattern.match(item)]
        print(f'Cal percentage for sum columns#{len(columns)}:{columns}')
        for col in columns:
            df[f'{col}_p'] = round(df[col] / df['sum_total_sum_'], 5)
        df.drop(columns=columns, inplace=True)

        pattern = re.compile(r'.*span.*_count')
        columns = [item for item in df.columns if pattern.match(item)]
        print(f'Cal percentage for count columns#{len(columns)}:{columns}')
        for col in columns:
            df[f'{col}_p'] = round(df[col] / df[f'sum_total_count_'], 5)
        df.drop(columns=columns, inplace=True)

        df.drop(columns=['sum_total_sum_', 'sum_total_count_'], inplace=True)

    except  Exception as error:
        print('Caught this error: ' + repr(error))
        print(f'Current column list:{df.columns}')
        raise error

    return df



#The function merge all the app together, but LDA will view it as different
def extend_feature( span_no=6, input=None, drop_useless_pkg=False, drop_long=False):

    #already merge all the app together
    df = summary_time_trend_on_usage(version=4,  drop_useless_pkg=drop_useless_pkg, drop_long=drop_long)
    # df = reduce_time_span(df, prefix, span_no)
    df.drop(columns=['day_dur'], inplace=True, errors='ignore')
    df = convert_count_to_percent(df)
    #
    #Extend top#n on usage
    df_label = summary_top_on_usage('p_sub_type',2)
    print(f'df_label@summary_top_on_usage:{df_label.shape}, {df_label.columns}')
    df = pd.merge(df, df_label, how='left', on='device')

    # df_label = summary_top_on_usage('kms_class',1)
    # df = pd.merge(df, df_label, how='left', on='device')

    # #Agg label on usage
    # df = summary_category(df, ['射击',
    #                             '游戏', '电商', '办公', '职', '体育', '医', '教育', '婴', '学习', '车', '母',
    #                             ])

    if input is not None:
        if 'device' not in list(input.columns):
            input.index.name = 'device'
            input = input.reset_index()
        #TODO , join is outer
        df = input.merge(df, on='device', how='left')

        from tiny.tfidf import attach_tfidf
        df = attach_tfidf(df)

        from tiny.util import extend_device_brand
        df = extend_device_brand(df)

    drop_list = ['tol_day_cnt_min', 'tol_day_cnt_max',
                 'p_type', 'p_sub_type',
                 'sum_day_min', 'sum_day_max']
    drop_list = [ col for col in df.columns if col in drop_list]
    df.drop(columns=drop_list, inplace=True)

    df.replace({np.nan:0, np.inf:0}, inplace=True)

    return df


@timed()
#@file_cache()
def get_summary_span24(df):
    #prefix = groupby[-1] if prefix is None else prefix

    columns = [key for key in df.columns if 'span24_' in key]
    gp_map = [(key, ['sum', 'count']) for key in columns if 'span24_' in key]
    gp_map = dict(gp_map)
    # print(gp_map)

    #统计总共用了多少Package
    gp_map['package'] = 'nunique'
    #统计Pkg总共用了多少天
    gp_map['start_base'] = ['min','max','nunique']

    df = df.groupby('device').agg(gp_map)

    df['total_sum']   = df[[key for key in df.columns if 'sum' in key]].sum(axis=1)
    df['total_count'] = df[[key for key in df.columns if 'count' in key]].sum(axis=1)

    df.rename({'package': 'pkg', 'start_base': 'day'}, axis=1, inplace=True)

    print(type(df.columns[0]))
    print('_'.join(df.columns[0]))
    #print(f'The latest colums:{df.columns}')
    df.columns = [f"sum_{'_'.join(key)}" for key in df.columns]

    print(f'The latest colums after get_summary_span24:{df.columns}')
    return df


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



@timed()
@file_cache(overwrite=False, type='h5')
def summary_time_trend_on_usage(version,drop_useless_pkg=False,drop_long=False):
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    path_list = sorted(list, reverse=True)
    path_list = [os.path.join(rootdir, item) for item in path_list if item.endswith('csv')]

    #from multiprocessing.dummy import Pool as ThreadPool

    from multiprocessing import Pool as ThreadPool

    pool = ThreadPool(processes=4)

    process_file = partial(summary_individual_file, drop_long=drop_long, drop_useless_pkg=drop_useless_pkg)
    results = pool.map(process_file, path_list)

    pool.close()
    pool.join()

    results = [item for item in results if len(item)>0]

    all = pd.concat(results)
    return all.reset_index()


def summary_individual_file(path, drop_long, drop_useless_pkg, ):
    print(f"Try to summary file:{path}")

    df = cal_duration_for_partition(path)

    # old_len = len(df)
    # df = df[df.start_base >= '2017-01-01']
    # print(f'{old_len - len(df))} rows({(old_len - len(df))/old_len}) <2017-01-01 is remove from {path}')

    if drop_long and drop_long < 1:
        print(f'Drop long session with session<={drop_long}, before:{len(df)})')
        df = df[df.day_duration <= drop_long]
        print(f'Drop long session with session<={drop_long}, after:{len(df)})')
    if drop_useless_pkg:
        from tiny.package import drop_useless_package
        print(f'The rows before drop:{len(df)} with:{drop_useless_pkg}')
        df = drop_useless_package(df, drop_useless_pkg)
        print(f'The rows after drop:{len(df)} with:{drop_useless_pkg}')
    df_weekday = get_summary_weekday(df)
    df_span = get_summary_span24(df)
    df = pd.concat([df_weekday, df_span], axis=1)
    return df

def get_summary_weekday(df):

    ##TODO
    #按照每个星期去统计
    gp = df.groupby(['device', 'weekday']).agg({'package': 'nunique', 'day_duration': 'sum', 'start':'count'})
    gp.reset_index(inplace=True)

    # #按照每天(weekday)去统计
    # gp0 = gp.pivot(index='device', columns='weekday', values='start')
    # gp0.columns = [f'action_{col}' for col in gp0.columns]

    gp1 = gp.pivot(index='device', columns='weekday', values='package')
    gp1.columns = [f'package_{col}' for col in gp1.columns]

    gp2 = gp.pivot(index='device', columns='weekday', values='day_duration')
    gp2.columns = [f'duration_{col}' for col in gp2.columns]


    #区分周末和工作日
    df['weekend'] = df.weekday // 5
    gp3 = df.groupby(['device', 'weekend']).agg({'package': 'nunique', 'weekday':'count', 'day_duration': 'sum'})
    gp3.reset_index(inplace=True)

    wk_end1 = gp3.pivot(index='device', columns='weekend', values='package')
    wk_end1.columns = [f'package_wk_{col}' for col in wk_end1.columns]

    wk_end2 = gp3.pivot(index='device', columns='weekend', values='day_duration')
    wk_end2.columns = [f'duration_wk_{col}' for col in wk_end2.columns]

    wk_end3 = gp3.pivot(index='device', columns='weekend', values='weekday')
    wk_end3.columns = [f'action_wk_{col}' for col in wk_end3.columns]

    wk = pd.concat([wk_end1, wk_end2, wk_end3], axis=1)


    #计算总数
    total = df.groupby(['device']).agg({'package': ['nunique', 'count'], 'duration': 'sum' ,'start_base':'nunique'})
    total.rename(columns={'package': 'pkg', 'duration': 'dur'}, inplace=True)
    total.columns = ['_'.join(item) for item in total.columns]

    merge = pd.concat([gp1, gp2, wk, total], axis=1)


    #工作日和周末的对比:
    merge['wk_compare_app_count'] = merge['package_wk_0'] / merge['package_wk_1']
    merge['wk_comapre_dur' ] = merge['duration_wk_0']/ merge['duration_wk_1']
    merge['wk_compare_action_count'] = merge['action_wk_0'] / merge['action_wk_1']

    merge['action_daily']    = merge['pkg_count']/merge['start_base_nunique']
    merge['dur_sum_daily']   = merge['dur_sum'] / merge['start_base_nunique']


    # 转换为package nunique 的Percentage
    columns = [col for col in merge.columns if f'package_' in col]
    print(f'will cal precent for weekly#package{len(columns)}:{columns}')
    for col in columns:
        merge[col] = merge[col] / merge['pkg_nunique']
    #merge.drop(columns=columns, inplace=True)

    columns = [col for col in merge.columns if f'duration_' in col]
    print(f'will cal precent for weekly#duration{len(columns)}:{columns}')
    for col in columns:
        merge[col] = merge[col] / merge['dur_sum']
    #merge.drop(columns=columns, inplace=True)



    return merge

def summary_pkg_activity(group_col, grou_method):
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    list = sorted(list, reverse=True)


    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path) and 'csv' in path:
            print(f"Try to summary file:{path}")
            pkg = cal_duration_for_partition(path)


def get_bottom_app(drop_level='count', limit=18363):
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
            df = cal_duration_for_partition(path)


if __name__ == '__main__':
    pass
    # for drop_useless_pkg in [True, False]:
    #     for drop_long in [1, 0.9, 0.7, 0.5, 0.3, 0.1]:
    #         summary_time_trend_on_usage(version=version,
    #                                     drop_useless_pkg=drop_useless_pkg,
    #                                     drop_long=drop_long)