

from tiny.usage import *
from tiny.util import *


@timed()
@file_cache(overwrite=False, type='h5')
def summary_top_on_usage(gp_col, top):
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    path_list = sorted(list, reverse=True)
    path_list = [os.path.join(rootdir, item) for item in path_list if item.endswith('csv')]

    from multiprocessing import Pool as ThreadPool

    pool = ThreadPool(processes=8)

    process_file_top = partial(summary_top_for_individual_file, gp_col=gp_col, top=top)
    results = pool.map(process_file_top, path_list)
    pool.close()
    pool.join()

    results = [item for item in results if len(item)>0]

    all = pd.concat(results)
    return all.reset_index()


@timed()
def summary_top_for_individual_file(path, gp_col, top):
    from tiny.usage import cal_duration_for_partition
    df = cal_duration_for_partition(path)
    df = extend_pkg_label(df)

    # group by specific label
    # p_type, p_sub_type, combine_type, package
    # gp_col = 'p_type'
    # top = 3

    ex_input = df
    # group by top#n
    gp = ex_input.groupby(['device', gp_col]).agg(
        {'start': ['count'], gp_col: ['nunique'], 'duration': ['sum'], 'start_base': ['nunique']})
    gp.columns = ["_".join(x) for x in gp.columns.ravel()]

    gp[f'{gp_col}_cnt_daily'] = gp[f'start_count'] / gp['start_base_nunique']
    gp[f'{gp_col}_dur_daily'] = gp[f'duration_sum'] / gp['start_base_nunique']

    gp = gp.reset_index()
    gp = gp.sort_values(['device', f'{gp_col}_cnt_daily'], ascending=False)
    gp['cum_sn'] = gp.groupby('device').cumcount()
    gp = gp[gp.cum_sn < top]

    gp
    gp00 = gp.pivot(index='device', columns='cum_sn', values=f'{gp_col}_cnt_daily')
    gp00.columns = [f'{gp_col}_cnt_top#{item}' for item in gp00.columns]

    # gp01 = gp.pivot(index='device', columns='cum_sn', values=f'{gp_col}_dur_daily')
    # gp01.columns = [f'{gp_col}_dur_top#{item}' for item in gp01.columns]

    gp02 = gp.pivot(index='device', columns='cum_sn', values=gp_col)
    gp02.columns = [f'{gp_col}_name_top#{item}' for item in gp02.columns]

    gp = ex_input.groupby(['device', gp_col]).agg({'start': ['count'], 'duration': ['sum'], 'start_base': ['nunique']})
    gp.columns = ["_".join(x) for x in gp.columns.ravel()]

    gp[f'{gp_col}_count'] = gp.count(axis=1)
    gp[f'{gp_col}_cnt_daily'] = gp[f'start_count'] / gp['start_base_nunique']
    gp[f'{gp_col}_dur_daily'] = gp[f'duration_sum'] / gp['start_base_nunique']
    gp
    gp = gp.reset_index()
    gp = gp.sort_values(['device', f'{gp_col}_cnt_daily'], ascending=False)
    gp
    gp['cum_sn'] = gp.groupby('device').cumcount()
    gp11 = gp.pivot(index='device', columns=gp_col, values=f'{gp_col}_cnt_daily')
    gp11.columns = [f'{gp_col}_cnt_{item}' for item in gp11.columns]

    gp12 = gp.pivot(index='device', columns=gp_col, values=f'{gp_col}_dur_daily')
    gp12.columns = [f'{gp_col}_dur_{item}' for item in gp12.columns]

    tmp = ex_input.groupby('device')[gp_col].nunique().to_frame()
    tmp.rename(columns={gp_col: f'{gp_col}_count'}, inplace=True)

    df = pd.concat([
                    gp00, #✔️
                   # gp01, #❎
                    gp02, #✔️
                    gp11, #✔️
                    gp12, #✔️
                    tmp,  #✔️
                    ], axis=1)

    df.fillna(0)

    # gp[[f'{gp_col}_count','cum_sn']] = gp.groupby('device').agg({'duration_sum':'count', 'package_nunique':'cumcount'})
    # gp
    # len(list(gp.columns))
    return df


def summary_usage(group_level):
    """
    group_col:
    group_method:
    :param group_level:
    :return:
    """


# def summary_category(df, category_list):
#     for category in category_list:
#         col = [item for item in df.columns if category in item and '_cnt_' in item]
#         print(f'There are {len(col)} cols for {category}')
#         df[f'{category}_cnt_agg'] = df[col].sum(axis=1)
#         df.drop(columns = col, inplace=True)
#
#         col = [item for item in df.columns if category in item and '_sum_' in item]
#         df[f'{category}_sum_agg'] = df[col].sum(axis=1)
#         df.drop(columns=col, inplace=True)
#     return df

if __name__ == '__main__':
    pass
    # summary_top_on_usage('p_type', 3)