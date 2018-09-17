from tiny.util import *
from tiny.lda import *

#
# @timed()
# @file_cache(type='pkl',overwrite=False)
# def get_tfidf(type, drop, limit):
#     tfidf = get_tfidf_app_and_usage(type, drop)
#
#     col = list(tfidf.sum().sort_values(ascending=False)[:limit].index)
#
#     return tfidf[col]
#
# @timed()
# @file_cache(type='pkl',overwrite=False)
# def get_tfidf_app_and_usage(type, drop):
#     cntTf = get_cntTf(type)
#
#     if drop:
#         cntTf = drop_useless_package(cntTf)
#     tfidf =  cal_tfidf(cntTf)
#
#     tfidf = tfidf.to_dense().replace({0: np.nan}).to_sparse().fillna(0)
#     # tmp = tfidf / tfidf
#     # tfidf[f'{type}_tfidf_{drop}_length'] = tmp.sum(axis=1)
#
#     return tfidf
#
# @timed()
# def cal_tfidf(cntTf):
#     index = cntTf.index
#     transformer = TfidfTransformer()
#     import scipy
#     print('Try to sparse cntTF')
#     cntTf.fillna(0, inplace=True)
#     cntTf = scipy.sparse.csr_matrix(cntTf.values)
#     print(f'Try to cal tfidf for {type(cntTf)}')
#     tfidf = transformer.fit_transform(cntTf)
#
#     df_weight = pd.SparseDataFrame(tfidf.toarray(), index=index)
#     return df_weight
#

@timed()
@file_cache(type='pkl', overwrite=False)
def base_on_usage_for_TF(version, mini=mini, col='package'):
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    list = sorted(list, reverse=True)
    if mini:
        list =  list[:3]

    tmp_list = []
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path) and 'csv' in path:
            print(f"Try to summary file:{path}")
            df = get_start_closed(path)
            if mini:
                df = df[:5000]

            #需要packge的类型,就扩展app的类型:p_type, p_sub_type
            if type != 'package':
                print(f'Try to merge pkg label for col:{col}')
                df = extend_pkg_label(df)

            df = split_days_all(df)
            df = extend_package_TF(df, col=col)
            if len(df) > 0 :
                tmp_list.append(df)
            else:
                print(f'The df is None for file:{path}')

    df = pd.concat(tmp_list)
    df.fillna(0,inplace=True)
    print(f'Share of device package usage is:{df.shape}')

    return df.sort_index()


@timed()
@file_cache(type='pkl')
def get_cntTf( group_level, agg_col, agg_method):
    if group_level == 'app':
        cntTf_app = base_on_package_install_for_TF(agg_col)
        cntTf = cntTf_app
    elif group_level =='usage' and agg_method == 'count':
        cntTf_all = base_on_usage_for_TF(version=version, mini=mini, col=agg_col)
        cntTf_count = cntTf_all[[col for col in cntTf_all.columns if 'count_' in col]]
        cntTf = cntTf_count
    elif group_level =='usage' and agg_method == 'sum':
        cntTf_all = base_on_usage_for_TF(version=version, mini=mini, col=agg_col)
        cntTf_duration = cntTf_all[[col for col in cntTf_all.columns if 'duration_' in col]]
        cntTf = cntTf_duration
    else:
        cntTf = pd.DataFrame()
        print(f'Unknown params:{group_level}')

    cntTf.fillna(0, inplace=True)
    return cntTf.to_sparse(fill_value=0)

#
# @timed()
# @file_cache()
# def get_cntTf_group(type):
#     cntTf = get_cntTf(type)
#     cntTf_label = extend_pkg_label(cntTf)
#     cntTf_label.replace({0:np.nan})
#     return cntTf_label.groupby(['p_type','p_sub_type']).agg(['sum','count'])

if __name__ == '__main__':

    for group_level in ['usage']:
        for agg_col in ['p_sub_type', 'package']:
            for agg_method in ['sum', 'count']:
                get_cntTf(group_level, agg_col, agg_method)

    get_cntTf('app', 'package', None)
    get_cntTf('app', 'p_sub_type', None)