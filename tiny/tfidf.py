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
#@file_cache(type='pkl')
def get_cntTf(type, group_type='package'):
    if type == 'app':
        cntTf_app = extend_package_install(type=group_type)
        cntTf = cntTf_app
    elif type == 'count':
        cntTf_all = extend_package(version=version, mini=mini, type=group_type)
        cntTf_count = cntTf_all[[col for col in cntTf_all.columns if 'count_' in col]]
        cntTf = cntTf_count
    elif type == 'duration':
        cntTf_all = extend_package(version=version, mini=mini, type=group_type)
        cntTf_duration = cntTf_all[[col for col in cntTf_all.columns if 'duration_' in col]]
        cntTf = cntTf_duration
    return cntTf

#
# @timed()
# @file_cache()
# def get_cntTf_group(type):
#     cntTf = get_cntTf(type)
#     cntTf_label = extend_pkg_label(cntTf)
#     cntTf_label.replace({0:np.nan})
#     return cntTf_label.groupby(['p_type','p_sub_type']).agg(['sum','count'])