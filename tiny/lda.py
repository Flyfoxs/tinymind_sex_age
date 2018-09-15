from utils_.util_cache_file import *
from tiny.util import *
import pandas as pd

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tiny.package import *
#
# @file_cache(overwrite=False)
# @timed()
# def get_lda_from_app_install(drop=False):
#
#     path = './input/'
#
#     deviceid_packages = pd.read_csv(path + 'deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
#
#     deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
#     deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))
#
#     apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
#     vectorizer = CountVectorizer()
#     cntTf = vectorizer.fit_transform(apps)
#     if drop:
#         cntTf = pd.DataFrame(data=cntTf.toarray(), index=deviceid_packages.device,
#                                  columns=vectorizer.get_feature_names())
#         cntTf = drop_useless_package(cntTf)
#         import scipy
#         cntTf = scipy.sparse.csr_matrix(cntTf.values)
#     lda = LatentDirichletAllocation(n_topics=5,
#                                     learning_offset=50.,
#                                     random_state=666)
#     docres = lda.fit_transform(cntTf)
#     deviceid_packages = pd.concat([deviceid_packages, pd.DataFrame(docres)], axis=1)
#     deviceid_packages = deviceid_packages.drop('apps', axis=1)
#
#
#     # transformer = TfidfTransformer()
#     # tfidf = transformer.fit_transform(cntTf)
#     # #word = vectorizer.get_feature_names()
#     # weight = tfidf.toarray()
#     # df_weight = pd.DataFrame(weight)
#     # feature = df_weight.columns
#     # df_weight['sum'] = 0
#     # for f in tqdm(feature):
#     #     df_weight['sum'] += df_weight[f]
#     # deviceid_packages['tfidf_sum'] = df_weight['sum']
#
#
#     return deviceid_packages
#

def attach_device_label(df):

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


#@file_cache(overwrite=True)
@timed()
def get_lda_from_usage(mini):

    drop =True

    df_list = [get_device_pkg('app', drop=False),
               get_device_pkg('count', drop=False),
               get_device_pkg('duration', drop=False),
               get_device_pkg('app', drop=drop),
               get_device_pkg('count', drop=drop),
               get_device_pkg('duration', drop=drop),
               ]

    for df in df_list:
        if 'device' in df.columns:
            df.set_index('device', inplace=True)


    all = pd.concat(df_list, axis=1)

    all.columns = [ str(col) for col in all.columns]

    all = all[['0', '1', '2', '3', '4',]]
    print(f'Device_pkg all column:{all.columns}')

    all = all.reset_index()
    return all


@timed()
@file_cache(overwrite=True)
def get_device_pkg(type='app', drop=False):
    from tiny.tfidf import get_cntTf
    cntTf = get_cntTf(type)

    if drop:
        cntTf = drop_useless_package(cntTf)

    print(f'Try to lda for type#{type}')

    tmp = cntTf / cntTf

    docres = get_lda_docres(cntTf)
    docres[f'{type}_{drop}_app_length'] = tmp.sum(axis=1)

    # docres = pd.concat([deviceid_packages, pd.DataFrame(docres)], axis=1)
    #

    # df_weight = get_tfidf(cntTf)
    # deviceid_packages[f'tfidf_sum_{type}'] = df_weight['sum']

    print(f'Already calculate lda for {type} DF')

    #docres.drop(columns=['apps'], inplace=True)
    #print(f'deviceid_packages column:{deviceid_packages.columns}')
    return docres



def get_lda_docres(cntTf):
    # Replace point
    print(f'cntTf type:{type(cntTf)}')
    # if not isinstance(cntTf, pd.DataFrame):
    #     cntTf = pd.DataFrame(cntTf.toarray())
    cntTf.fillna(0, inplace=True)
    lda = LatentDirichletAllocation(n_topics=5,
                                    learning_offset=50.,
                                    random_state=666)
    #print(f'cntTf column:{cntTf.columns}')

    import scipy
    print('Convert df to csr_matrix')
    cntTf_sparse =  scipy.sparse.csr_matrix(cntTf.values)
    print('Lda analysis begin')
    docres = lda.fit_transform(cntTf_sparse)
    print('Lda analysis end')
    docres = pd.DataFrame(docres, index=cntTf.index)
    return docres


if __name__ == '__main__':
    get_lda_from_usage()
    extend_package(version=1)


