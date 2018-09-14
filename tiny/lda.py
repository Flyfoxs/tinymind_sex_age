from utils_.util_cache_file import *
from tiny.util import *
import pandas as pd

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


@file_cache()
@timed()
def get_lda_feature():

    path = './input/'
    data = pd.DataFrame()
    sex_age = pd.read_excel('./input/性别年龄对照表.xlsx')
    deviceid_packages = pd.read_csv(path + 'deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    deviceid_test = pd.read_csv(path + 'deviceid_test.tsv', sep='\t', names=['device'])
    deviceid_train = pd.read_csv(path + 'deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])

    deviceid_train = pd.concat([deviceid_train, deviceid_test])

    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))

    apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    cntTf = vectorizer.fit_transform(apps)
    tfidf = transformer.fit_transform(cntTf)
    #word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    df_weight = pd.DataFrame(weight)
    feature = df_weight.columns
    df_weight['sum'] = 0
    for f in tqdm(feature):
        df_weight['sum'] += df_weight[f]
    deviceid_packages['tfidf_sum'] = df_weight['sum']

    lda = LatentDirichletAllocation(n_topics=5,
                                    learning_offset=50.,
                                    random_state=666)
    docres = lda.fit_transform(cntTf)

    deviceid_packages = pd.concat([deviceid_packages, pd.DataFrame(docres)], axis=1)

    temp = deviceid_packages.drop('apps', axis=1)
    deviceid_train = pd.merge(deviceid_train, temp, on='device', how='left')

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

    return deviceid_train



#@file_cache(overwrite=True)
@timed()
def get_lda_from_usage(mini):
    import pandas as pd

    import numpy as np
    from tqdm import tqdm
    from sklearn.decomposition import LatentDirichletAllocation

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    path = './input/'
    data = pd.DataFrame()
    sex_age = pd.read_excel('./input/性别年龄对照表.xlsx')
    deviceid_test = pd.read_csv(path + 'deviceid_test.tsv', sep='\t', names=['device'])
    deviceid_train = pd.read_csv(path + 'deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])

    deviceid_train = pd.concat([deviceid_train, deviceid_test])

    pkg = get_device_pkg_all()

    deviceid_train = pd.merge(deviceid_train, pkg, on='device', how='left')

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

    deviceid_train.columns = [ str(item)  for item in deviceid_train.columns ]
    return deviceid_train


def get_device_pkg_all():
    app = get_device_pkg('app')
    count = get_device_pkg('count')
    duration = get_device_pkg('duration')

    app.set_index('device', inplace=True)
    count.set_index('device', inplace=True)
    duration.set_index('device', inplace=True)

    all = pd.concat([app,
                     count,
                     duration,
                     ], axis=1)

    print(f'Device_pkg all column:{all.columns}')

    all = all.reset_index()
    return all


@timed()
@file_cache()
def get_device_pkg(type='app'):
    deviceid_packages = pd.read_csv('./input/deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    if type=='app':
        deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))

    deviceid_packages.sort_values('device', inplace=True)


    apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
    deviceid_packages.drop(columns=['apps'], inplace=True)
    vectorizer = CountVectorizer()
    cntTf_app = vectorizer.fit_transform(apps)

    cntTf_all = extend_package(version=version, mini=mini)
    cntTf_count = cntTf_all[[col for col in cntTf_all.columns if 'count_' in col]]
    cntTf_duration = cntTf_all[[col for col in cntTf_all.columns if 'duration_' in col]]

    cntf_map =  {'app':cntTf_app,
                  'count':cntTf_count,
                  'duration':cntTf_duration,
                 }

    cntTf = cntf_map[type]
    docres = get_lda_docres(cntTf)
    df_weight = get_tfidf(cntTf)
    deviceid_packages = pd.concat([deviceid_packages, pd.DataFrame(docres)], axis=1)
    deviceid_packages[f'tfidf_sum_{type}'] = df_weight['sum']

    print(f'Already calculate lda for {type} DF')


    print(f'deviceid_packages column:{deviceid_packages.columns}')
    return deviceid_packages


def get_tfidf(cntTf):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(cntTf)
    # word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    df_weight = pd.DataFrame(weight)
    feature = df_weight.columns
    df_weight['sum'] = 0
    for f in tqdm(feature):
        df_weight['sum'] += df_weight[f]
    return df_weight


def get_lda_docres(cntTf):
    # Replace point
    print(f'cntTf type:{type(cntTf)}')
    if not isinstance(cntTf, pd.DataFrame):
        cntTf = pd.DataFrame(cntTf.toarray())
    cntTf.fillna(0, inplace=True)
    lda = LatentDirichletAllocation(n_topics=5,
                                    learning_offset=50.,
                                    random_state=666)
    #print(f'cntTf column:{cntTf.columns}')
    print('Lda analysis begin')
    docres = lda.fit_transform(cntTf)
    print('Lda analysis end')
    return docres


if __name__ == '__main__':
    get_lda_from_usage()
    extend_package(version=1)


