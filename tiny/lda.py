from utils_.util_cache_file import *
from tiny.util import *

@file_cache()
@timed()
def get_lda_feature():
    import pandas as pd

    import numpy as np
    from tqdm import tqdm
    from sklearn.decomposition import LatentDirichletAllocation

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

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



@file_cache(overwrite=True)
@timed()
def get_lda_from_usage():
    import pandas as pd

    import numpy as np
    from tqdm import tqdm
    from sklearn.decomposition import LatentDirichletAllocation

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    path = './input/'
    data = pd.DataFrame()
    sex_age = pd.read_excel('./input/性别年龄对照表.xlsx')
    deviceid_packages = pd.read_csv(path + 'deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    deviceid_test = pd.read_csv(path + 'deviceid_test.tsv', sep='\t', names=['device'])
    deviceid_train = pd.read_csv(path + 'deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])

    deviceid_train = pd.concat([deviceid_train, deviceid_test])

    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))

    deviceid_packages.sort_values('device', inplace=True)

    #apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    #Replace point
    cntTf = extend_package(version=1)
    # cntTf = cntTf/cntTf
    cntTf.fillna(0, inplace=True)

    print(f'cntTf column:{cntTf.columns}')

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
    print('Lda analysis begin')
    docres = lda.fit_transform(cntTf)
    print('Lda analysis end')

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

    deviceid_train.columns = [ str(item)  for item in deviceid_train.columns ]
    return deviceid_train


if __name__ == '__main__':
    get_lda_from_usage()
    extend_package(version=1)


