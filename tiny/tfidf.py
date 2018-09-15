from tiny.util import *
from tiny.lda import *

def get_tfidf_app_and_usage(type, drop):
    cntTf = get_cntTf(type)

    if drop:
        cntTf = drop_useless_package(cntTf)
    tfidf =  get_tfidf(cntTf)


    tmp = tfidf / tfidf
    tfidf[f'{type}_tfidf_{drop}_length'] = tmp.sum(axis=1)

    return tfidf


def get_tfidf(cntTf):
    transformer = TfidfTransformer()
    import scipy
    cntTf = scipy.sparse.csr_matrix(cntTf.values)
    tfidf = transformer.fit_transform(cntTf)
    # word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    df_weight = pd.DataFrame(weight, index=cntTf.index)
    return df_weight


@timed()
def get_cntTf(type):
    if type == 'app':
        deviceid_packages = pd.read_csv('./input/deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
        deviceid_packages.sort_values('device', inplace=True)
        print(f'Try to load packge for type:{type}')
        deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))

        # deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))

        apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
        vectorizer = CountVectorizer()
        cntTf_app = vectorizer.fit_transform(apps)
        cntTf_app = pd.DataFrame(cntTf_app.toarray(),
                                 columns=vectorizer.get_feature_names(),
                                 index=deviceid_packages.device)
        cntTf = cntTf_app
    elif type == 'count':
        cntTf_all = extend_package(version=version, mini=mini)
        cntTf_count = cntTf_all[[col for col in cntTf_all.columns if 'count_' in col]]
        cntTf = cntTf_count
    elif type == 'duration':
        cntTf_all = extend_package(version=version, mini=mini)
        cntTf_duration = cntTf_all[[col for col in cntTf_all.columns if 'duration_' in col]]
        cntTf = cntTf_duration
    return cntTf

