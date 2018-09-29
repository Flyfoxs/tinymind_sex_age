#import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tiny.tfidf import *
from tiny.usage import *

def get_device_app_sequence_individual_file(file):
    tmp = cal_duration_for_partition(file)

    tmp = tmp.sort_values(['device', 'start'])
    #tmp.head()

    df = pd.DataFrame( {'apps':'', 'length':0}, index = tmp.device.unique())
    #df.head()
    for name, group in tmp.groupby('device'):
        #print(type(group.package))
        df.loc[name , 'apps'] = ' '.join(group.package)
        df.loc[name , 'length'] = len(group.package)
    return df


@timed()
@file_cache(overwrite=True)
def get_device_app_sequence():
    """
    apps
    app1, app2, app3
    """
    rootdir = './output/start_close/'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    path_list = sorted(list, reverse=True)
    path_list = [os.path.join(rootdir, item) for item in path_list if item.endswith('csv')]

    from multiprocessing import Pool as ThreadPool

    pool = ThreadPool(processes=8)

    results = pool.map(get_device_app_sequence_individual_file, path_list)
    pool.close()
    pool.join()

    results = [item for item in results if len(item)>0]

    all = pd.concat(results)
    all.index.name='device'
    all.reset_index(inplace=True)
    all = attach_device_train_label(all)

    return all.reset_index()[['sex_age', 'apps']]


def get_package_label(package_list=None):
    package = pd.read_csv('input/package_label.tsv', sep='\t', header=None, )
    package.columns = ['package', 'p_type', 'p_sub_type']
    if package_list is None:
        return package
    else:
        return package[package.package.isin(package_list)]


@timed()
@file_cache(overwrite=True)
def get_app_group():
    """
    apps, kms_*
    """
    from gensim.models import word2vec
    INPUT_FILE = "./cache/get_device_app_sequence_[]_{}.csv"

    sentences = word2vec.Text8Corpus(INPUT_FILE)  # 训练模型，部分参数如下
    model_20 = word2vec.Word2Vec(sentences, size=20, hs=1, min_count=0, window=5)

    model = model_20
    X = model[model.wv.vocab]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(X)

    y_kmeans = kmeans.predict(X)
    #print(y_kmeans.shape)

    df = pd.DataFrame({'package':list(model.wv.vocab.keys()), 'kms_class':y_kmeans, } )

    df['kms_class'] = df['kms_class'].apply(lambda val: f'kms_{val}')

    return df[[ 'package', 'kms_class',]]


if __name__ == '__main__':
    pass
    all  = get_device_app_sequence()
    # df = get_app_group()
    # print(df.shape)
    #all.to_csv('del.csv')

