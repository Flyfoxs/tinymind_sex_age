import pandas as pd
import os
import matplotlib.pyplot as plt
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils_.util_log import *
from utils_.util_date import *
from utils_.util_cache_file import *
from pandas.tseries.offsets import Week
from utils_.util_pandas import *
from tiny.util import *


@file_cache(overwrite=False)
@timed()
def get_drop_list_for_install(reverse=False):
    deviceid_packages = pd.read_csv('./input/deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])
    # deviceid_train = pd.read_csv('./input/deviceid_train.tsv', sep='\t', names=['device', 'sex', 'age'])
    #
    # deviceid_train = pd.concat([deviceid_train, deviceid_test])

    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))

    apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
    vectorizer = CountVectorizer()
    cntTf = vectorizer.fit_transform(apps)

    tokens = vectorizer.get_feature_names()
    device_app = pd.DataFrame(data=cntTf.toarray(), index=deviceid_packages.device,
                 columns=tokens)

    device_app_test = device_app[device_app.index.isin(deviceid_test.device)]

    device_app_test = device_app_test.sum()
    if reverse:
        device_app_test = device_app_test[device_app_test == 1]
    else:
        device_app_test = device_app_test[device_app_test == 0]

    device_app_test.index.rename('package', inplace=True)

    return device_app_test.to_frame().reset_index()

def drop_useless_package(df):
    useless = get_drop_list_for_install()
    print(f'Column will be remove:{useless.package[:10]}')
    print(f'Column will be compare:{df.columns[:10]}')
    df.columns = [ col.split('_')[-1]
                   if '_' in col  else col
                   for col in df.columns
                   ]
    columns = [ col for col in df.columns if col in useless.package.values ]
    print(f'There are {len(columns)} will be droped:{columns[:10]}')
    df.drop(columns=columns, inplace=True)
    return df
#
# @timed()
# def get_drop_list_for_usage():
#     df = extend_package(version=1, mini=False)
#     count = df[[col for col in df.columns if 'count_' in col]]
#
#     deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])
#
#     device_usage_test = count[count.index.isin(deviceid_test.device)]
#
#     tmp = device_usage_test.sum()
#     return tmp[tmp==0]
#



@timed()
#Can not save to pkl
@file_cache(type='pkl', overwrite=False)
def base_on_package_install_for_TF(type='package'):
    deviceid_packages = pd.read_csv('./input/deviceid_packages.tsv', sep='\t', names=['device', 'apps'])
    deviceid_packages.sort_values('device', inplace=True)
    print(f'Try to load packge for type:{type}')
    deviceid_packages['apps'] = deviceid_packages['apps'].apply(lambda x: x.split(','))
    # deviceid_packages['app_lenghth'] = deviceid_packages['apps'].apply(lambda x: len(x))
    apps = deviceid_packages['apps'].apply(lambda x: ' '.join(x)).tolist()
    vectorizer = CountVectorizer()
    cntTf_app = vectorizer.fit_transform(apps)
    cntTf_app = pd.SparseDataFrame(cntTf_app.toarray(),
                                     columns=vectorizer.get_feature_names(),
                                     index=deviceid_packages.device)


    return cntTf_app



@timed()
#@file_cache(type='pkl', overwrite=True)
def extend_package_count_df(df, col='package'):
    p = df.groupby(['device', col])['start_base'].nunique().reset_index()
    #p = df.groupby(['device', 'package'])['duration'].sum().reset_index()
    p = p.pivot(index='device', columns=col, values='start_base').reset_index()
    print(f'Device_Package: convert {df.shape} to {p.shape} ')
    p.set_index('device', inplace=True)
    p.columns=[f'count_{col}_{item}' for item in p.columns]
    return p

@timed()
#@file_cache(type='pkl', overwrite=True)
def extend_package_duration_df(df, col='package'):
    #p = df.groupby(['device', 'package'])['start_base'].nunique().reset_index()
    p = df.groupby(['device', col])['duration'].sum().reset_index()
    p = p.pivot(index='device', columns=col, values='duration').reset_index()
    print(f'Device_Package: convert {df.shape} to {p.shape} ')
    p.set_index('device', inplace=True)
    p.columns = [f'duration_{col}_{item}' for item in p.columns]
    return p


def extend_package_merge(df, col='package'):
    return pd.concat([
                     extend_package_count_df(df, col=col) ,
                      extend_package_duration_df(df, col=col),
                      ], axis=1)

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

