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


@file_cache()
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

    device_app_test.index.rename('device', inplace=True)

    return device_app_test.to_frame().reset_index()

def drop_useless_package(df):
    useless = get_drop_list_for_install()
    df.columns = [ col.split('_')[-1]
                   if '_' in col  else col
                   for col in df.columns
                   ]
    columns = [ col for col in df.columns if col in useless.device ]
    print(f'There are {len(columns)} will be droped:{columns}')
    df.drop(columns=columns, inplace=True)
    return df

@timed()
def get_drop_list_for_usage():
    df = extend_package(version=1, mini=False)
    count = df[[col for col in df.columns if 'count_' in col]]

    deviceid_test = pd.read_csv('./input/deviceid_test.tsv', sep='\t', names=['device'])

    device_usage_test = count[count.index.isin(deviceid_test.device)]

    tmp = device_usage_test.sum()
    return tmp[tmp==0]




@file_cache(type='pkl', overwrite=False)
@timed()
def extend_package(version=1, mini=mini):
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
                df = df[:1000]
            df = split_days_all(df)
            df = extend_package_merge(df)
            if len(df) > 0 :
                tmp_list.append(df)
            else:
                print(f'The df is None for file:{path}')

    df = pd.concat(tmp_list)
    print(f'Share of device package usage is:{df.shape}')

    return df.sort_index()


@timed()
#@file_cache(type='pkl', overwrite=True)
def extend_package_count_df(df):
    p = df.groupby(['device', 'package'])['start_base'].nunique().reset_index()
    #p = df.groupby(['device', 'package'])['duration'].sum().reset_index()
    p = p.pivot(index='device', columns='package', values='start_base').reset_index()
    print(f'Device_Package: convert {df.shape} to {p.shape} ')
    p.set_index('device', inplace=True)
    p.columns=[f'count_{item}' for item in p.columns]
    return p

@timed()
#@file_cache(type='pkl', overwrite=True)
def extend_package_duration_df(df):
    #p = df.groupby(['device', 'package'])['start_base'].nunique().reset_index()
    p = df.groupby(['device', 'package'])['duration'].sum().reset_index()
    p = p.pivot(index='device', columns='package', values='duration').reset_index()
    print(f'Device_Package: convert {df.shape} to {p.shape} ')
    p.set_index('device', inplace=True)
    p.columns = [f'duration_{item}' for item in p.columns]
    return p


def extend_package_merge(df):
    return pd.concat([
                     extend_package_count_df(df) ,
                      extend_package_duration_df(df),
                      ], axis=1)


