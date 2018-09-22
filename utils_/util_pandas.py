
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from utils_.util_log import *
import numpy as np
import pandas as pd


@timed(logger)
def convert_label_encode(sample):
    try:
        #Label encode
        obj_col = sample.select_dtypes(include=['object']).columns
        obj_col = [ item for item in obj_col if item != 'device']
        print(f'{obj_col} will convert to label encode, and fillna with Other')


        sample = sample.apply(lambda x: x.fillna('Other')
                                if x.name in obj_col else x,
                                                      reduce=False)

        label_encode = defaultdict(LabelEncoder)
        sample = sample.apply(lambda x: label_encode[x.name].fit_transform(x.astype(str))
                        if x.name in obj_col else x,
                        reduce=False)


        return sample
    except Exception as e:
        print(f'The columns typs is {sample.dtypes.sort_values()}')
        raise e


import numpy as np
def check_exception(df, index=None):
    df = df.copy(deep=True)
    if index is not None and index in df:
        df.set_index(index,inplace=True)
    df = df.select_dtypes( #include=['float64', 'int'],
                           exclude=['object', 'datetime64[ns]'],)
    try:
        x, y = np.where(np.isinf(df.values) | np.isnan(df.values))
    except Exception as error:
        print(df.dtypes.sort_values())
        raise error
    if len(x)>0:
        print(x.min(), x.max()+1, y.min(), y.max()+1)
        df = df.iloc[x.min():(x.max()+1), y.min():(y.max()+1)]
        return df.iloc[:3, :4]
    else:
        return pd.DataFrame()

def merge_score(file_list):
    df_merge = None
    for  weight, name, file  in file_list:

       df = pd.read_csv(file, index_col ='DeviceID')

       df = df * weight

       if df_merge is None:
           df_merge = df
       else:
           df_merge = df_merge+df
    return df_merge

if __name__ == '__main__':
    file_list = [
        (0.5, 'lg' , './sub/baseline_lg_2.64604_.csv'),
        (0.5, 'dnn',  './sub/baseline_dnn_2.664605327415466_epochs 200, dense 20, dropout 0.51.csv'),
    ]
    score = merge_score(file_list)
    score = round(score, 10)
    weight=[str(f'{file[1]}|{file[0]}') for file in file_list]
    score.to_csv(f'./sub/merge_score_{"_".join(weight)}.csv')



