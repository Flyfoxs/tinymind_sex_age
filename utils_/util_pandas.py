
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from utils_.util_log import *
import numpy as np
import pandas as pd


@timed(logger)
def convert_label_encode(sample,  list=[]):

    label_encode = defaultdict(LabelEncoder)
    sample = sample.apply(lambda x: label_encode[x.name].fit_transform(x.astype(str))
                    if x.name in list else x,
                    reduce=False)


    return sample


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
        return df.iloc[:4, :3]
    else:
        return pd.DataFrame()