
from utils_.util_log import *
import pandas as pd
import  os
class Cache_File:
    def __init__(self):
        self.cache_path='./cache/'
        self.enable=True
        self.date_list = ['start','close','start_base','weekbegin']
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

    def get_path(self, key):
        return f'{self.cache_path}{key}.csv'

    def readFile(self, key):
        if self.enable:
            path = self.get_path(key)
            if os.path.exists(path):
                logger.debug(f"try to read cache from file:{path}")

                #check if the file have the data type column
                df = pd.read_csv(path, nrows=1)
                tmp_data_list = [item for item in self.date_list if item in df.columns]

                df =pd.read_csv(path, parse_dates = tmp_data_list)
                logger.debug(f"Return {len(df) } resut from file cache:{path}")
                return df
            else:
                logger.debug(f"Can not find cache from file:{path}")
                return None
        else:
            logger.debug( "disable cache")


    def writeFile(self, key, val):
        if isinstance(val, pd.DataFrame ) and len(val)>0:
            path = self.get_path(key)
            logger.debug( f"====Write {len(val)} records to File#{path}" )
            val.to_csv(path, index=False, )
            return val
        else:
            logger.warning('The return is not DataFrame or it is None')
            return None

cache =  Cache_File()

import pickle, functools
def file_cache(overwrite=False):
    """
    :param time: How long the case can keep, default is 1 week
    :param overwrite: If force overwrite the cache
    :return:
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            mini_args = get_mini_args(args)
            logger.debug(f'fn:{f.__name__}, para:{str(mini_args)}, kw:{str(kwargs)}')
            key = '_'.join([f.__name__, str(mini_args), str(kwargs)])
            if overwrite==False:
                val = cache.readFile(key)
            if val is None or overwrite:
                val = f(*args, **kwargs) # call the wrapped function, save in cache
                cache.writeFile(key, val)
            return val # read value from cache
        return wrapper
    return decorator

def get_mini_args(args):
    args_mini = [item.split('/')[-1] if isinstance(item, str) else item
                    for item in args
                        if (type(item) in (tuple, list, dict) and len(item) <= 5)
                            or type(item) not in (tuple, list, dict, pd.DataFrame)
                 ]



    df_list  =  [item for item in args if isinstance( item, pd.DataFrame) ]

    i=0
    for df in df_list:
        args_mini.append(f'df{i}_{len(df)}')
        i += 1

    return args_mini

if __name__ == '__main__':

    @timed()
    @file_cache()
    def test_cache(name):
        import time
        import numpy  as np
        time.sleep(3)
        return pd.DataFrame(data= np.arange(0,10).reshape(2,5))

    print(test_cache('Felix'))
    #print(test_cache('Felix'))




