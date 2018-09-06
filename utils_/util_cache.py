import redis
from utils_.util_log import *
from redis.sentinel import Sentinel
class Cache:
    enable=True
    #enable=False
    #10.224.38.43---slave, 10.224.38.31--master
    #conn=redis.Redis(host='10.224.38.31', port=8690,db=13, password='cisco2016redis')
    master=None
    slave=None

    def __init__(self):
        self.master = redis.Redis(host='10.224.38.31', port=8690, db=13, password='cisco2016redis')
        self.slave = redis.Redis(host='10.224.38.43', port=8690, db=13, password='cisco2016redis')
        try:
            self.master.set('a','a_value')
        except:
            tmp=self.slave
            self.slave=self.master
            self.master=tmp


    def readRedis(self,key):
        if self.enable:
            val = self.slave.get(key)
            key_obj = pickle.loads(key)
            if val is None:
                logger.debug("Redis:can not find data for KEY:%s \n" % (str(key_obj)) )
                return None
            else:
                logger.debug( "Redis:get VALUE from Redis by KEY:%s \n" % ( str(key_obj)))
                return pickle.loads(val)
        else:
            logger.debug( "disable cache")


    def writeRedis(self, key, val, time ):
        if val is None:
            logger.debug( "Val is None, don't save it to redis \n")
        else:
            self.master.set(key, pickle.dumps(val) )
            self.master.expire(key, time) #1week
            logger.debug( "====Write value of KEY:%s to redis \n" % str(pickle.loads((key))))

redisCache =  Cache()

import pickle, functools
def rd_cache(time=60*60*24*7, overwrite=False):
    """
    :param time: How long the case can keep, default is 1 week
    :param overwrite: If force overwrite the cache
    :return:
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            logger.debug(f'fn:{f.__name__}, para:{str(args)}, para:{str(kwargs)}')
            key = pickle.dumps((f.__name__, args, kwargs))
            if overwrite==False:
                val = redisCache.readRedis(key)
            if val is None or overwrite:
                val = f(*args, **kwargs) # call the wrapped function, save in cache
                redisCache.writeRedis(key, val, time)
            return val # read value from cache
        return wrapper
    return decorator

if __name__ == '__main__':

    @timed()
    @rd_cache(10)
    def test_cache(name):
        import time
        import numpy  as np
        time.sleep(3)
        return np.random.random()

    test_cache('Felix')
    test_cache('Felix')




