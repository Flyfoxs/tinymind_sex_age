
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from utils_.util_log import *


@timed(logger)
def convert_label_encode(sample,  excluded_list=[]):

    label_encode = defaultdict(LabelEncoder)
    sample = sample.apply(lambda x: label_encode[x.name].fit_transform(x.astype(str))
                    if x.name not in excluded_list else x,
                    reduce=False)


    return sample