from  tiny.package import *
from  tiny.util import *
from  tiny.lda import *
from  tiny.tfidf import *
print('begin')
# cntTf = get_cntTf('app')

# tfidf =  cal_tfidf(cntTf)

#p_type	p_sub_type	combine_type
# extend_package(version=1, mini=False,type='combine_type')
# extend_package(version=1, mini=False,type='p_type')
# extend_package(version=1, mini=False,type='p_sub_type')
# extend_package(version=1, mini=False,type='package')


rootdir = './output/start_close/'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
# list = sorted(list, reverse=True)



final_path = []
for i in range(0, len(list)):
    path = os.path.join(rootdir, list[i])
    if os.path.isfile(path) and 'csv' in path:
        final_path.append(path)



from joblib import Parallel, delayed

Parallel(n_jobs=4, backend="threading")(
    delayed(cal_duration_for_partition)(item) for item in final_path
)
