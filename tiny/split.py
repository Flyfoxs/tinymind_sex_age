from  tiny.util import *
#split_start_close()

rootdir = './output/start_close/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
       path = os.path.join(rootdir,list[i])
       if os.path.isfile(path) and 'csv' in path:
           print(path)