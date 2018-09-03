from tiny.util import *

tmp = pd.read_csv('input/deviceid_packages.tsv', sep='\t', header=None)
tmp.columns = ['device', 'package_list']

tmp = tmp[tmp.device.isin(get_test().iloc[:, 0])]

package_list_all = frozenset.union(*tmp.iloc[:, 1].apply(lambda v: frozenset(v.split(','))))

print(len(package_list_all))

i =1
for package in package_list_all:
    i += 1
    print(f'{i}/{len(package_list_all)}')
    tmp[package] = tmp.apply(lambda _: int(package in _.package_list), axis=1)

tmp.to_csv('./output/deviceid_package.tsv')