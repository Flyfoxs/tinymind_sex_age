import pandas as pd

def read_result_for_ensemble(file):
    #file = f'./output/best/{name}.h5'
    store = pd.HDFStore(file)
    ensemble = (store["train"],
                store["label"] if 'label' in store else None,
                store["test"])
    store.close()
    return ensemble


if __name__ == '__main__':
    file_name = './output/best/baseline_2.613952_2804_xgb_1630_.h5'

    _, _, test = read_result_for_ensemble(file_name)

    new_file_name = file_name.split('/')[-1]

    test = test[['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2',
         '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']]
    test.index.name='DeviceID'
    test.to_csv(f'./sub/extract_{new_file_name}.csv')