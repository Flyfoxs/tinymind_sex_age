folder="./output/best/"
import pandas as pd

def read_result_for_ensemble(file):
    #file = f'./output/best/{name}.h5'
    store = pd.HDFStore(file)
    return store["train"], store["label"], store["test"]



file_list = [
    ''

]