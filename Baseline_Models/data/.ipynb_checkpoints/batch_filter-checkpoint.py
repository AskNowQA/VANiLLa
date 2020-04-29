import pandas as pd
from pandas.io.json import json_normalize
import json

dataset_file = open('Batch_1_train.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)
tab = tab.dropna(axis = 0, how = 'any')
tab.to_json(r'Batch_1_train_filter.json',orient='records')

dataset_file = open('Batch_1_test.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)
tab = tab.dropna(axis = 0, how = 'any')
tab.to_json(r'Batch_1_test_filter.json',orient='records')