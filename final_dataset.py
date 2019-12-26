import pandas as pd
from pandas.io.json import json_normalize
import json


dataset_file = open('Dataset_SimpleQA_type_labels.json','r')
dataset_decode = json.load(dataset_file)
final_tab_type = json_normalize(dataset_decode).sort_values(by=['question_id'])

dataset_file = open('Dataset_SimpleQA_labels.json','r')
dataset_decode = json.load(dataset_file)
final_tab_label = json_normalize(dataset_decode).sort_values(by=['question_id'])

final_tab_label['type_list_label'] = final_tab_type['type_list_label']
print(final_tab_label['type_list_label'].head(5))
final_tab_label.to_json(r'Dataset_SimpleQA_labels_all.json',orient='records')
print('Done')
