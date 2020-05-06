import pandas
from pandas.io.json import json_normalize
import json
input_file = open('test.json','r')
json_decode = json.load(input_file)
tab = json_normalize(json_decode,'relation')
tab.columns = ['rel']
frequency= tab.groupby('rel')['rel'].size().reset_index(name='freq').sort_values(by=['freq'],ascending=False)
frequency.to_csv('relation_distribution.csv',sep = '|',index=False)
print("yes")
