import pandas as pd
from pandas.io.json import json_normalize
import json


dataset_file = open('test.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)

distribution = pd.read_csv("relation_distribution.csv",sep='|').head(50)

tab_rel = tab['relation'].apply(pd.Series)
tab_rel = tab_rel.rename(columns  = lambda x : 'tag_' + str(x))
final_tab = pd.concat([tab[:],tab_rel[:]],axis=1)
final_tab = final_tab.drop(['relation'],axis=1)
final_tab.rename(columns = {'tag_0':'relation'}, inplace=True)

final_questions = pd.DataFrame(columns= final_tab.columns)

for item in distribution['rel']:
	tab_questions = final_tab[final_tab.relation == item].head(200)
	final_questions = pd.concat([final_questions[:],tab_questions[:]],axis=0)
final_questions.to_json(r'Dataset_SimpleQA.json',orient='records')
print(final_questions.head(3))

