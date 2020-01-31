import pandas as pd
from pandas.io.json import json_normalize
import json
import csv

dataset_file = open('Dataset_SimpleQA_qualifiers_2500.json','r')
dataset_decode = json.load(dataset_file)
tab = pd.DataFrame(data=dataset_decode, columns= ['question_id', 'question','answer_entity_labels'])

#print (tab.head(3))
#print (tab.describe)

tab_ent= tab['answer_entity_labels'].apply(pd.Series)
tab_ent = tab_ent.rename(columns  = lambda x : 'tag_' + str(x))
new_tab = pd.concat([tab[:],tab_ent[:]],axis=1)
new_tab = new_tab[new_tab['tag_1'].isna()]
new_tab = new_tab[new_tab['tag_2'].isna()]
new_tab = new_tab.drop(['answer_entity_labels'],axis=1)
new_tab.rename(columns = {'tag_0':'answer_entity_labels'}, inplace=True)
new_tab = new_tab.dropna(axis = 'columns', how= 'all')

#print(new_tab.head(3))
#print(new_tab.describe)

#index = list(range(0,961))

tab1 = new_tab.iloc[:961]
tab1.rename(columns = {'question_id':'UID1','question':'Ques1','answer_entity_labels':'Ans1'},inplace=True)
tab1 = tab1.reset_index(drop=True)
print (list(tab1.columns))
print(tab1.head(3))

tab2 = new_tab.iloc[961:]
tab2.rename(columns = {'question_id':'UID2','question':'Ques2','answer_entity_labels':'Ans2'},inplace=True)
tab2 = tab2.reset_index(drop=True)
print (list(tab2.columns))
print(tab2.head(3))


final_tab = tab1.join(tab2)
print (list(final_tab.columns))

final_tab.to_csv(r'Dataset_AMT.csv',encoding = 'utf-8',quoting = csv.QUOTE_ALL, index_label = 'Index')

print(final_tab.head(3))
print(final_tab.describe)
