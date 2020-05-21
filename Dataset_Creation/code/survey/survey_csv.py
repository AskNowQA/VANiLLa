# -*- coding: utf-8 -*-
import pandas as pd
import csv

dataset_file1 = open('survey_sample.csv','r')
dataset = pd.read_csv(dataset_file1)

#dataset = dataset.loc[:,['question_id','answer','question_entity_label','question_relation_label']]

dataset['triple'] = ""

for index, row in dataset.iterrows():
    triple = "<" + row['question_entity_label'] + "> <" + row['question_relation_label'] + "> <" + row['answer'] + ">"
    dataset.set_value(index,'triple',triple)
    
dataset = dataset.drop(['index','answer_sentence','answer','question_entity_label','question_relation_label'], axis = 1)
dataset.rename(columns = {'question_id':'UID'}, inplace =True)
print(dataset.shape)

index = len(dataset.index)

tab1 = dataset.iloc[:int(index/2)]
tab1.rename(columns = {'UID':'UID1','triple':'Triple1'},inplace=True)
tab1 = tab1.reset_index(drop=True)
#print (list(tab1.columns))
print(tab1.head(3))

tab2 = dataset.iloc[int(index/2):]
tab2.rename(columns = {'UID':'UID2','triple':'Triple2'},inplace=True)
tab2 = tab2.reset_index(drop=True)
#print (list(tab2.columns))
print(tab2.head(3))


final_tab = tab1.join(tab2)
print (list(final_tab.columns))


final_tab.to_csv('survey_AMT.csv', index_label = 'index', encoding = 'utf-8',quoting = csv.QUOTE_ALL)