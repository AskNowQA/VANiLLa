import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np 

dataset_file = open('Dataset_test.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)

distribution = pd.read_csv("relation_distribution.csv",sep='|').head(50)
tab_rel = tab['relation'].apply(pd.Series)
tab_rel = tab_rel.rename(columns  = lambda x : 'tag_' + str(x))
final_tab = pd.concat([tab[:],tab_rel[:]],axis=1)
final_tab = final_tab.drop(['relation'],axis=1)
final_tab.rename(columns = {'tag_0':'relation'}, inplace=True)

for rel in distribution['rel']:
	tab_questions = final_tab[final_tab.relation == rel]['question']
	question = ""
	for q in tab_questions:
		question += q + " "
	wordlist = question.split()
	word_unique_list = np.unique(np.array(wordlist))
	term_count = pd.DataFrame(columns = ['word','frequency'])
	i = 0
	for w in word_unique_list:
		frames = pd.DataFrame({'word':w,'frequency': wordlist.count(w)},index=[i])
		term_count = pd.concat([term_count, frames],axis=0)
		i =+ 1
	filepath= "Term_Count_All/" + rel + ".csv"
	term_count.to_csv(filepath,sep = '|',index=False,encoding='utf-8')
	print(rel)
	print(term_count.head(3))
