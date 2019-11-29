import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np 

dataset_file = open('Dataset_SimpleQA.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)

distribution = pd.read_csv("relation_distribution.csv",sep='|').head(50)

for rel in distribution['rel']:
	tab_questions = tab[tab.relation == rel]['question']
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
	filepath= "Term_Count/" + rel + ".csv"
	term_count.to_csv(filepath,sep = '|',index=False,encoding='utf-8')
