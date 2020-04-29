import pandas as pd
from pandas.io.json import json_normalize
import json
import csv
import re, math
from collections import Counter
from nltk.util import ngrams

"""
WORD = re.compile(r'\w+')

def text_to_vector(text):
	words = WORD.findall(text)
     	return Counter(words)

def get_cosine(vec1, vec2):
	intersection = set(vec1.keys()) & set(vec2.keys())
     	numerator = sum([vec1[x] * vec2[x] for x in intersection])

     	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     	denominator = math.sqrt(sum1) * math.sqrt(sum2)

     	if not denominator:
        	return 0.0
     	else:
        	return float(numerator) / denominator


def similarity (q1,q2):
	vec1 = text_to_vector(q1)
	vec2 = text_to_vector(q2)
	return get_cosine(vec1,vec2)




s1 = "Hello World"
s2 = "Hi all"
print(similarity(s1,s2))

"""

def similarity(s1,s2,a1,a2):
	s1 = s1.lower().replace(a1.lower(),"<ent>")
	s2 = s2.lower().replace(a2.lower(),"<ent>")
	s1_ngrams = list(ngrams(s1.split(),4))
	s2_ngrams = list(ngrams(s2.split(),4))
	#print(s1_ngrams)
	#print(s2_ngrams) 	
	for s1_ngram in s1_ngrams:
		for s2_ngram in s2_ngrams:
			if s2_ngram==s1_ngram:
				return True
	return False 

"""
s1 = "I eat a lot of apples everyday"
s2 = "I dont get ill because i have lot of apples and oranges everyday"

print(similarity(s1,s2))
"""

dataset_file = open('Dataset_SimpleQA_qualifiers.json','r') #change the file name here to fetch the data from
dataset_decode = json.load(dataset_file)
tab = pd.DataFrame(data=dataset_decode,columns= ['question_id', 'question','question_entity_label','answer_entity_labels','question_relation'])

tab_ent= tab['answer_entity_labels'].apply(pd.Series)
tab_ent = tab_ent.rename(columns  = lambda x : 'tag_' + str(x))
new_tab = pd.concat([tab[:],tab_ent[:]],axis=1)
new_tab = new_tab[new_tab['tag_1'].isna()]
new_tab = new_tab[new_tab['tag_2'].isna()]
new_tab = new_tab.drop(['answer_entity_labels'],axis=1)
new_tab.rename(columns = {'tag_0':'answer_entity_labels'}, inplace=True)
new_tab = new_tab.dropna(axis = 'columns', how= 'all')
new_tab = new_tab.dropna(axis='index', how = 'any')
new_tab = new_tab.reset_index()

#threshold = 0.80

final_tab = pd.DataFrame(columns = tab.columns)

distribution = pd.read_csv("relation_distribution.csv",sep='|')
distribution = distribution.iloc[50:100]

for item in distribution['rel']:
        tab_questions = new_tab[new_tab.question_relation == item]
	tab_questions = tab_questions.reset_index()
	selected_question = tab_questions.head(1)
	for index,q in enumerate(tab_questions['question']):
		checker = False
		ans = tab_questions['question_entity_label'].iloc[index]
        	for i,q_selected in enumerate(selected_question['question']):
			ans_selected = selected_question['question_entity_label'].iloc[i]
			sim = similarity(q,q_selected,ans,ans_selected)
			#print(sim)
                	if sim :
				checker = True
				break
				#print("a")
		if not checker :
			tab_new = tab_questions.iloc[index]
			selected_question = selected_question.append(tab_new)
		
	final_tab = final_tab.append(selected_question)
	#filename = 'question/Dataset_batch_' + item + '.json'
final_tab.to_json(r'Batch/Dataset_batch_2_ent_4.json',orient='records')
print("Yes")                                     
