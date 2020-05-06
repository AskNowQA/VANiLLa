# -*- coding: utf-8 -*-
import pandas as pd
import json
from nltk.util import ngrams

def generate_answer(new_answer, new_entity, entity, ans_sen, ans):
    ans_repl = ans_sen.replace(ans.lower(),new_answer.lower())
    entity_repl = ans_repl.replace(entity.lower(),new_entity.lower())
    return entity_repl
    

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
x= "abc"
y = "mno"
ans = "xyz"
entity = "pqs"
ans_sen = "pqs love xyz."
print(generate_answer (x, y, entity, ans_sen,ans))
print(x)
print(y)
print(entity)
print(ans)
print(ans_sen)

"""
dataset_file1 = open('../../data/Postprocessing/Final_Dataset_labels.csv','r')
dataset = pd.read_csv(dataset_file1)
dataset['answer_sentence'] = dataset['answer_sentence'].str.lower()
#print(dataset.columns)

dataset_file2 = open('../../data/Preprocessing/CSQA_version/Dataset_SimpleQA_qualifiers.json','r') #change the file name here to fetch the data from
dataset_decode2 = json.load(dataset_file2)
tab1 = pd.DataFrame(data=dataset_decode2,columns= ['question_id', 'question','question_entity_label','question_relation', 'answer_entity_labels'])

tab_ent= tab1['answer_entity_labels'].apply(pd.Series)
tab_ent = tab_ent.rename(columns  = lambda x : 'tag_' + str(x))
new_tab = pd.concat([tab1[:],tab_ent[:]],axis=1)
new_tab = new_tab[new_tab['tag_1'].isnull()]
new_tab = new_tab[new_tab['tag_2'].isnull()]
new_tab = new_tab.drop(['answer_entity_labels'],axis=1)
new_tab.rename(columns = {'tag_0':'answer'}, inplace=True)
new_tab = new_tab.dropna(axis = 'columns', how= 'all')
new_tab = new_tab.dropna(axis='index', how = 'any')
tab1 = new_tab
#tab1 = tab1.sort_values(by=['question_id'])

dataset_file3 = open('../../data/Preprocessing/SimpleQuestionWikidata_version/Dataset_SimpleQA_labels.json','r') #change the file name here to fetch the data from
dataset_decode3 = json.load(dataset_file3)
tab2 = pd.DataFrame(data=dataset_decode3,columns= ['question_id', 'question','question_entity_label','question_relation', 'answer_entity_label'])
tab2.rename(columns = {'answer_entity_label':'answer'}, inplace =True)
tab2 = tab2.dropna(how = 'any', axis = 'index')

dataset_file4 = open('../../data/Postprocessing/final_dataset_relation_distribution.csv','r')
distribution = pd.read_csv(dataset_file4, sep='|')
#print(distribution.columns)

columns= ['question_id', 'question','question_entity_label','question_relation', 'answer', 'answer_sentence']
tab_predicate = dataset

for item in distribution['rel']:
    dataset_questions = dataset[dataset.question_relation == item]
    dataset_questions = dataset_questions.reset_index(drop = True)
    
    tab1_questions = tab1[tab1.question_relation == item]
    tab1__questions = tab1_questions.reset_index(drop = True)
    
    tab2_questions = tab2[tab2.question_relation == item]
    tab2__questions = tab2_questions.reset_index(drop = True)
    
    for i,q in enumerate(dataset_questions['question']):
        q_entity = dataset_questions['question_entity_label'].iloc[i]
        row = dataset_questions.iloc[i]
        answer_sentence = row.answer_sentence
        #print(row.answer)
        
        tab1_similar = pd.DataFrame(columns = columns)
        tab2_similar = pd.DataFrame(columns = columns)
        
        for j,q1 in enumerate(tab1_questions['question']):
            q1_entity = tab1_questions['question_entity_label'].iloc[j]
            q1_id = tab1_questions['question_id'].iloc[j]
            
            if(q1_id in tab_predicate['question_id']):
                    continue
                
            sim = similarity(q, q1, q_entity, q1_entity)
            
            if(sim):
                tab1_new = tab1_questions.iloc[j]
                tab1_new['answer_sentence']= generate_answer(tab1_new.answer,
                        tab1_new.question_entity_label,
                        row.question_entity_label,
                        answer_sentence,
                        row.answer)
                #print(tab1_new['answer_sentence'])
                tab1_similar = tab1_similar.append(tab1_new)
                
        for k,q2 in enumerate(tab2_questions['question']):
            q2_entity = tab2_questions['question_entity_label'].iloc[k]
            q2_id = tab2_questions['question_id'].iloc[k]
            
            if(q2_id in tab_predicate['question_id']):
                    continue
            
            sim = similarity(q, q2, q_entity, q2_entity)
            
            if(sim):
                tab2_new = tab2_questions.iloc[k]
                tab2_new['answer_sentence']= generate_answer(tab2_new.answer,
                        tab2_new.question_entity_label,
                        row.question_entity_label,
                        answer_sentence,
                        row.answer)
                tab2_similar = tab2_similar.append(tab2_new)
                
        tab_similar = pd.concat([tab1_similar,tab2_similar], axis = 0).head(10)
        #tab_similar = tab_similar.drop_duplicates(keep = 'first').head(10)
        tab_predicate = tab_predicate.append(tab_similar)
        
#tab_predicate = tab_predicate.drop_duplicates(keep='first')        
tab_predicate.to_csv("../../data/Postprocessing/Extended_Dataset10.csv",index= False)
print(tab_predicate.shape)
print("YES")
