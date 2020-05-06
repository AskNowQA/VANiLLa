import pandas as pd
import json

dataset_file1 = open('../../data/AMT/AMT_results/Final_Dataset.json','r') #change the file name here to fettch the data from
tab1 = pd.read_json(dataset_file1, lines=True)
#tab1 = tab1.sort_values(by=['question_id'])                                               
dataset_file2 = open('../../data/Preprocessing/CSQA_version/Dataset_SimpleQA_qualifiers.json','r') #change the file name here to fetch the data from
dataset_decode2 = json.load(dataset_file2)
tab2 = pd.DataFrame(data=dataset_decode2,columns= ['question_id', 'question_entity_label','question_relation'])
#print(tab2.head)
"""
tab_ent= tab2['answer_entity_labels'].apply(pd.Series)
tab_ent = tab_ent.rename(columns  = lambda x : 'tag_' + str(x))
new_tab = pd.concat([tab2[:],tab_ent[:]],axis=1)
new_tab = new_tab[new_tab['tag_1'].isnull()]
new_tab = new_tab[new_tab['tag_2'].isnull()]
new_tab = new_tab.drop(['answer_entity_labels'],axis=1)
new_tab.rename(columns = {'tag_0':'answer_entity_label'}, inplace=True)
new_tab = new_tab.dropna(axis = 'columns', how= 'all')
new_tab = new_tab.dropna(axis='index', how = 'any')
tab2 = tab2.sort_values(by=['question_id'])
"""
dataset_file3 = open('../../data/Preprocessing/SimpleQuestionWikidata_version/Dataset_SimpleQA_labels.json','r') #change the file name here to fetch the data from
dataset_decode3 = json.load(dataset_file3)
tab3 = pd.DataFrame(data=dataset_decode3,columns= ['question_id', 'question_entity_label','question_relation'])
#tab3 =tab3.sort_values(by=['question_id'])

columns = ['question_id','question','answer','answer_sentence','question_entity_label','question_relation']
final_tab = pd.DataFrame(columns =columns)

for i in tab1['question_id']:
    if (i < 70000):
        tab_l = tab2[tab2.question_id == i].reset_index(drop = True).drop(['question_id'], axis = 1).dropna(how = 'any').drop_duplicates(keep= 'first')
        #print(tab_l)
        #print()
        #tab = tab.drop(['question_id'], axis = 1)
        tab_d = tab1[tab1.question_id == i].reset_index(drop = True)
        #print(tab_d)
        #print()
        tab = pd.concat([tab_l[:],tab_d[:]], axis = 1)
        #print(tab)
    else:
        tab_l = tab3[tab3.question_id == i].reset_index(drop = True).drop(['question_id'], axis = 1).dropna(how = 'any').drop_duplicates(keep= 'first')
        #print(tab_l)
        #print(tab_l)
        #print()
        #tab = tab.drop(['question_id'], axis = 1)
        tab_d = tab1[tab1.question_id == i].reset_index(drop = True)
        #print(tab_d)
        #print()
        tab = pd.concat([tab_l[:],tab_d[:]], axis = 1)
        #print(tab)
    #old_size = final_tab.shape[0]
    final_tab = final_tab.append(tab, ignore_index = True)
    #new_size = final_tab.shape[0]
    #if((old_size + 1) != new_size):
     #   print(old_size)
    #print(final_tab.shape)
final_tab = final_tab.dropna(how = 'any')    
#final_tab = final_tab.drop_duplicates(keep='first')
print(final_tab.shape)
final_tab.to_csv('Final_Dataset_labels.csv', index=False)
print("Yes")
