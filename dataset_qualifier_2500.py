import pandas as pd
from pandas.io.json import json_normalize
import json


dataset_file = open('Dataset_SimpleQA_qualifiers.json','r')
dataset_decode = json.load(dataset_file)
final_tab = pd.DataFrame(data=dataset_decode)
#print(final_tab.)
#final_tab = json_normalize(tab)
distribution = pd.read_csv("relation_distribution.csv",sep='|')
distribution = distribution.iloc[50:100] #change the values here to select the section of the relations

final_questions = pd.DataFrame(columns= final_tab.columns)

for item in distribution['rel']:
        tab_questions = final_tab[final_tab.question_relation == item].head(40) #change the value here to select the section of the questions
        final_questions = pd.concat([final_questions[:],tab_questions[:]],axis=0)
final_questions.to_json(r'Dataset_SimpleQA_qualifiers_2000.json',orient='records') #change the file name accordingly
print(final_questions.head(5))
print("Yes")
