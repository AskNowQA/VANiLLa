import pandas as pd
from pandas.io.json import json_normalize
import json
import csv

data = pd.read_csv("Batch_1_results.csv")

tab1 = data[['Input.UID1', 'Input.Ques1', 'Input.Ans1', 'Answer.ans_sen_1']]
tab1.rename(columns = {'Input.UID1': 'question_id', 'Input.Ques1':'question', 'Input.Ans1':'answer', 'Answer.ans_sen_1':'answer_sentence'},inplace = True)

tab2 = data [['Input.UID2', 'Input.Ques2', 'Input.Ans2',  'Answer.ans_sen_2']]
tab2.rename(columns = {'Input.UID2': 'question_id', 'Input.Ques2':'question', 'Input.Ans2':'answer', 'Answer.ans_sen_2':'answer_sentence'},inplace = True)

final_tab = pd.concat([tab1,tab2],axis = 0)

#print (list(tab1.columns))
#print (tab1.head(3))
#print(tab1.describe)

#print (list(tab2.columns))
#print (tab2.head(3))
#print(tab2.describe)

#print (list(final_tab.columns))
#print (final_tab.head(3))
#print(final_tab.describe)

final_tab.to_json(r'Batch_1_results.json',orient='records')
print("yes")








