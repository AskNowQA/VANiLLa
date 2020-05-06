# -*- coding: utf-8 -*-
import pandas as pd

datasetfile = open('../../data/Postprocessing/Final_Dataset_labels.csv','r')
tab = pd.read_csv(datasetfile)

#tab = final_tab['question_relation']

tab.rename(columns = {'question_relation':'rel'}, inplace=True)
print(tab.columns)
#tab.columns = ['rel']
frequency= tab.groupby('rel')['rel'].size().reset_index(name='freq').sort_values(by=['freq'],ascending=False)
print(frequency.shape)
frequency.to_csv('final_dataset_relation_distribution.csv',sep = '|',index=False)
print("yes")
