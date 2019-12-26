# Master-Thesis-NLG

Dataset_test.json : contains simple questions from CSQA datset

dataset_test.py : code for fetching test.json


frequency.csv : contains the distribution of relations in the test.json file

distribution.py : code to fetch frequency.csv


Dataset_SimpleQA_new.json : contains 10000 Simple questions from CSQA (50 relations and 200 questions each) with relation labels

dataset_new.py : code to fetch Dataset_SimpleQA_new.json

Dataset_SimpleQA_first1000.json : contains 1000 simple questions from CSQA ( 50 relations and 20 questions each) with relation labels

dataset_first1000.py : code to fetch Dataset_SimDataset_SimpleQA_first1000.py


Dataset_SimpleQA.json : Dataset containing 10000 Simple questions from CSQA with labels for relation, entity, type_list (50 relations and 200 questions for each)

{"answer":"association football player","answer_entity":["Q937857"],"entity":["Q5984210"],"question":"What does Luis Quijanes do for a living ?","question_id":6,"relation":"P106","relation_label":"occupation","type_list":"Q12737077","entity_labels":["Luis Quijanes"],"type_list_label":"occupation"}

labels.py : code to fetch the Dataset_SimpleQA.json


Dataset_SimpleQA_first1000_labels.json : Dataset containing 1000 Simple questions from CSQA with labels for relation, entity, type_list (50 relations and 20 questions for each)

labels.py : code to fetch the Dataset_SimpleQA_first1000_labels.json


term-count.py: Calculates the word count for 200 questions for all 50 relations

folder Term_Count : contains csv files of term_count for every relation on 200 questions 


term-count_all.py: Calculates the word count for all questions for all 50 relations

folder Term_Count_All : contains csv files of term_count for every relation on all questions

Dataset_SimpleQA_labels_all.json: The dataset with all the labels. 

Dataset_SimpleQA_labels.json, Dataset_SimpleQA_rel_labels.json, Dataset_SimpleQA_entity_labels.json, Dataset_SimpleQA_type_labels.json: Some Temporary files

final_dataset.py and dataset_contents.py : code to fetch the Dataset_SimpleQA_labels_all.json 



