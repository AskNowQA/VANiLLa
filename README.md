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
 
{"answer_entity":["Q336264"],"answer_entity_labels":["Kyoto University"],"entity":["Q11661760"],"entity_labels":["Masaru Aoki"],"question":"Which national university is Masaru Aoki an alumni of ?","question_id":1,"relation":"P69","relation_label":"educated at","type_list":"Q265662","type_list_label":"national university"}


Dataset_SimpleQA_rel_labels.json, Dataset_SimpleQA_entity_labels.json, Dataset_SimpleQA_type_labels.json: Some Temporary files

dataset_contents.py : code to fetch the Dataset_SimpleQA_labels_all.json 


Dataset_SimpleQA_qualifiers.json : contains 64371 questions with all labels and qualifier information

(without Qualifier info) :

{
    "question_relation_label": "occupation",
    "question": "What is the job of Carlos Ruiz Ar\u00e1nega ?",
    "question_entity_label": "Carlos Ruiz Ar\u00e1nega",
    "type_list": "Q12737077",
    "question_entity": "Q5042595",
    "answer_entity_list": [
      "Q937857"
    ],
    "answer_entity_labels": [
      "association football player"
    ],
    "type_list_label": "occupation",
    "qualifier_info": NaN,
    "question_relation": "P106",
    "question_id": 56965
  }

(with Qualifier info):

{
    "question_relation_label": "molecular function",
    "question": "Which chemical bond represents molecular function of cyclic 3',5'-adenosine monophosphate phosphodiesterase    SDY_3208 ?",
    "question_entity_label": "Cyclic 3',5'-adenosine monophosphate phosphodiesterase SDY_3208",
    "type_list": "Q44424",
    "question_entity": "Q27474840",
    "answer_entity_list": [
      "Q13667380"
    ],
    "answer_entity_labels": [
      "metal ion binding"
    ],
    "type_list_label": "chemical bond",
    "qualifier_info": {
      "0": {
        "0": {
          "qualifier_pred": "P459",
          "qualifier_pred_label": "determination method",
          "qualifier_value_label": "IEA",
          "qualifier_value": "Q23190881"
        },
        "object": "Q13667380"
      }
    },
    "question_relation": "P680",
    "question_id": 56961
  }

dataset_qualifier.py : code to create Dataset_SimpleQA_qualifiers.json

Dataset_SimpleQA_qualifiers_2500.json: Sample Dataset of 2500 questions (top 50 relations each having 50 questions)  

Dataset_SimpleQA_qualifiers_2000.json: Sample Dataset of 2000 questions (next 50 relations that is 50-99 each having 40 questions)

dataset_qualifier_2500.py: change the parameters and file name to fetch different number of relations and questions and create separate json files. Fetching 2500 question for the top 50 relations(each have 50 questons) to create Dataset_SimpleQA_qualifiers_2500.json. Also used to create Dataset_SimpleQA_qualifiers_2000.json (next 50 relations that is 50-99 each having 40 questions) 

Dataset_batch1_AMT.csv : csv file containing 1922 questions (961 lines each line having 2 questions) without questions with multiple answers 
Format : "Index","UID1","Ques1","Ans1","UID2","Ques2","Ans2"

Dataset_batch2_AMT.csv : csv file containing 1504 questions (752 lines each line having 2 questions) without questions with multiple answers
Format : "Index","UID1","Ques1","Ans1","UID2","Ques2","Ans2"

dataset_csv.py : code to create csv file for AMT task. Change file names to create the different csv files for the different experiment.

Batch_1_results.csv: csv file created from first batch of AMT experiment

Batch_1_results.json : clear json file created from Batch_1_results.csv without metadata

batch1_json.py: creating the json file from Batch_1_results.csv
