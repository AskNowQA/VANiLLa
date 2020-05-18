# -*- coding: utf-8 -*-

import pandas as pd
from SPARQLWrapper import SPARQLWrapper , JSON

def getLabels(id):
    query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wd: <http://www.wikidata.org/entity/> SELECT  ?label WHERE { wd:"+id+" rdfs:label ?label .  FILTER (langMatches( lang(?label), \"EN\" ) ) } LIMIT 1"
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql", agent = 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36')
    sparql.setQuery(query)
    
    try:
        sparql.setReturnFormat(JSON)            
        uri = sparql.query().convert()['results']['bindings'][0]['label']['value']
        return uri
    
    except Exception as e:
    		print("type error: " + str(e))


dataset_file1 = open('Final_Dataset_labels.csv','r')
dataset = pd.read_csv(dataset_file1)

sample = dataset.sample(n=700)

#sample = sample.loc[:,['question_id','answer','answer_sentence','question_entity_label','question_relation']]

sample['question_relation_label'] = ""
sample = sample.reset_index(drop = True)

for i,rel in enumerate(sample['question_relation']):
	label = getLabels(rel)
	sample.set_value(i,'question_relation_label', label)

print(sample.shape)

sample = sample.dropna(how = 'any', axis = 'index')
sample = sample.drop(['question_relation', 'question'], axis = 1)
print(sample.shape)

sample = sample.reset_index(drop = True).head(500)
sample.to_csv('survey_sample.csv', index_label = 'index')

print(sample.shape)
