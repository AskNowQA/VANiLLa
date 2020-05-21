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


dataset_file1 = open('../../data/Postprocessing/Final_Dataset_labels.csv','r')
dataset = pd.read_csv(dataset_file1)

sample= dataset.sample(n=500)

sample = sample.loc[:,['question_id','answer','question_entity_label',
                       'question_relation']]
sample = sample.sort_values(by=['question_id']).reset_index(drop = True)
sample['question_relation_label'] = ""

for i,rel in enumerate(sample['question_relation']):
	label = getLabels(rel)
	sample.set_value(i,'question_relation_label', label)

sample= sample.dropna(how = 'any', axis = 'index')
sample.to_csv('survey_sample.csv', index_label = 'index')

print(sample.shape)
