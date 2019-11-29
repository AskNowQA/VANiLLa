from SPARQLWrapper import SPARQLWrapper , JSON
import pandas as pd
from pandas.io.json import json_normalize
import json



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

dataset_file = open('Dataset_SimpleQA_first1000.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)
#tab = tab.head(10)

tab_type = tab['type_list'].apply(pd.Series)
tab_type = tab_type.rename(columns  = lambda x : 'tag_' + str(x))
final_tab = pd.concat([tab[:],tab_type[:]],axis=1)
final_tab = final_tab.drop(['type_list'],axis=1)
final_tab.rename(columns = {'tag_0':'type_list'}, inplace=True)

final_tab["entity_labels"] = [list() for x in range(len(tab.index))]
final_tab["type_list_label"] =""
i=0
for entity_list in final_tab['entity']:
	entity_label = []
	for entity in entity_list:
		label = getLabels(entity)
		entity_label.append(label)	
	final_tab.set_value(i,'entity_labels', entity_label)
	i=i+1
i=0
for type in final_tab['type_list']:
	type_label = getLabels(type)
	final_tab.set_value(i,'type_list_label', type_label)
        i=i+1

final_tab.to_json(r'Dataset_SimpleQA_first1000_labels.json',orient='records')
print(final_tab.head(1))


