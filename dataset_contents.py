import pandas as pd
from pandas.io.json import json_normalize
import json
from SPARQLWrapper import SPARQLWrapper , JSON

def getLabels(qid):
	query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wd: <http://www.wikidata.org/entity/> SELECT  ?label WHERE { wd:"+qid+" rdfs:label ?label .  FILTER (langMatches( lang(?label), \"EN\" ) ) } LIMIT 1"
	sparql = SPARQLWrapper("http://query.wikidata.org/sparql", agent = 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36')
	sparql.setQuery(query)

	try:
    		sparql.setReturnFormat(JSON)            
    		uri = sparql.query().convert()['results']['bindings'][0]['label']['value']
		return uri

	except Exception as e:
    		print("type error: " + str(e)) 

def getQualifierInfo(sub_id, rel_id, obj_id):
	query = "PREFIX ps: <http://www.wikidata.org/prop/statement/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX p: <http://www.wikidata.org/prop/> SELECT ?q ?t WHERE { wd:"+sub_id+" p:"+rel_id+"  ?s . ?s ps:"+rel_id+" wd:"+obj_id+" . ?s  ?q ?t  . FILTER NOT EXISTS { FILTER(regex(str(?q),\"value\")||regex(str(?q),\"w3\") ||regex(str(?q),\"wikiba.se\")||?t=wd:"+obj_id+") } }"
	sparql = SPARQLWrapper("http://query.wikidata.org/sparql", agent = 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36')
	sparql.setQuery(query)

	try:
		sparql.setReturnFormat(JSON)
		qualifiers = sparql.query().convert()
		for qualifier in qualifiers["results"]["bindings"]:
			 print((qualifier["q"]["value"],qualifier["t"]["value"]))
	except Exception as e:
        	print("type error: " + str(e))

"""
dataset_file = open('Dataset_test.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)

distribution = pd.read_csv("relation_distribution.csv",sep='|')

tab_rel = tab['relation'].apply(pd.Series)
tab_rel = tab_rel.rename(columns  = lambda x : 'tag_' + str(x))
final_tab_rel = pd.concat([tab[:],tab_rel[:]],axis=1)
final_tab_rel = final_tab_rel.drop(['relation'],axis=1)
final_tab_rel.rename(columns = {'tag_0':'relation'}, inplace=True)

rel_label_tab = pd.DataFrame(columns= final_tab_rel.columns)
rel_label_tab["relation_label"]=""

for item in distribution['rel']:
	rel_label = getLabels(item)
	tab_questions = final_tab_rel[final_tab_rel.relation == item]
	tab_questions['relation_label']=rel_label
	rel_label_tab = pd.concat([rel_label_tab[:],tab_questions[:]],axis=0)
rel_label_tab=rel_label_tab.sort_index()

rel_label_tab.to_json(r'Dataset_SimpleQA_rel_labels.json',orient='records')

print(rel_label_tab.head(5))

dataset_file = open('Dataset_SimpleQA_rel_labels.json','r')
dataset_decode = json.load(dataset_file)
rel_label_tab = json_normalize(dataset_decode)

tab_type = rel_label_tab['type_list'].apply(pd.Series)
tab_type = tab_type.rename(columns  = lambda x : 'tag_' + str(x))
final_tab_type = pd.concat([rel_label_tab[:],tab_type[:]],axis=1)
final_tab_type = final_tab_type.drop(['type_list'],axis=1)
final_tab_type.rename(columns = {'tag_0':'type_list'}, inplace=True)

final_tab_type["entity_labels"] = [list() for x in range(len(rel_label_tab.index))]
final_tab_type["type_list_label"] =""

i=0
for entity_list in final_tab_type['entity']:
	entity_label = []
	for entity in entity_list:
		label = getLabels(entity)
		entity_label.append(label)	
	final_tab_type.set_value(i,'entity_labels', entity_label)
	i=i+1
final_tab_type.to_json(r'Dataset_SimpleQA_type_labels.json',orient='records')

print(final_tab_type.head(5))

i=0
dataset_file = open('Dataset_SimpleQA_entity_labels.json','r')
dataset_decode = json.load(dataset_file)
final_tab_type = json_normalize(dataset_decode)

for wikidata_type in final_tab_type['type_list']:
	type_label = getLabels(wikidata_type)
	final_tab_type.set_value(i,'type_list_label', type_label)
        i=i+1

final_tab_type.to_json(r'Dataset_SimpleQA_type_labels.json',orient='records')

"""
dataset_file = open('Dataset_SimpleQA_entity_labels.json','r')
dataset_decode = json.load(dataset_file)
final_tab_answer = json_normalize(dataset_decode)

final_tab_answer["answer_entity_labels"] = [list() for x in range(len(final_tab_answer.index))]

i=0
for answer_entity_list in final_tab_answer['answer_entity']:
        answer_entity_label = []
        for answer_entity in answer_entity_list:
                label = getLabels(answer_entity)
                answer_entity_label.append(label)
	print(answer_entity_label)
        final_tab_answer.set_value(i,'answer_entity_labels', answer_entity_label)
        i=i+1

final_labels_tab = final_tab_answer.drop(["answer"],axis=1)
final_labels_tab.to_json(r'Dataset_SimpleQA_labels.json',orient='records')

print(final_labels_tab.head(5))

#getQualifierInfo("Q202735","P1411","Q107258")
print("Done")

