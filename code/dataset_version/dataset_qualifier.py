import pandas as pd
from pandas.io.json import json_normalize
import os,json
import numpy as np
from SPARQLWrapper import SPARQLWrapper , JSON
#from io import open

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
                return qualifiers["results"]["bindings"]
		#for qualifier in qualifiers["results"]["bindings"]:
                         #print((qualifier["q"]["value"],qualifier["t"]["value"]))
	except Exception as e:
		print("type error: " + str(e))






#getQualifierInfo("Q202735","P1411","Q107258")
fname = 'Dataset_SimpleQA_qualifiers.json'
dataset_file = open('Dataset_SimpleQA_labels_all.json','r')
dataset_decode = json.load(dataset_file)
tab1 = json_normalize(dataset_decode).sort_values(by=['question_id'])
tab = tab1.drop_duplicates(subset = 'question_id', keep='first')
#print(tab.shape)
#print(tab['question_id'].head(5))

tab_ent= tab['entity'].apply(pd.Series)
tab_ent = tab_ent.rename(columns  = lambda x : 'tag_' + str(x))
new_tab = pd.concat([tab[:],tab_ent[:]],axis=1)
new_tab = new_tab[new_tab['tag_1'].isna()]
new_tab = new_tab.drop(['entity'],axis=1)
new_tab.rename(columns = {'tag_0':'entity'}, inplace=True)
new_tab = new_tab.dropna(axis = 'columns', how= 'all')
#print(new_tab.shape)
#print(new_tab['question_id'].head(5))

tab_ent_label= new_tab['entity_labels'].apply(pd.Series)
tab_ent_label = tab_ent_label.rename(columns  = lambda x : 'tag_label_' + str(x))
final_tab = pd.concat([new_tab[:],tab_ent_label[:]],axis=1)
final_tab = final_tab.drop(['entity_labels'],axis=1)
final_tab.rename(columns = {'tag_label_0':'entity_label'}, inplace=True)
#print(final_tab.shape)
#print(final_tab.head(5))

#tab["qualifier"]=[dict() for x in range(len(tab.index))]

for item in final_tab['question_id']:
	#i = final_tab.iloc[item]
	#print(i)
	sub_entity = final_tab.iloc[item].entity
	relation =  final_tab.iloc[item].relation
	obj_entity = final_tab.iloc[item].answer_entity
	item_dict={}
	item_dict['question_id']=item
	item_dict['question']=final_tab.iloc[item].question
	item_dict['question_entity'] = final_tab.iloc[item].entity
        item_dict['question_entity_label'] = final_tab.iloc[item].entity_label
	item_dict['question_relation'] = final_tab.iloc[item].relation
        item_dict['question_relation_label'] = final_tab.iloc[item].relation_label
	item_dict['type_list'] = final_tab.iloc[item].type_list
        item_dict['type_list_label'] = final_tab.iloc[item].type_list_label
	item_dict['answer_entity_list'] = final_tab.iloc[item].answer_entity
	item_dict['answer_entity_labels'] = final_tab.iloc[item].answer_entity_labels
	item_dict['qualifier_info']={}
	i=0
	for obj in obj_entity:
		j=0
		qualifier_info=getQualifierInfo(sub_entity,relation,obj)
		if qualifier_info:
			item_dict['qualifier_info'][i]={}
			for qualifier in qualifier_info:
				qualifier_dict={}
				qualifier_dict["qualifier_pred"]=qualifier["q"]["value"][39:]
                       		qualifier_dict["qualifier_pred_label"]=getLabels(qualifier["q"]["value"][39:])
				if "http://www.wikidata.org/entity/" in qualifier["t"]["value"]:
					qualifier_dict["qualifier_value"]=qualifier["t"]["value"][31:]
					qualifier_dict["qualifier_value_label"]=getLabels(qualifier["t"]["value"][31:])
				else:
					qualifier_dict["qualifier_value"]=qualifier["t"]["value"]
                                       	qualifier_dict["qualifier_value_label"]= np.NaN
				item_dict['qualifier_info'][i][j]=qualifier_dict
				j +=1
			item_dict['qualifier_info'][i]['object']=obj
			i +=1
	if not item_dict['qualifier_info']:
		item_dict['qualifier_info'] = np.NaN
	#print (item_dict)
	with open(fname) as af:
		data = json.load(af)
		data.append(item_dict)
        with open(fname, 'w') as f:
		json.dump(data, f, indent = 2)
print("YES")
