import pandas as pd
from pandas.io.json import json_normalize
import json
from SPARQLWrapper import SPARQLWrapper , JSON

def getLabel(id):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wd: <http://www.wikidata.org/entity/> SELECT  ?label WHERE { wd:"+id+" rdfs:label ?label .  FILTER (langMatches( lang(?label), \"EN\" ) ) } LIMIT 1"
        sparql = SPARQLWrapper("http://query.wikidata.org/sparql", agent = 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36')
        sparql.setQuery(query)

        try:
                sparql.setReturnFormat(JSON)
                uri = sparql.query().convert()['results']['bindings'][0]['label']['value']
                return uri

        except Exception as e:
                print("type error: " + str(e))

dataset_file = open('test.json','r')
dataset_decode = json.load(dataset_file)
tab = json_normalize(dataset_decode)

distribution = pd.read_csv("relation_distribution.csv",sep='|').head(50)

tab_rel = tab['relation'].apply(pd.Series)
tab_rel = tab_rel.rename(columns  = lambda x : 'tag_' + str(x))
final_tab = pd.concat([tab[:],tab_rel[:]],axis=1)
final_tab = final_tab.drop(['relation'],axis=1)
final_tab.rename(columns = {'tag_0':'relation'}, inplace=True)

final_questions = pd.DataFrame(columns= final_tab.columns)
final_questions["relation_label"]=""

for item in distribution['rel']:
	rel_label = getLabel(item)
	tab_questions = final_tab[final_tab.relation == item].head(20)
	tab_questions['relation_label']=rel_label
	final_questions = pd.concat([final_questions[:],tab_questions[:]],axis=0)
final_questions.to_json(r'Dataset_SimpleQA_first1000.json',orient='records')
print(final_questions.head(3))

