import os, json

path_to_json_output = 'train'
js_output = 'test.json'
count = 1
for i in range(0,199):
    path_to_json = os.path.join(path_to_json_output,'QA_')+ str(i)
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js),encoding ='utf-8') as json_file:
            json_text = json.load(json_file)
            my_dict={}
            i = 'false'
            for item in json_text:
                if i == 'true':
                    my_dict["answer"] = item.get('utterance')
                    my_dict["answer_entity"] = item.get('entities_in_utterance')
                    i = 'false'
                    #print(my_dict)
                    with open(os.path.join(path_to_json_output, js_output),encoding ='utf-8') as af:
                        data = json.load(af)
                        data.append(my_dict)
                    with open(os.path.join(path_to_json_output, js_output),'w',encoding ='utf-8') as f:
                        json.dump(data, f, indent = 2)
                    my_dict = {}
                if item.get('question-type')=="Simple Question (Direct)":
                    my_dict["question_id"] = count
                    count =count + 1
                    my_dict["question"]=item.get('utterance')
                    my_dict["entity"]=item.get('entities_in_utterance')
                    my_dict["relation"] = item.get('relations')
                    my_dict["type_list"] = item.get('type_list')
                    i='true'  
                    
print("yes")
