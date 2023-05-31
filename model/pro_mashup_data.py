import json
import os
import random


def create_train_test():
    curPath = os.path.abspath(os.path.dirname('__file__'))
    rootPath = os.path.split(curPath)[0]

    with open(rootPath + '/data/used_api_list.json', 'r') as f:
            used_api_list = json.load(f)

    used_api_dic={}
    for i,api in enumerate(used_api_list):
        used_api_dic[api]=i
    print(len(used_api_dic))#17182

    with open(rootPath + '/data/mashup_used_api.json', 'r') as f:
        mashup_used_api = json.load(f)
    train_list=[]
    random_list=[]
    random.seed(2023)
    for i,api_list in enumerate(mashup_used_api):
        api_id_list=[]
        for api in api_list:
            api_id_list.append(used_api_dic[api])
        if len(api_list)<=1:
            train_list.append([i,api_id_list[0]])
        else:
            random_one=random.choice(api_id_list)
            train_list.append([i, random_one])
            for api_id in api_id_list:
                if api_id!=random_one:
                    random_list.append([i,api_id])
    sample_num=1000
    test_list=random.sample(random_list,sample_num)
    for i in random_list:
        if i not in test_list:
            train_list.append(i)
    print("random_list:",len(random_list))
    print(len(train_list),len(test_list))
    # with open(rootPath + '/data/train_mashup_api.json', 'w') as f:
    #         json.dump(train_list,f)
    # with open(rootPath + '/data/test_mashup_api.json', 'w') as f:
    #         json.dump(test_list,f)

if __name__=="__main__":
    create_train_test()