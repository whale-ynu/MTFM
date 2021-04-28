"""
对以下数据进行整理
GetMashup.json
GetIntroduceImformation.json
api-Category_and_Secondary Categories.json
"""
#%%
import os
import sys
import json
import re
import pandas as pd
import time
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src.utils import tokenize

# mashup文件
mashup_description_file = 'data/source_data_v1/mashup_document.txt'
mashup_tag_file = 'data/source_data_v1/mashup_tag.txt'
mashup_used_api_file = 'data/source_data_v1/mashup_used_api.txt'

# 获取mashup描述
mashup_name = []
mashup_description = []
data_tmp = json.loads(open(mashup_description_file, 'r', encoding='utf-8').read())
for key, value in data_tmp.items():
    mashup_name.append(key)
    mashup_description.append(value)

# 获取mashup的tag和category
mashup_tag_category = []
data_tmp = json.loads(open(mashup_tag_file, 'r', encoding='utf-8').read())
for mashup in mashup_name:
    if mashup in data_tmp.keys():
        mashup_tag_category.append(data_tmp[mashup])
    else:
        mashup_tag_category.append([])

# 获取mashup的tag
mashup_tag = []
for tags in mashup_tag_category:
    tag_list = []
    for tag in tags:
        if re.search(pattern=r'_tag', string=tag):
            pass
        else:
            tag_list.append(tag)

    mashup_tag.append(set(tag_list))


# 获取mashup的category
mashup_category = []
for tags in mashup_tag_category:
    category_list = []
    for tag in tags:
        if re.search(pattern=r'_category', string=tag):
            category_list.append(tag)

    mashup_category.append(set(category_list))


# 获取mashup所用api
mashup_used_api = json.loads(open(mashup_used_api_file, 'r', encoding='utf-8').read())


# api文件
api_description_file = 'data/source_data_v1/api_document.txt'
api_tag_file = 'data/source_data_v1/api_tag.txt'

# 获取api描述
data_tmp = json.loads(open(file=api_description_file, mode='r', encoding='utf-8').read())
api_name = []
api_description = []
for key, value in data_tmp.items():
    api_name.append(key)
    api_description.append(value)

# 获取api的tag和category
api_tag_category = []
data_tmp = json.loads(open(api_tag_file, 'r', encoding='utf-8').read())
for api in api_name:
    if api in data_tmp.keys():
        api_tag_category.append(data_tmp[api])
    else:
        api_tag_category.append([])


# 获取api的tag
api_tag = []
for tags in api_tag_category:
    tag_list = []
    for tag in tags:
        if re.search(pattern=r'_tag', string=tag):
            pass
        else:
            tag_list.append(tag)

    api_tag.append(set(tag_list))


# 获取api的category
api_category = []
for tags in api_tag_category:
    category_list = []
    for tag in tags:
        if re.search(pattern=r'_category', string=tag):
            category_list.append(tag)
    api_category.append(set(category_list))

# 整理mashup信息
mashup_list_v1 = []
mashup_count = 0
for i, mashup in enumerate(mashup_name):
    try:
        sample = [mashup,
                  mashup_description[i],
                  list(mashup_category[i]),
                  list(mashup_used_api[mashup])
        ]

        mashup_list_v1.append(sample)
    except:
        mashup_count += 1

# 整理API信息
api_list_v1 = []
api_count = 0
for i, api in enumerate(api_name):
    try:
        sample = [api,
                  api_description[i],
                  list(api_category[i])
        ]
        api_list_v1.append(sample)
    except:
        api_count += 1


# mashup_list_v1
for mashup in mashup_list_v1:
    # 处理mashup_name
    mashup[0] = mashup[0].split('>')[0].split('/')[4].split('_')[0].replace(' ', '-').lower()
    # 处理mashup_des
    mashup[1] = tokenize(mashup[1])
    # 处理mashup_category
    for index, tag in enumerate(mashup[2]):
        mashup[2][index] = tag.split('>')[0].split('/')[4].split('_')[0].replace(' ', '-').lower()
    # 处理 mashup_used_api
    for index, api in enumerate(mashup[3]):
        mashup[3][index] = api.split('>')[0].split('/')[4].split('_')[0].replace(' ', '-').lower()


# api_list_v1
for api in api_list_v1:
    api[0] = api[0].split('>')[0].split('/')[4].split('_')[0].replace(' ', '-').lower()
    api[1] = tokenize(api[1])
    for index, tag in enumerate(api[2]):
        api[2][index] = tag.split('>')[0].split('/')[4].split('_')[0].replace(' ', '-').lower()


with open('data/source_data/GetMashup.json', mode='r', encoding='utf-8') as f:
    GetMashup = f.readlines()
with open('data/source_data/GetIntroduceImformation.json', mode='r', encoding='utf-8') as f:
    GetIntroduceInformation = f.readlines()
with open('data/source_data/api-Category_and_Secondary Categories.json', mode='r', encoding='utf-8') as f:
    GetSummary = f.readlines()

# 处理GetMashup
mashup_list = []
tmp_list = []
for mashup in GetMashup:
    tmp_list.append(json.loads(mashup))
for index, tmp in enumerate(tmp_list):
    for key, value in tmp.items():
        for tmp_key, tmp_value in value.items():
            try:
                if tmp_key is None or tmp_value['mashup imformation'] is None:
                    mashup_count += 1
                else:
                    mashup_list.append([tmp_key,
                                        tmp_value['mashup imformation'],
                                        list(set(tmp_value['tags'] + tmp_value['Categories'])),
                                        tmp_value['Related APIs']])
            except:
                # print(tmp_key, tmp_value, '\n')
                mashup_count += 1

# 处理GetIntroduceInformation和GetSummary
tmp_list = []
for api in GetIntroduceInformation:
    tmp_list.append(json.loads(api))
tmp_list_2 = []
for api_2 in GetSummary:
    tmp_list_2.append(json.loads(api_2))

api_list = []
for tmp1, tmp2 in zip(tmp_list, tmp_list_2):
    assert tmp1.keys() == tmp2.keys()
    try:
        for key, value in tmp1.items():
            if key is None or value['news_text'] is None:
                api_count += 1
            else:
                api_list.append([key, value["news_text"], list(set([value["tags"]] + tmp2[key]))])
    except:
        api_count += 1


for api in api_list:
    # 处理api_name
    api[0] = api[0].split('/')[4].lower().replace(' ', '-')
    # 处理api_description
    api[1] = tokenize(api[1])
    # 处理api_category
    api[2] = [category.lower().replace(' ', '-') for category in api[2]]


for mashup in mashup_list:
    # 处理mashup_name
    mashup[0] = mashup[0].lower().replace(' ', '-')
    # 处理mashup_description
    mashup[1] = tokenize(mashup[1])
    # 处理mashup_category
    mashup[2] = [category.lower().replace(' ', '-') for category in mashup[2]]
    # 处理mashup_used_api
    mashup[3] = [api.lower().replace(' ', '-') for api in mashup[3]]


api_list.extend(api_list_v1)
mashup_list.extend(mashup_list_v1)
api_name = []
api_description = []
api_category = []
used_api_list = []
mashup_name = []
mashup_description = []
mashup_category = []
mashup_used_api = []
category_list = []

for api in api_list:
    if len(api[0]) == 0 or len(api[1]) < 2 or len(api[2]) == 0:
        api_count += 1
        print(api)
    else:
        if api[0] not in api_name:
            api_name.append(api[0])
            api_description.append(api[1])
            api_category.append(api[2])
            for category in api[2]:
                category_list.append(category)

for mashup in mashup_list:
    if len(mashup[0]) == 0 or len(mashup[1]) < 2 or len(mashup[2]) == 0 or len(set(mashup[3]) & set(api_name)) == 0:
        mashup_count += 1
        print(mashup)
    else:
        if mashup[0] not in mashup_name:
            mashup_name.append(mashup[0])
            mashup_description.append(mashup[1])
            mashup_category.append(mashup[2])
            for category in mashup[2]:
                category_list.append(category)
            tmp_list = []
            for used_api in mashup[3]:
                if used_api in api_name:
                    tmp_list.append(used_api)
                    used_api_list.append(used_api)
            mashup_used_api.append(tmp_list)

used_api_list = list(set(used_api_list))
category_list = list(set(category_list))
# print('处理数据集：', time.time() - start_time)
print('无效的mashup数量： %d' % mashup_count)
print('无效的api数量： %d' % api_count)

#%%
# 写入文件
with open('data/mashup_name.json', 'w') as f:
    json.dump(mashup_name, f)
with open('data/mashup_description.json', 'w') as f:
    json.dump(mashup_description, f)
with open('data/mashup_used_api.json', 'w') as f:
    json.dump(mashup_used_api, f)
with open('data/mashup_category.json', 'w') as f:
    json.dump(mashup_category, f)

with open('data/api_name.json', 'w') as f:
    json.dump(api_name, f)
with open('data/api_description.json', 'w') as f:
    json.dump(api_description, f)
with open('data/api_category.json', 'w') as f:
    json.dump(api_category, f)

with open('data/used_api_list.json', 'w') as f:
    json.dump(used_api_list, f)
with open('data/category_list.json', 'w') as f:
    json.dump(category_list, f)

