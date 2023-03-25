import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
import pickle
import json
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# Read the tsv file
df = pd.read_table("manynames.tsv", index_col = 0)
# print(df)
# print(df.columns.values)
# print(df[['        vg_image_id','topname','altname','responses','domain']])
selected_cols = ['        vg_image_id','topname','altname','responses','domain']
#df[selected_cols].to_csv('top&alt.csv', index=False)
top_alt_names_freq = Counter(df['topname']) + Counter(df['altname'])
common_words = top_alt_names_freq.most_common(5)
# print(top_alt_names_freq)
# print(common_words)
f = pd.DataFrame.from_records(list(dict(top_alt_names_freq).items()), columns=['name','frequency'])
f['log_frequency'] = np.log(f['frequency'])

#print(f)

n, bins, patches = plt.hist(f['log_frequency'], range=(0, 9))

plt.xlabel('Log Frequency')
plt.ylabel('Number of names')
for i in range(len(n)):
    plt.text(bins[i], n[i]*1.02, int(n[i]), fontsize=8, horizontalalignment="left")
plt.title('Frequency Distribution of Names (Log Scale)')
plt.show()

# 将 log_frequency 列中的值按照指定的区间进行分组，并添加一个新的列来标识每个值所属的区间
bins = [0, 3, 6, 9]
labels = ['0-3', '3-6', '6-9']
f['frequency_range'] = pd.cut(f['log_frequency'], bins=bins, labels=labels, include_lowest=True)

# 根据不同的区间，将 name 列中的值储存在不同的 list 中
names_0_3 = f.loc[f['frequency_range'] == '0-3', 'name'].tolist()
names_3_6 = f.loc[f['frequency_range'] == '3-6', 'name'].tolist()
names_6_9 = f.loc[f['frequency_range'] == '6-9', 'name'].tolist()

# 打印结果
# print(names_0_3)
# print(names_3_6)
# print(names_6_9)


def choose_newname(row):
    # 如果altname为NaN，则选topname
    if pd.isna(row['altname']):
        return row['topname']
    else:
        # 获取topname和altname所属的频率区间
        topname_freq_range = f.loc[f['name'] == row['topname'], 'frequency_range'].values[0]
        altname_freq_range = f.loc[f['name'] == row['altname'], 'frequency_range'].values[0]

        # 如果topname和altname其中有一个属于names_3_6和names_6_9，另一个属于names_0_3，则把属于names_3_6和names_6_9的那个分配给newname
        if topname_freq_range in ['3-6', '6-9'] and altname_freq_range == '0-3':
            return row['topname']
        elif topname_freq_range == '0-3' and altname_freq_range in ['3-6', '6-9']:
            return row['altname']

        # 如果topname和altname都属于names_0_3，或者都在names_3_6和names_6_9，则在topname和altname里面随机选一个分配给newname
        else:
            return random.choice([row['topname'], row['altname']])

df['newname'] = df.apply(choose_newname, axis=1)
#print(df[['topname', 'altname', 'newname']])


list_of_domain = ["people","clothing","home","buildings","food","vehicles","animals_plants"]
result = []
lists = []
for i in list_of_domain:
    df_filtered = df[df.domain == i].copy()  # 创建df的副本
    newname_freq = dict(Counter(df_filtered['newname']))
    new_d = dict(sorted(newname_freq.items(), key=lambda x:x[1], reverse=True))
    a = list(new_d.keys())
    lists.append(a)
    result.append({'domain': i, 'newname_freq': new_d})
#print(lists)

# 将结果列表转换为数据框
result_df = pd.DataFrame(result)
#print(result_df)

# 将数据框输出到CSV文件
#result_df.to_csv('domain_newname_distribution.csv', index=False)


#percentile rank method to see where the most frequent names in ManyNames
# dataset stand in the ranking (e.g., Is the most frequent name "building"
# in the Buildings domain from your "domain_newname_distribution" also considered
# as a High-frequency word if we use the percentile rank method?).

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# The number is the word frequency per million tokens.
# all names, not just for top names
data = pickle.load(open("dic_freq.pkl", "rb"))
#print(data)
names = list(data.keys())
frequencies = list(data.values())

# all response names
df = pd.read_table("manynames.tsv", index_col = 0)



# replace single quotes with double quotes
df['responses'] = df['responses'].apply(lambda x: x.replace("'", "\""))
# iterate over the rows in the 'responses' column and catch any errors, error is in the 1994 row
for i, row in df.iterrows():
    try:
        json.loads(row['responses'])
    except json.JSONDecodeError as e:
        # print the error message and the row number
        print(f"JSONDecodeError: {e.msg} in row {i}")
df.loc[1994, 'responses'] = '{"mound": 11, "pitcher\'s mound": 6, "pitchers mound": 5}'
all_names = list(set([k for d in df["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))


# 7 domains: "people","clothing","home","buildings","food","vehicles","animals_plants"
# get all names in each domain
df_people = df[df.domain == "people"]
people_names = list(set([k for d in df_people["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_clothing = df[df.domain == "clothing"]
clothing_names = list(set([k for d in df_clothing["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_home = df[df.domain == "home"]
home_names = list(set([k for d in df_home["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_buildings = df[df.domain == "buildings"]
buildings_names = list(set([k for d in df_buildings["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_food = df[df.domain == "food"]
food_names = list(set([k for d in df_food["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_vehicles = df[df.domain == "vehicles"]
vehicles_names = list(set([k for d in df_vehicles["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_animals_plants = df[df.domain == "animals_plants"]
animals_plants_names = list(set([k for d in df_animals_plants["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))


#### frequency dic for each domain
people_dic = {key: data[key] for key in people_names if key in data}
clothing_dic = {key: data[key] for key in clothing_names if key in data}
home_dic = {key: data[key] for key in home_names if key in data}
buildings_dic = {key: data[key] for key in buildings_names if key in data}
food_dic = {key: data[key] for key in food_names if key in data}
vehicles_dic = {key: data[key] for key in vehicles_names if key in data}
animals_plants_dic = {key: data[key] for key in animals_plants_names if key in data}

list_of_domain_dic = [people_dic,clothing_dic,home_dic,buildings_dic,food_dic,vehicles_dic,animals_plants_dic]
# # create a list of frequency values

n = 0
for i,j,l in zip(lists, list_of_domain,list_of_domain_dic):
    print(j)
    print('*'*60)
    freqs = list(l.values())

    # calculate the percentile ranks of the frequency values
    percentiles = [0, 25, 50, 75, 100]
    percentile_values = [np.percentile(freqs, p) for p in percentiles]

    # categorize the words into high, middle, and low frequency levels based on their percentile ranks
    high_freq_words = [word for word, freq in l.items() if freq >= percentile_values[-2]]
    middle_freq_words = [word for word, freq in l.items() if
                         percentile_values[1] < freq < percentile_values[-2]]
    low_freq_words = [word for word, freq in l.items() if freq <= percentile_values[1]]

    print("High-frequency words:", high_freq_words)
    print("Middle-frequency words:", middle_freq_words)
    print("Low-frequency words:", low_freq_words)
    list_top10 = lists[n][:10]
    High_frequency_words_in_list_top10 = []
    Middle_frequency_words_in_list_top10 = []
    Low_frequency_words_in_list_top10 = []
    for name in list_top10:
        if name in high_freq_words:
            High_frequency_words_in_list_top10.append(name)
        elif name in middle_freq_words:
            Middle_frequency_words_in_list_top10.append(name)
        else:
            Low_frequency_words_in_list_top10.append(name)
    print("High_frequency_words_in_list_top10:", High_frequency_words_in_list_top10)
    print("Middle_frequency_words_in_list_top10:", Middle_frequency_words_in_list_top10)
    print("Low_frequency_words_in_list_top10:", Low_frequency_words_in_list_top10)
    n = n+1



