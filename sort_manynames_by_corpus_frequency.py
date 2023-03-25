from collections import Counter
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import pickle
import json
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.width', 1000)

def choose_newname(row):
    # if altname is NaN，then choose topname
    if pd.isna(row['altname']):
        return row['topname']
    else:

        # if one of topname and altname belongs to low-frequency, and the other belongs to middle/high frequency, then choose the one belongs to middle/high frequency
        if row['altname'] in low_freq_words:
            if row['topname'] in high_freq_words or row['topname'] in middle_freq_words:
                return row['topname']
        elif row['topname'] in low_freq_words:
            if row['altname'] in high_freq_words or row['altname'] in middle_freq_words:
                return row['altname']
        # if topname and altname both belong to low-frequency or middle and high frequency，then randomly assign topname or altname to newname
        else:
            return random.choice([row['topname'], row['altname']])

def choose_newname2(row):
    # if altname is NaN，then choose topname
    if pd.isna(row['altname']):
        return row['topname']
    else:
        if int(data.get(row['topname'])) > int(data.get(row['altname'])):
            return row['altname']
        else:
            return row['topname']


# The number is the word frequency per million tokens.
# all names, not just for top names
data = pickle.load(open("dic_freq.pkl", "rb"))
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


list_of_domain = ["people","clothing","home","buildings","food","vehicles","animals_plants"]

# 7 domains: "people","clothing","home","buildings","food","vehicles","animals_plants"
# get all names in each domain
df_people = df[df.domain == "people"].copy()
people_names = list(set([k for d in df_people["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_clothing = df[df.domain == "clothing"].copy()
clothing_names = list(set([k for d in df_clothing["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_home = df[df.domain == "home"].copy()
home_names = list(set([k for d in df_home["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_buildings = df[df.domain == "buildings"].copy()
buildings_names = list(set([k for d in df_buildings["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_food = df[df.domain == "food"].copy()
food_names = list(set([k for d in df_food["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_vehicles = df[df.domain == "vehicles"].copy()
vehicles_names = list(set([k for d in df_vehicles["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

df_animals_plants = df[df.domain == "animals_plants"].copy()
animals_plants_names = list(set([k for d in df_animals_plants["responses"].apply(lambda x: json.loads(x)) for k in d.keys()]))

list_of_domain_df_in_manynames = [df_people,df_clothing,df_home,df_buildings,df_food,df_vehicles,df_animals_plants]

#### frequency dic for each domain
people_dic = {key: data[key] for key in people_names if key in data}
clothing_dic = {key: data[key] for key in clothing_names if key in data}
home_dic = {key: data[key] for key in home_names if key in data}
buildings_dic = {key: data[key] for key in buildings_names if key in data}
food_dic = {key: data[key] for key in food_names if key in data}
vehicles_dic = {key: data[key] for key in vehicles_names if key in data}
animals_plants_dic = {key: data[key] for key in animals_plants_names if key in data}

list_of_domain_dic = [people_dic,clothing_dic,home_dic,buildings_dic,food_dic,vehicles_dic,animals_plants_dic]

log_people_dic = {k: np.log(v+1) for k, v in people_dic.items()}
log_clothing_dic = {k: np.log(v+1) for k, v in clothing_dic.items()}
log_home_dic = {k: np.log(v+1) for k, v in home_dic.items()}
log_buildings_dic = {k: np.log(v+1) for k, v in buildings_dic.items()}
log_food_dic = {k: np.log(v+1) for k, v in food_dic.items()}
log_vehicles_dic = {k: np.log(v+1) for k, v in vehicles_dic.items()}
log_animals_plants_dic = {k: np.log(v+1) for k, v in animals_plants_dic.items()}

list_of_domain_log_dic =[log_people_dic,log_clothing_dic,log_home_dic,log_buildings_dic,log_food_dic,log_vehicles_dic,log_animals_plants_dic]




# # create a list of frequency values
n = 0
result = []
for j,l,i in zip(list_of_domain,list_of_domain_log_dic,list_of_domain_df_in_manynames):
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

    # print("High-frequency words:", high_freq_words)
    # print("Middle-frequency words:", middle_freq_words)
    # print("Low-frequency words:", low_freq_words)
    df_domain = i
    df_domain['newname'] = df_domain.apply(choose_newname, axis=1)
    # print(df_domain[['topname','altname','newname']])
    newname_freq = dict(Counter(df_domain['newname']))
    new_d = dict(sorted(newname_freq.items(), key=lambda x:x[1], reverse=True))
    result.append({'domain': j, 'newname_freq': new_d})
    # lists in manynames-newname assigned
    lists = []
    a = list(new_d.keys())
    list_top10 = a[:12]
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
result_df = pd.DataFrame(result)
print(result_df)
#result_df.to_csv("newname_freq_sorted_by_corpus_log_freq.csv")
#print(df[['topname', 'altname', 'newname']])







