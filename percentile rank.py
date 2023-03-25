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
print(data)
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


# create a list of frequency values
freqs = list(buildings_dic.values())

# calculate the percentile ranks of the frequency values
percentiles = [0, 25, 50, 75, 100]
percentile_values = [np.percentile(freqs, p) for p in percentiles]

# categorize the words into high, middle, and low frequency levels based on their percentile ranks
high_freq_words = [word for word, freq in buildings_dic.items() if freq >= percentile_values[-2]]
middle_freq_words = [word for word, freq in buildings_dic.items() if percentile_values[1] < freq < percentile_values[-2]]
low_freq_words = [word for word, freq in buildings_dic.items() if freq <= percentile_values[1]]

print("High-frequency words:", high_freq_words)
print("Middle-frequency words:", middle_freq_words)
print("Low-frequency words:", low_freq_words)

list_top10_buildings = ['building','bridge','house','dugout','tent','overpass','canopy','home','grill','road']
High_frequency_words_in_list_top10_buildings = []
Middle_frequency_words_in_list_top10_buildings = []
Low_frequency_words_in_list_top10_buildings = []
for name in list_top10_buildings:
    if name in high_freq_words:
        High_frequency_words_in_list_top10_buildings.append(name)
    elif name in middle_freq_words:
        Middle_frequency_words_in_list_top10_buildings.append(name)
    else:
        Low_frequency_words_in_list_top10_buildings.append(name)
print("High_frequency_words_in_list_top10_buildings:",High_frequency_words_in_list_top10_buildings)
print("Middle_frequency_words_in_list_top10_buildings:",Middle_frequency_words_in_list_top10_buildings)
print("Low_frequency_words_in_list_top10_buildings:",Low_frequency_words_in_list_top10_buildings)


