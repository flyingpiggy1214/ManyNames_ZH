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

          
            
# plot
# whole domain
# create a histogram of the log frequency
log_all_names_dic = {k: np.log10(v+1) for k, v in data.items()}

plt.hist(log_all_names_dic.values(), bins=30)
plt.title('Histogram of all_names frequency')
plt.xlabel('Log10 frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency
plt.hist(people_dic.values())
plt.title('Histogram of all_names frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()


# 7 different domains
####### People ###########
# create a histogram of the log frequency
log_people_dic = {k: np.log(v+1) for k, v in people_dic.items()}

plt.hist(log_people_dic.values(), bins=30)
plt.title('Histogram of People frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency 
plt.hist(people_dic.values())
plt.title('Histogram of People frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()


####### Clothing ###########
log_clothing_dic = {k: np.log(v+1) for k, v in clothing_dic.items()}

plt.hist(log_clothing_dic.values(), bins=30)
plt.title('Histogram of Clothing frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency 
plt.hist(clothing_dic.values())
plt.title('Histogram of Clothing frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()

####### "home" ###########
log_home_dic = {k: np.log(v+1) for k, v in home_dic.items()}

plt.hist(log_home_dic.values(), bins=30)
plt.title('Histogram of home frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency
plt.hist(home_dic.values())
plt.title('Histogram of home frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()

####### "buildings" ###########
log_buildings_dic = {k: np.log(v+1) for k, v in buildings_dic.items()}

plt.hist(log_buildings_dic.values(), bins=30)
plt.title('Histogram of buildings frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency
plt.hist(buildings_dic.values())
plt.title('Histogram of buildings frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()

####### "food" ###########
log_food_dic = {k: np.log(v+1) for k, v in food_dic.items()}

plt.hist(log_food_dic.values(), bins=30)
plt.title('Histogram of food frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency
plt.hist(food_dic.values())
plt.title('Histogram of food frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()

####### "vehicles" ###########
log_vehicles_dic = {k: np.log(v+1) for k, v in vehicles_dic.items()}

plt.hist(log_vehicles_dic.values(), bins=30)
plt.title('Histogram of vehicles frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency
plt.hist(vehicles_dic.values())
plt.title('Histogram of vehicles frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()

####### "animals_plants" ###########
log_animals_plants_dic = {k: np.log(v+1) for k, v in animals_plants_dic.items()}

plt.hist(log_animals_plants_dic.values(), bins=30)
plt.title('Histogram of animals_plants frequency')
plt.xlabel('Log frequency')
plt.ylabel('Number of names')
plt.show()

# raw frequency
plt.hist(animals_plants_dic.values())
plt.title('Histogram of animals_plants frequency')
plt.xlabel('frequency')
plt.ylabel('Number of names')
plt.show()