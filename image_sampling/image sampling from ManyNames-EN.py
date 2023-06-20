# %% ---- DEPENDENCIES
import sys
import numpy as np
import pickle
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
import os
import requests
import urllib.request


# %% ---- FUNCTION TO LOAD MANYNAMES.TSV
def load_cleaned_results(filename="manynames.tsv", sep="\t",
                         index_col=None):
    # read tsv
    resdf = pd.read_csv(filename, sep=sep, index_col=index_col)

    # remove any old index columns
    columns = [col for col in resdf.columns if not col.startswith("Unnamed")]
    resdf = resdf[columns]

    # run eval on nested lists/dictionaries
    evcols = ['vg_same_object', 'vg_inadequacy_type',
              'bbox_xywh', 'clusters', 'responses', 'singletons',
              'same_object', 'adequacy_mean', 'inadequacy_type','incorrect']

    for icol in evcols:
        if icol in resdf:
            resdf[icol] = resdf[icol].apply(lambda x: eval(x))

    # return df
    return resdf


# %% ---- DIRECTLY RUN
if __name__ == "__main__":

    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = "manynames.tsv"

    print("Loading data from", fn)
    manynames = load_cleaned_results(filename=fn)
    # print(manynames.head())

##======== step 1: Select images where the intended object is clearly identified =====##
same_object_responses =[]
for i,j,q,m in zip(manynames['same_object'], manynames['topname'], manynames['total_responses'],manynames['incorrect']):
    keys = list(m.keys())
    values = list(m.values())
    top_dic = i.get(j)
    if keys == []:
        number = q
    else:
        number = q
        for k,v in zip(keys, values):
            if top_dic.get(k) is None:
                continue
            elif top_dic.get(k) == 'NA':
                continue
            elif int(top_dic.get(k)) > 0:
                number += int(v)
    same_object_responses.append(number)
manynames['same_object_responses'] = same_object_responses

######## the threshold is 27 ########
manynames_27 = manynames[manynames['same_object_responses'] >= 27]


##========== Step 2: race_identification ===========##
asian_df = pd.read_excel('race_identification_annotation.xlsx', sheet_name='asian')
black_df = pd.read_excel('race_identification_annotation.xlsx', sheet_name='black')

asian_df = asian_df[asian_df['annotation'] == True]
black_df = black_df[black_df['annotation'] == True]

# pick out the images that have been annotated as Asian or Black
asian_manynames_27 = manynames_27.loc[manynames_27['link_mn'].isin(asian_df['link_mn'])]
black_manyname_27 = manynames_27.loc[manynames_27['link_mn'].isin(black_df['link_mn'])]

# summing the asian and black dataframe into non-white dataframe
sampled_images_nonwhite = pd.concat([asian_manynames_27, black_manyname_27])
# sampled_images_nonwhite.to_csv('sampled_images_nonwhite.csv', index=False)


##========== Step 3: automatic sampling ===========##

# =========divide the images into three naming variation bands==========:
# divide low, mid, and high naming variation using quantiles.
# Each naming variation band contains ⅓ of the images.


# Divide images into quantiles
naming_variation_bins= pd.qcut(manynames_27['H'].rank(method='first'), q=3, labels=["low", "mid", "high"],duplicates='drop')

# Create subsets for low, mid, and high variation images
low_variation_images = manynames_27[naming_variation_bins == "low"]
mid_variation_images = manynames_27[naming_variation_bins == "mid"]
high_variation_images = manynames_27[naming_variation_bins == "high"]

# Create subsets for low, mid, and high variation images for Asian and Black

asian_low_variation_images = low_variation_images[low_variation_images['link_mn'].isin(asian_df['link_mn'])]
asian_mid_variation_images = mid_variation_images[mid_variation_images['link_mn'].isin(asian_df['link_mn'])]
asian_high_variation_images = high_variation_images[high_variation_images['link_mn'].isin(asian_df['link_mn'])]

black_low_variation_images = low_variation_images[low_variation_images['link_mn'].isin(black_df['link_mn'])]
black_mid_variation_images = mid_variation_images[mid_variation_images['link_mn'].isin(black_df['link_mn'])]
black_high_variation_images = high_variation_images[high_variation_images['link_mn'].isin(black_df['link_mn'])]

# print('*'*60)
low_variation_images_people = low_variation_images[low_variation_images.domain == "people"].copy()
mid_variation_images_people = mid_variation_images[mid_variation_images.domain == "people"].copy()
high_variation_images_people = high_variation_images[high_variation_images.domain == "people"].copy()


# ========divide the names into three frequency bands (where frequency = corpus-based)======
# in an analogous fashion:
# low, mid, and high naming variation using quantiles.
# Each naming variation band contains ⅓ of the images.

# get the lexical frequency of topname based on the corpus (SUBTLEX-US)
data = pickle.load(open("dic_freq.pkl", "rb"))
names = list(data.keys())
frequencies = list(data.values())
manynames_27_copy = manynames_27.copy()
manynames_27_copy.loc[:, 'topname_frequency'] = manynames_27['topname'].map(dict(zip(names, frequencies)))

# Divide names into quantiles
frequency_bins= pd.qcut(manynames_27_copy['topname_frequency'].rank(method='first'), q=3, labels=["low", "mid", "high"],duplicates='drop')

# Create subsets for low, mid, and high variation images
low_frequency_names = manynames_27_copy[frequency_bins == "low"]
mid_frequency_names = manynames_27_copy[frequency_bins == "mid"]
high_frequency_names = manynames_27_copy[frequency_bins == "high"]

# 'cow' and 'train' are the two names that have overlapping frequency bands
# assign the topname 'cow' to mid_frequency_names
cow_data = low_frequency_names.loc[low_frequency_names['topname'] == 'cow']
low_frequency_names = low_frequency_names.loc[low_frequency_names['topname'] != 'cow']
mid_frequency_names = pd.concat([mid_frequency_names, cow_data])
# assign the topname 'train' to high_frequency_names
train_data = mid_frequency_names.loc[mid_frequency_names['topname'] == 'train']
mid_frequency_names = mid_frequency_names.loc[mid_frequency_names['topname'] != 'train']
high_frequency_names = pd.concat([high_frequency_names, train_data])



asian_low_frequency_names = low_frequency_names[low_frequency_names['link_mn'].isin(asian_df['link_mn'])]
asian_mid_frequency_names = mid_frequency_names[mid_frequency_names['link_mn'].isin(asian_df['link_mn'])]
asian_high_frequency_names = high_frequency_names[high_frequency_names['link_mn'].isin(asian_df['link_mn'])]

black_low_frequency_names = low_frequency_names[low_frequency_names['link_mn'].isin(black_df['link_mn'])]
black_mid_frequency_names = mid_frequency_names[mid_frequency_names['link_mn'].isin(black_df['link_mn'])]
black_high_frequency_names = high_frequency_names[high_frequency_names['link_mn'].isin(black_df['link_mn'])]

# create a new dataframe for people domain images(removing the 93 selected non-white images)
manynames_27_people = manynames_27[manynames_27.domain == "people"].copy()
manynames_27_people_white = manynames_27_people[~manynames_27_people.link_mn.isin(sampled_images_nonwhite['link_mn'])].copy()


# select corresponding from manynames_27_people_white for the same topname and within the same naming variation band
# woman: low 3, mid 29, high 11
# man: low 1, mid 13, high 6
# girl: low 0, mid 2, high 9
# boy: low 0, mid 0, high 11
# child: low 0, mid 0, high 7
# skier: low 0, mid 0, high 1
low_variation_images_people_white = low_variation_images_people[low_variation_images_people.link_mn.isin(manynames_27_people_white['link_mn'])].copy()
mid_variation_images_people_white = mid_variation_images_people[mid_variation_images_people.link_mn.isin(manynames_27_people_white['link_mn'])].copy()
high_variation_images_people_white = high_variation_images_people[high_variation_images_people.link_mn.isin(manynames_27_people_white['link_mn'])].copy()

# put those sampling images into a new dataframe: sampled_images_white
np.random.seed(1234)
sampled_images_white = pd.DataFrame()
for topname in manynames_27_people_white['topname'].unique():
    if topname == 'woman':
        sampled_images_white = pd.concat([sampled_images_white, low_variation_images_people_white[low_variation_images_people_white['topname'] == topname].sample(3)])
        sampled_images_white = pd.concat([sampled_images_white, mid_variation_images_people_white[mid_variation_images_people_white['topname'] == topname].sample(29)])
        sampled_images_white = pd.concat([sampled_images_white, high_variation_images_people_white[high_variation_images_people_white['topname'] == topname].sample(11)])
    if topname == 'man':
        sampled_images_white = pd.concat([sampled_images_white, low_variation_images_people_white[low_variation_images_people_white['topname'] == topname].sample(1)])
        sampled_images_white = pd.concat([sampled_images_white, mid_variation_images_people_white[mid_variation_images_people_white['topname'] == topname].sample(13)])
        sampled_images_white = pd.concat([sampled_images_white, high_variation_images_people_white[high_variation_images_people_white['topname'] == topname].sample(6)])
    if topname == 'girl':
        sampled_images_white = pd.concat([sampled_images_white, mid_variation_images_people_white[mid_variation_images_people_white['topname'] == topname].sample(2)])
        sampled_images_white = pd.concat([sampled_images_white, high_variation_images_people_white[high_variation_images_people_white['topname'] == topname].sample(9)])
    if topname == 'boy':
        sampled_images_white = pd.concat([sampled_images_white, high_variation_images_people_white[high_variation_images_people_white['topname'] == topname].sample(11)])
    if topname == 'child':
        sampled_images_white = pd.concat([sampled_images_white, high_variation_images_people_white[high_variation_images_people_white['topname'] == topname].sample(7)])
    if topname == 'skier':
        sampled_images_white = pd.concat([sampled_images_white, high_variation_images_people_white[high_variation_images_people_white['topname'] == topname].sample(1)])

# sampled_images_white.to_csv('sampled_images_white.csv',index=False)

#========= automatic sampling for the rest of the topnames =========#

# get a new dataframe for the rest images(exclude the sampled_images_white & sampled_images_nonwhite)
sampled_race_identification_images = pd.concat([sampled_images_white,sampled_images_nonwhite])
rest_images = manynames_27[~manynames_27.link_mn.isin(sampled_race_identification_images['link_mn'])].copy()

rest_images_low_variation = rest_images[rest_images.link_mn.isin(low_variation_images['link_mn'])].copy()
rest_images_mid_variation = rest_images[rest_images.link_mn.isin(mid_variation_images['link_mn'])].copy()
rest_images_high_variation = rest_images[rest_images.link_mn.isin(high_variation_images['link_mn'])].copy()

# step 1: supplement the sampled race identification topnames till 10,10,10 for low, mid, high naming variation or the maximum
sampled_images = pd.DataFrame()
for topname in rest_images['topname'].unique():
    if topname == 'woman':
        sampled_images = pd.concat([sampled_images, rest_images_low_variation[rest_images_low_variation['topname'] == topname].sample(4)])
    if topname == 'man':
        sampled_images = pd.concat([sampled_images, rest_images_low_variation[rest_images_low_variation['topname'] == topname].sample(8)])
    if topname == 'girl':
        sampled_images = pd.concat([sampled_images, rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].sample(6)])
    if topname == 'boy':
        sampled_images = pd.concat([sampled_images, rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].sample(10)])
    if topname == 'skier':
        sampled_images = pd.concat([sampled_images, rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].sample(6)])
        sampled_images = pd.concat([sampled_images, rest_images_high_variation[rest_images_high_variation['topname'] == topname].sample(8)])

sampled_images = pd.concat([sampled_images, sampled_race_identification_images])

# step 2: supplement the rest topnames in each corpus frequency group: 10,10,10 for low, mid, high naming variation,
# until they reach 600, 600, 600 for each corpus frequency group
# encounter topnames of "woman", "man", "girl", "boy","child", "skier", skip them

rest_images = manynames_27[~manynames_27.link_mn.isin(sampled_images['link_mn'])].copy()
rest_images_low_variation = rest_images[rest_images.link_mn.isin(low_variation_images['link_mn'])].copy()
rest_images_mid_variation = rest_images[rest_images.link_mn.isin(mid_variation_images['link_mn'])].copy()
rest_images_high_variation = rest_images[rest_images.link_mn.isin(high_variation_images['link_mn'])].copy()

# sample for 7 domain: home, animals_plants, vehicles, people, clothing, food, buildings.

# for low corpus frequency group
def sample_low_freq_images_by_domain(domain):
    # for low corpus frequency group
    random.seed(5678)
    images_low_frequency = pd.DataFrame()
    sampled_topnames_low_frequency = []
    excluded_topnames = ['woman', 'man', 'girl', 'boy', 'child', 'skier']
    threshold_low_frequency = 60

    low_frequency_names_domain = low_frequency_names[low_frequency_names['domain'] == domain]
    while len(images_low_frequency) < threshold_low_frequency:
        # randomly sample a topname that has not been sampled before
        while True:
            topname = random.sample(list(low_frequency_names_domain['topname'].unique()),1)[0]
            if topname in excluded_topnames or topname in sampled_topnames_low_frequency:
                continue
            else:
                break
        sampled_topnames_low_frequency.append(topname)
        low_variation_topnames = rest_images_low_variation[rest_images_low_variation['topname'] == topname].copy()
        mid_variation_topnames = rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].copy()
        high_variation_topnames = rest_images_high_variation[rest_images_high_variation['topname'] == topname].copy()
        if len(low_variation_topnames) > 0:
            images_low_frequency = pd.concat([images_low_frequency, rest_images_low_variation[rest_images_low_variation['topname'] == topname].sample(min(10, len(low_variation_topnames)))])
        if len(mid_variation_topnames) > 0:
            images_low_frequency = pd.concat([images_low_frequency, rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].sample(min(10,len(mid_variation_topnames)))])
        if len(high_variation_topnames) > 0:
            images_low_frequency = pd.concat([images_low_frequency, rest_images_high_variation[rest_images_high_variation['topname'] == topname].sample(min(10,len(high_variation_topnames)))])
        # check if all topnames have been sampled
        if domain == 'people':
            if len(sampled_topnames_low_frequency) == len(low_frequency_names_domain['topname'].unique()) - 1:
                print(f"Reached all possible topnames. Final length of images_low_frequency: {len(images_low_frequency)}")
                break
        else:
            if len(sampled_topnames_low_frequency) == len(low_frequency_names_domain['topname'].unique()):
                print(f"Reached all possible topnames. Final length of images_low_frequency: {len(images_low_frequency)}")
                break
    print(f'len(images_low_frequency):{len(images_low_frequency)}')
    return images_low_frequency

# for mid corpus frequency group
def sample_mid_freq_images_by_domain(domain):
    random.seed(5678)
    images_mid_frequency = pd.DataFrame()
    sampled_topnames_mid_frequency = []
    excluded_topnames = ['woman', 'man', 'girl', 'boy', 'child', 'skier']
    threshold_mid_frequency = 60
    mid_frequency_names_domain = mid_frequency_names[mid_frequency_names['domain'] == domain]
    while len(images_mid_frequency) < threshold_mid_frequency:
        # randomly sample a topname that has not been sampled before
        while True:
            topname = random.sample(list(mid_frequency_names_domain['topname'].unique()),1)[0]
            if topname in excluded_topnames or topname in sampled_topnames_mid_frequency:
                continue
            else:
                break
        sampled_topnames_mid_frequency.append(topname)
        low_variation_topnames = rest_images_low_variation[rest_images_low_variation['topname'] == topname].copy()
        mid_variation_topnames = rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].copy()
        high_variation_topnames = rest_images_high_variation[rest_images_high_variation['topname'] == topname].copy()
        if len(low_variation_topnames) > 0:
            images_mid_frequency = pd.concat([images_mid_frequency, rest_images_low_variation[rest_images_low_variation['topname'] == topname].sample(min(10, len(low_variation_topnames)))])
        if len(mid_variation_topnames) > 0:
            images_mid_frequency = pd.concat([images_mid_frequency, rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].sample(min(10,len(mid_variation_topnames)))])
        if len(high_variation_topnames) > 0:
            images_mid_frequency = pd.concat([images_mid_frequency, rest_images_high_variation[rest_images_high_variation['topname'] == topname].sample(min(10,len(high_variation_topnames)))])
        # check if all topnames have been sampled
        if len(sampled_topnames_mid_frequency) == len(mid_frequency_names_domain['topname'].unique()):
            print(f"Reached all possible topnames. Final length of images_mid_frequency: {len(images_mid_frequency)}")
            break
    print(f'len(images_mid_frequency):{len(images_mid_frequency)}')
    return images_mid_frequency

# for high corpus frequency group
def sample_high_freq_images_by_domain(domain):
    random.seed(5678)
    images_high_frequency = pd.DataFrame()
    sampled_topnames_high_frequency = []
    excluded_topnames = ['woman', 'man', 'girl', 'boy', 'child', 'skier']
    threshold_high_frequency = 60
    high_frequency_names_domain = high_frequency_names[high_frequency_names['domain'] == domain]
    while len(images_high_frequency) < threshold_high_frequency:
        if list(high_frequency_names_domain['topname'].unique()) == []:
            break
        # randomly sample a topname that has not been sampled before
        while True:
            topname = random.sample(list(high_frequency_names_domain['topname'].unique()),1)[0]
            if topname in excluded_topnames or topname in sampled_topnames_high_frequency:
                continue
            else:
                break
        sampled_topnames_high_frequency.append(topname)
        low_variation_topnames = rest_images_low_variation[rest_images_low_variation['topname'] == topname].copy()
        mid_variation_topnames = rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].copy()
        high_variation_topnames = rest_images_high_variation[rest_images_high_variation['topname'] == topname].copy()
        if len(low_variation_topnames) > 0:
            images_high_frequency = pd.concat([images_high_frequency, rest_images_low_variation[rest_images_low_variation['topname'] == topname].sample(min(10, len(low_variation_topnames)))])
        if len(mid_variation_topnames) > 0:
            images_high_frequency = pd.concat([images_high_frequency, rest_images_mid_variation[rest_images_mid_variation['topname'] == topname].sample(min(10,len(mid_variation_topnames)))])
        if len(high_variation_topnames) > 0:
            images_high_frequency = pd.concat([images_high_frequency, rest_images_high_variation[rest_images_high_variation['topname'] == topname].sample(min(10,len(high_variation_topnames)))])
        # check if all topnames have been sampled
        # check if all topnames have been sampled
        if domain == 'people':
            if len(sampled_topnames_high_frequency) == len(high_frequency_names_domain['topname'].unique()) - 5:
                print(f"Reached all possible topnames. Final length of images_high_frequency: {len(images_high_frequency)}")
                break
        else:
            if len(sampled_topnames_high_frequency) == len(high_frequency_names_domain['topname'].unique()):
                print(f"Reached all possible topnames. Final length of images_high_frequency: {len(images_high_frequency)}")
                break
    print(f'len(images_high_frequency):{len(images_high_frequency)}')
    return images_high_frequency

# sample for each domain: home, animals_plants, vehicles, people, clothing, food, buildings.

sample_images_by_home = pd.concat([sample_low_freq_images_by_domain('home'), sample_mid_freq_images_by_domain('home'), sample_high_freq_images_by_domain('home')])
sample_images_by_animals_plants = pd.concat([sample_low_freq_images_by_domain('animals_plants'), sample_mid_freq_images_by_domain('animals_plants'), sample_high_freq_images_by_domain('animals_plants')])
sample_images_by_vehicles = pd.concat([sample_low_freq_images_by_domain('vehicles'), sample_mid_freq_images_by_domain('vehicles'), sample_high_freq_images_by_domain('vehicles')])
sample_images_by_people = pd.concat([sample_low_freq_images_by_domain('people'), sample_mid_freq_images_by_domain('people'), sample_high_freq_images_by_domain('people')])
sample_images_by_clothing = pd.concat([sample_low_freq_images_by_domain('clothing'), sample_mid_freq_images_by_domain('clothing'), sample_high_freq_images_by_domain('clothing')])
sample_images_by_food = pd.concat([sample_low_freq_images_by_domain('food'), sample_mid_freq_images_by_domain('food'), sample_high_freq_images_by_domain('food')])
sample_images_by_buildings = pd.concat([sample_low_freq_images_by_domain('buildings'), sample_mid_freq_images_by_domain('buildings'), sample_high_freq_images_by_domain('buildings')])

images = pd.concat([sample_images_by_home, sample_images_by_animals_plants, sample_images_by_vehicles, sample_images_by_people, sample_images_by_clothing, sample_images_by_food, sample_images_by_buildings,sampled_images])

#=========== drop noise "shoe": only one shoe in the sample =================#
images = images[images['topname'] != 'shoe']
#images.to_csv('images1319.csv', index=False)


#================= plot the distribution of the sampled images (for whole dataset) ==================#


#plot corpus frequency, including only the topname
images_copy = images.copy()
images_copy.loc[:, 'topname_corpus_frequency'] = images_copy['topname'].map(dict(zip(names, frequencies)))
plt.hist(np.log10(images_copy['topname_corpus_frequency']+1), bins=30)
plt.title('Histogram of topnames corpus_based frequency - samples')
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot manynames frequency, including only the topnames
topnames_freq = Counter(images['topname'])
# f = pd.DataFrame.from_records(list(dict(topnames_freq).items()), columns=['name','frequency'])
manynames_topname_frequency_dic = dict(topnames_freq)
images_copy.loc[:, 'topname_manynames_frequency'] = images_copy['topname'].map(topnames_freq)
#log_topnames_dic_in_manynames = {k: np.log10(v+1) for k, v in manynames_topname_frequency_dic.items()}
plt.hist(np.log10(images_copy['topname_manynames_frequency']+1), bins=30)
plt.title('Histogram of topnames manynames_based frequency - samples')
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot naming variation
# print(images['H'])
n, bins, patches = plt.hist(images['H'], bins=30)
plt.title('Histogram of naming variation - samples')
plt.xlabel('naming variation')
plt.ylabel('Number')
plt.show()


# plot the distribution of the sampled images (for low, mid, high frequency groups)
sampled_images_low_frequency = images[images['link_mn'].isin(low_frequency_names['link_mn'])]
sampled_images_mid_frequency = images[images['link_mn'].isin(mid_frequency_names['link_mn'])]
sampled_images_high_frequency = images[images['link_mn'].isin(high_frequency_names['link_mn'])]
# print(len(sampled_images_low_frequency), len(sampled_images_mid_frequency), len(sampled_images_high_frequency))
# print(Counter(sampled_images_low_frequency['topname']))
# print(Counter(sampled_images_mid_frequency['topname']))
# print(Counter(sampled_images_high_frequency['topname']))

low_variation_in_samples = images[images['link_mn'].isin(low_variation_images['link_mn'])]
mid_variation_in_samples = images[images['link_mn'].isin(mid_variation_images['link_mn'])]
high_variation_in_samples = images[images['link_mn'].isin(high_variation_images['link_mn'])]

# print(f'len(low_variation_in_samples):{len(low_variation_in_samples)}')
# print(Counter(low_variation_in_samples['topname']))
# print(f'len(mid_variation_in_samples):{len(mid_variation_in_samples)}')
# print(Counter(mid_variation_in_samples['topname']))
# print(f'len(high_variation_in_samples):{len(high_variation_in_samples)}')
# print(Counter(high_variation_in_samples['topname']))
#
# print(len(high_variation_images))
# print(Counter(high_variation_images['topname']))


## plot for low frequency group of sampled pics

#plot corpus frequency, including only the topname
sampled_images_low_frequency_copy = sampled_images_low_frequency.copy()
sampled_images_low_frequency_copy.loc[:, 'topname_corpus_frequency'] = sampled_images_low_frequency_copy['topname'].map(dict(zip(names, frequencies)))
plt.hist(np.log10(sampled_images_low_frequency_copy['topname_corpus_frequency']+1), bins=30)
plt.title('Histogram of topnames corpus_based frequency - samples - low frequency group', fontsize = 10)
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot manynames frequency, including only the topnames
topnames_freq = Counter(sampled_images_low_frequency['topname'])
# f = pd.DataFrame.from_records(list(dict(topnames_freq).items()), columns=['name','frequency'])
manynames_topname_frequency_dic = dict(topnames_freq)
sampled_images_low_frequency_copy.loc[:, 'topname_manynames_frequency'] = sampled_images_low_frequency_copy['topname'].map(topnames_freq)
#log_topnames_dic_in_manynames = {k: np.log10(v+1) for k, v in manynames_topname_frequency_dic.items()}
plt.hist(np.log10(sampled_images_low_frequency_copy['topname_manynames_frequency']+1), bins=30)
plt.title('Histogram of topnames manynames_based frequency - samples - low frequency group', fontsize = 10)
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot naming variation
#print(sampled_images_low_frequency['H'])
n, bins, patches = plt.hist(sampled_images_low_frequency['H'], bins=30)
plt.title('Histogram of naming variation - samples - low frequency group', fontsize = 10)
plt.xlabel('naming variation')
plt.ylabel('Number')
plt.show()

## plot for mid frequency group of sampled pics

#plot corpus frequency, including only the topname
sampled_images_mid_frequency_copy = sampled_images_mid_frequency.copy()
sampled_images_mid_frequency_copy.loc[:, 'topname_corpus_frequency'] = sampled_images_mid_frequency_copy['topname'].map(dict(zip(names, frequencies)))
plt.hist(np.log10(sampled_images_mid_frequency_copy['topname_corpus_frequency']+1), bins=30)
plt.title('Histogram of topnames corpus_based frequency - samples - mid frequency group', fontsize = 10)
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot manynames frequency, including only the topnames
topnames_freq = Counter(sampled_images_mid_frequency['topname'])
# f = pd.DataFrame.from_records(list(dict(topnames_freq).items()), columns=['name','frequency'])
manynames_topname_frequency_dic = dict(topnames_freq)
sampled_images_mid_frequency_copy.loc[:, 'topname_manynames_frequency'] = sampled_images_mid_frequency_copy['topname'].map(topnames_freq)
#log_topnames_dic_in_manynames = {k: np.log10(v+1) for k, v in manynames_topname_frequency_dic.items()}
plt.hist(np.log10(sampled_images_mid_frequency_copy['topname_manynames_frequency']+1), bins=30)
plt.title('Histogram of topnames manynames_based frequency - samples - mid frequency group', fontsize = 10)
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot naming variation
print(sampled_images_mid_frequency['H'])
n, bins, patches = plt.hist(sampled_images_mid_frequency['H'], bins=30)
plt.title('Histogram of naming variation - samples - mid frequency group', fontsize = 10)
plt.xlabel('naming variation')
plt.ylabel('Number')
plt.show()

## plot for high frequency group of sampled pics

#plot corpus frequency, including only the topname
sampled_images_high_frequency_copy = sampled_images_high_frequency.copy()
sampled_images_high_frequency_copy.loc[:, 'topname_corpus_frequency'] = sampled_images_high_frequency_copy['topname'].map(dict(zip(names, frequencies)))
plt.hist(np.log10(sampled_images_high_frequency_copy['topname_corpus_frequency']+1), bins=30)
plt.title('Histogram of topnames corpus_based frequency - samples - high frequency group', fontsize = 10)
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot manynames frequency, including only the topnames
topnames_freq = Counter(sampled_images_high_frequency['topname'])
# f = pd.DataFrame.from_records(list(dict(topnames_freq).items()), columns=['name','frequency'])
manynames_topname_frequency_dic = dict(topnames_freq)
sampled_images_high_frequency_copy.loc[:, 'topname_manynames_frequency'] = sampled_images_high_frequency_copy['topname'].map(topnames_freq)
#log_topnames_dic_in_manynames = {k: np.log10(v+1) for k, v in manynames_topname_frequency_dic.items()}
plt.hist(np.log10(sampled_images_high_frequency_copy['topname_manynames_frequency']+1), bins=30)
plt.title('Histogram of topnames manynames_based frequency - samples - high frequency group', fontsize = 10)
plt.xlabel('Log10 frequency')
plt.ylabel('Number of pics')
plt.show()

#plot naming variation
print(sampled_images_high_frequency['H'])
n, bins, patches = plt.hist(sampled_images_high_frequency['H'], bins=30)
plt.title('Histogram of naming variation - samples - high frequency group', fontsize = 10)
plt.xlabel('naming variation')
plt.ylabel('Number')
plt.show()

# barplots
# the distribution of images across domains
domain_counts = images['domain'].value_counts()
plt.bar(domain_counts.index, domain_counts.values)
plt.xlabel('Domain')
plt.ylabel('Count')
plt.title('Number of Images by Domain - samples')
plt.xticks(rotation=45)
plt.show()

# the distribution of topnames across domains

domain_topname_counts = images.groupby(['domain', 'topname']).size()
topname_counts = images.groupby('topname').size()
domain_topname_counts = domain_topname_counts.unstack().apply(lambda x: x / topname_counts[x.name], axis=0)
domain_topname_counts = domain_topname_counts.fillna(0)
ax = domain_topname_counts.plot(kind='bar', stacked=True, color = (0, 0.2, 0.6), legend=False)
plt.xlabel('Domain')
plt.ylabel('Count')
plt.title('Number of Topnames by Domain - samples')
plt.xticks(rotation=45)
plt.show()





