import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import json
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# load corpus frequency
data = pickle.load(open("dic_freq.pkl", "rb"))
print(data)
names = list(data.keys())
frequencies = list(data.values())
# load manynames
df = pd.read_table("manynames.tsv", index_col = 0)

# #plot corpus frequency, including only the top name
# topnames_list = df['topname'].tolist()
# print(topnames_list)
# print(len(set(topnames_list)))
# log_topnames_dic_in_corpus = {k: np.log10(v+1) for k, v in data.items() if k in topnames_list}
# plt.hist(log_all_names_dic.values(), bins=30)
# plt.title('Histogram of topnames corpus_based frequency')
# plt.xlabel('Log10 frequency')
# plt.ylabel('Number of names')
# plt.show()

# #plot manynames frequency, including only the topnames
# topnames_freq = Counter(df['topname'])
# # f = pd.DataFrame.from_records(list(dict(topnames_freq).items()), columns=['name','frequency'])
# manynames_topname_frequency_dic = dict(topnames_freq)
# log_topnames_dic_in_manynames = {k: np.log10(v+1) for k, v in manynames_topname_frequency_dic.items()}
# plt.hist(log_topnames_dic_in_manynames.values(), bins=30)
# plt.title('Histogram of topnames manynames_based frequency')
# plt.xlabel('Log10 frequency')
# plt.ylabel('Number of names')
# plt.show()

#plot naming variation
print(df['H'])
n, bins, patches = plt.hist(df['H'], bins=30)
plt.title('Histogram of naming variation')
plt.xlabel('frequency')
plt.ylabel('Number')
plt.show()