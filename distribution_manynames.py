from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file

df = pd.read_table("manynames.tsv", index_col = 0)
print(df)
#list all name pair types without repetition
#7 domains: "people","clothing","home","buildings","food","vehicles","animals_plants"

list_of_domain = ["people","clothing","home","buildings","food","vehicles","animals_plants"]
result = []
lists = []
for i in list_of_domain:
    df_filtered = df[df.domain == i].copy()  # 创建df的副本
    topname_freq = dict(Counter(df_filtered['topname']))
    top_d = dict(sorted(topname_freq.items(), key=lambda x:x[1], reverse=True))
    result.append({'domain': i, 'topname_freq': top_d})

# 将结果列表转换为数据框
result_df = pd.DataFrame(result)
print(result_df)
result_df.to_csv("manynames sorted by topname.csv")



# Group the data by topname
# df = pd.DataFrame.from_records(list(dict(topname_freq).items()), columns=['topname','frequency'])
# df = df.sort_values(by=['frequency'], ascending=False)

# # Plot
# plt.bar(df['topname'], df['frequency'], color='#5cb85c')
# for a,b in zip(df['topname'],df['frequency']):
#     plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=5)
# plt.xticks(rotation='vertical',fontsize=5)
# plt.xlabel('topname')
# plt.ylabel('Absolute Frequency')
# plt.title(f'{domain_chooser} topname frequency')
# # plt.title('overall topname frequency')
# # Show the plot
# plt.savefig(f'{domain_chooser} topname frequency.svg',dpi = 600)
# # plt.savefig('overall topname frequency.svg',dpi = 600)
# plt.show()


