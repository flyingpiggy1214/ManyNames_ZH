# %% ---- DEPENDENCIES
import sys


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 10000)

#load manynames



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
    print(manynames.head())

# Select images where the intended object is clear
print(manynames[['responses','same_object']])
same_object_responses =[]
for i,j,q,m in zip(manynames['same_object'], manynames['topname'], manynames['total_responses'],manynames['incorrect']):
    keys = list(m.keys())
    values = list(m.values())
    print(keys)
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
    print(number)
    same_object_responses.append(number)
manynames['same_object_responses'] = same_object_responses
print(manynames.head())

num_25 = 0
for response in same_object_responses:
    if response >= 25:
        num_25 += 1

num_27 = 0
for response in same_object_responses:
    if response >= 27:
        num_27 += 1

num_30 = 0
for response in same_object_responses:
    if response >= 30:
        num_30 += 1

num_33 = 0
for response in same_object_responses:
    if response >= 33:
        num_33 += 1

print(num_25,num_27,num_30,num_33)
