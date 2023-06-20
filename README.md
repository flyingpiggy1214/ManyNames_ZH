# ManyNames_ZH
ManyNames_ZH is the Mandarin Chinese version of the [ManyNames](https://github.com/amore-upf/manynames) dataset. It provides up to 20 name annotations for 1319 objects in images selected from ManyNames. For an illustration see the image below.
## Notation
| Abbreviation | Description  |
|  ----  | ----  |
|  MN  | ManyNames  |
|  VG  | VisualGenome |
| domain  | Categorisation of objects into people, animals_plants, vehicles, food, home, buildings, and clothing |
## Data files
The data is provided in a tab-separated text file (.tsv). The first rows contain the column labels. Nested data is stored as Python dictionaries (i.e., "{key: value}").   
  
The columns are labelled as follows (the most important columns are listed first):  
| Column | Description  |
|  ----  | ----  |
|  vg_object_id  | The VG id of the object  |
|  link_mn  | The url to the image, with the object marked |
|  responses  | Correct responses and their counts |
|  domain  | The MN domain of the object  |
|  topname_Mandarin  | The most frequent Mandarin Chinese name of the object |
|  topname_English  | The most frequent English name of the object |
|  H  | The naming agreement measure from Snodgrass and Vanderwart (1980)  |
|  N  | The number of types in the responses |
| familiarity  | Weighted average of corpus-frequency of responses |
| list  | Lists of images assigned to participants |
## Subfolder: image_sampling/
