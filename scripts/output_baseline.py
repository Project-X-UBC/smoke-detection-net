import json
import pandas as pd
import numpy as np

GRID_SIZE = 16

with open('../data/full/big_smoke/train.json', 'r') as json_f:
    js = json.load(json_f)

labels = []
for i in js:
    labels.append(i['label'])

df = pd.DataFrame(data=labels)
print(df.head())
print(df.describe())
baseline = 1 - df.sum().sum() / (len(df)*GRID_SIZE)
print("baseline if predict all 0s = " + str(baseline))
