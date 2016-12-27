import sys
import pandas as pd
import csv
import json
import operator
import cPickle as pickle
import matplotlib.pyplot as plt

#with open("nn_train_datav3_test.json","r") as f:
#    org = json.load(f)
#filenames = org['filenames']

whale_ids = pd.read_csv("data/train.csv")
#print whale_ids

count_dict = {}
for idx, whale_id in enumerate(whale_ids['whaleID']):
    #print whale_id
    count_dict[whale_id] = 0

for idx, whale_id in enumerate(whale_ids['whaleID']):
    #print whale_id
    count_dict[whale_id] += 1

count = 0
for key in count_dict.keys():
    if count_dict[key] >= 2:
        count += 1
print 'number of keys that have more than 10 images: %d' % count