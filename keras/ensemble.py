import csv
import json
import os

datadir = 'data'
csvfile = 'data/submissions/head-64x64_200e_bs32_noda.csv'
output = 'data/submissions/ensemble.csv'

# Read a csv file
probs=[]  # prediction probabilities
files=[]  # filenames
header=[] # header strings
with open(csvfile) as f:
    lis=[line.split() for line in f]        # create a list of lists

    for i,x in enumerate(lis):              #print the list items
      row_elems = x[0].split(',')
      if i == 0:
        header = row_elems
      else:
        files.append(row_elems[0])
        probs.append([ int(x) for x in row_elems[1:] ]) # convert to int

# debug
print len(files)
print len(probs)
print files[0]
print probs[0]
print files[1]
print probs[1]

##################################
#   TODO: add ensemble code here
#################################


# Reconstruct the csv (reusing a part of classifier.py)
with open(os.path.join(datadir, 'sample_submission.csv'), 'r') as fd:
    header = fd.readline()

with open(output, 'w') as fd:
  fd.write(header)
  for i in range(len(probs)):
    fd.write('{},'.format(files[i]))
    row = probs[i]#.tolist()
    fd.write(','.join(['{:.3e}'.format(elem) for elem in row]))
    fd.write('\n')
