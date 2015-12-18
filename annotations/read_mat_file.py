# Read a mat file of annotation session and convert it to a json file.

import scipy as sp
from scipy import io
import json
import ntpath

filename = '../7_err_annotated.mat'

print sp.__version__

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def mat2json(filename, outfilename):
    a = io.loadmat(filename)
    positiveInstances  = a['positiveInstances']


    #print positiveInstances
    print type(positiveInstances)
    print type(positiveInstances[0])
    print type(positiveInstances[0][0])
    print positiveInstances[0][0]
    #print positiveInstances[0][0][0][0]
    #print positiveInstances[0][0][1]
    #print positiveInstances[0][0][1][0][0]
    #print positiveInstances[0][0][1][0][1]#""""""

    json_data = {}
    json_data['annotations'] = []
    for instance in positiveInstances[0]:
      data={}
      data['x'] = str(instance[1][0][0])
      data['y'] = str(instance[1][0][1])
      data['w'] = str(instance[1][0][2])
      data['h'] = str(instance[1][0][3])

      # get filename. extract the filename only from full path
      data['file_name'] = path_leaf(instance[0][0])
      #print data

      json_data['annotations'].append(data)
    print len(json_data['annotations'])

    with open(outfilename, 'w') as outfile:
      json.dump(json_data, outfile)

mat2json(filename, '../7_err_annotated.json')