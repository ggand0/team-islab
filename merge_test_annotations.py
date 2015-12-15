import os
import json

outfilename = 'master_annotations_test.json'
master_json = {}
master_json['annotations'] = []

detector_jsonpath = 'annotations_test/testAnnotations2015-11-14.json'
detector_json = None
with open(detector_jsonpath, 'r') as infile:
  detector_json = json.load(infile)
#print detector_json[0]
#print len(detector_json)

manual_jsonpaths = ['annotations_test/%d_err_annotated.json'%n for n in range(0, 10)]
print len(manual_jsonpaths)
manually_annotated_files = []

# Get manually annotated data and append them into master_json
print 'Loading manually annotated %d files...' % len(manual_jsonpaths)
for path in manual_jsonpaths:
  with open(path, 'r') as infile:
    manual_json = json.load(infile)
  #print manual_json['annotations'][0]
  #print len(manual_json['annotations'])

  for d in manual_json['annotations']:
    manually_annotated_files.append(d['file_name']) # we'll use this later
    master_json['annotations'].append({
      'filename': d['file_name'],
      'annotations': {'height': d['h'], 'width': d['w'], 'y': d['y'], 'x': d['x'], 'class': 'Head'},
      #'class': 'image'
    })
print 'manually annotated files: %d' % len(manually_annotated_files)
#print master_json['annotations'][0]
#print len(master_json['annotations'])

# Append the rest of data (automatically detected ROIs) into master_json
detector_files = []
for d in detector_json:
  if not d['filename'] in manually_annotated_files:
    d['annotations'] = d['annotations'][0]  # don't use array for now
    master_json['annotations'].append(d)


print 'num of output annotations: %d'%len(master_json['annotations'])
#print master_json['annotations'][0]
#print master_json['annotations'][1000]
with open(outfilename, 'w') as outfile:
  json.dump(master_json, outfile)
print 'DONE.'