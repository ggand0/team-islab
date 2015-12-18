import sys
import pandas as pd
import csv
import json
import cPickle as pickle

def export_to_csv(preds, filenames, csv_filename):
    print('Exporting to a csv file...')
    whale_ids=pd.read_csv("data/sample_submission.csv")
    output = pd.DataFrame(columns=whale_ids.columns.values)
    #print type(whale_ids['Image']) # => pandas.Series
    print len(whale_ids['Image'])

    # get the dict
    reverse_map = pickle.load(open("bin/label_map_reverse.bin", "rb"))

    # convert label ids to whale ids and constru ct pandas.DataFrame obj(similar to csv structure)
    # sample_submission-row-base output (keeps the original order)
    for idx, filename in enumerate(whale_ids['Image']):
        row_dict={}
        # set zeros first
        for whale_id in whale_ids.columns.values[1:]:
            row_dict[whale_id] = 0

        # preds and filenames should have the same index for a sample
        index = filenames.index(filename)
        label = preds[index].argmax()
        whale_id = reverse_map[label]
        row_dict[whale_id] = 1
        row_dict['Image'] = filename
        output.loc[idx] = row_dict

        if idx % 1000 == 0:
            print('%d, %d' % (idx, label))

    # preds-base output (breaks the original order)
    """
    for idx, pred in enumerate(preds):
        row_dict={}
        # set zeros first
        for whale_id in whale_ids.columns.values[1:]:
            row_dict[whale_id] = 0

        label = pred.argmax()
        whale_id = reverse_map[label]
        row_dict[whale_id]=1
        row_dict['Image']=filenames[idx]
        #new_row = pd.Series(row_data, index=whale_ids.columns.values)
        output.loc[idx] = row_dict

        if idx % 1000 == 0:
            print('%d, %d' % (idx, label))
    """


    # Export to csv
    print('debug')
    print sys.getsizeof(output)/1000000.0
    output.to_csv(csv_filename)

# debug
if __name__ == '__main__':
    #with open("nn_train_datav3_test.json","r") as f:
    #    org = json.load(f)
    filenames = pickle.load(open('bin/filenames_test.bin', "rb"))
    preds = pickle.load(open('bin/head-64x64_da_preds.bin', "rb"))
    print filenames[0]
    print preds[0]

    export_to_csv(preds, filenames, 'test0.csv')