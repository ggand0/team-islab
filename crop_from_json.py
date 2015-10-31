import json
import numpy as np
import matplotlib.pyplot as plt

with open('nn_train_data_sample.json') as data_file:
    data = json.load(data_file)
    #print data['imgs']
    b = np.array(data['imgs'][0])
    print b.shape
    c = b.reshape((50,50,3))
    plt.imshow(c)
    plt.show()