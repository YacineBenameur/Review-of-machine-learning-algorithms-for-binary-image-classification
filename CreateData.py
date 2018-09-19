'''
This rapid code saves pictures of cars and airplanes from CIFAR-10 dataset
for more informations about dataset please refer to the official website :"https://www.cs.toronto.edu/~kriz/cifar.html"

The saved data is a dictionnary containing :
1.X_train: 10000 pictures 32*32 used for training,the first 5000 are airplanes and the others are cars
2.y_train: airplanes are labeled as '0' and cars as '1'
3.X_test: 2000 pictures 32*32 used for testing,the first 1000 are airplanes and the others are cars
4.y_test: airplanes are labeled as '0' and cars as '1'


Author:Benameur Yacine
'''


import numpy as np
import pickle


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def load_cfar10_test(cifar10_dataset_folder_path):
    
    with open(cifar10_dataset_folder_path + '/test_batch' , mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def RGBtogray(data):
    newdata = np.zeros([data.shape[0], data.shape[1], data.shape[2]], float)
    num    = data.shape[0]
    height = data.shape[1]
    width  = data.shape[2]
    for n in range(num):
        for row in range(height):
            for col in range(width):
                newdata[n, row, col] = 0.299 * data[n, row, col, 0] + 0.587 * data[n, row, col, 1] + 0.114 * data[n, row, col, 2]
    return newdata

folder="C:/Users/HP/Desktop/imageClassification/datasets/cifar-10"


data_batch=np.vstack((load_cfar10_batch(folder,1)[0],load_cfar10_batch(folder,2)[0],load_cfar10_batch(folder,3)[0],
                      load_cfar10_batch(folder,4)[0],load_cfar10_batch(folder,5)[0]))
data_labels=np.hstack((load_cfar10_batch(folder,1)[1],load_cfar10_batch(folder,2)[1],load_cfar10_batch(folder,3)[1],
                      load_cfar10_batch(folder,4)[1],load_cfar10_batch(folder,5)[1])).T

test_batch=load_cfar10_test(folder)[0]
test_labels=load_cfar10_test(folder)[1]

X_train=np.vstack((RGBtogray(data_batch[np.array(data_labels)==0]),RGBtogray(data_batch[np.array(data_labels)==1])))
y_train=np.array(np.sum(np.array(data_labels)==0)*[0]+np.sum(np.array(data_labels)==1)*[1])

X_test=np.vstack((RGBtogray(test_batch[np.array(test_labels)==0]),RGBtogray(test_batch[np.array(test_labels)==1])))
y_test=np.array(np.sum(np.array(test_labels)==0)*[0]+np.sum(np.array(test_labels)==1)*[1])

processed_data={'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}

'''save data'''
with open('data', 'wb') as handle:
    pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''load data :
def load_data(filename):
    import pickle
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return (data['X_train'],data['X_test'],data['y_train'],data['y_test'])
X_train, X_test, y_train, y_test =load_data('data')

    '''





