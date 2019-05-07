import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_training_batch(folder, batch_index):

    file_name = folder + '/data_batch_' + str(batch_index)
    dict = unpickle(file_name)

    return dict

def load_testing_batch(folder):

    file_name = folder + '/test_batch'
    dict = unpickle(file_name)

    return dict

def features_reshape(feature):
    feature = np.reshape(feature, (-1,32,32,3), order='F')

    return feature

def display_data_stat(folder,batch_index,data_id):
    file_name = folder + '/data_batch_' + str(batch_index)
    dict = unpickle(file_name)
    features = dict['data']
    features = features_reshape(features)
    get_image = np.squeeze(features[data_id,:,:,:])
    print(get_image.shape)
    plt.imshow(get_image)

def preprocess_data(folder_path):
    # read training data
    vali_ratio = 0.2
    for idx in range(1,6):
        dict = load_training_batch(folder_path, idx)
        features = dict['data']
        labels = dict['labels']
        
        if idx == 1:
            total_features = features
            total_labels = labels
        else:
            total_features = np.concatenate((total_features, features), axis=0)
            total_labels = np.concatenate((total_labels, labels), axis=0)
    
    # seperate validation set
    length = len(total_labels)
    vali_length = int(length*vali_ratio)
    vali_features = total_features[range(vali_length),:]
    vali_labels = np.array(total_labels[range(vali_length)])
    train_features = total_features[vali_length:length,:]
    train_labels = np.array(total_labels[vali_length:length])
    
    # reading test data
    dict = load_testing_batch(folder_path)
    test_features = dict['data']
    test_labels = np.array(dict['labels'])

    # min max normalization
    scaler = MinMaxScaler()
    vali_features = scaler.fit_transform(vali_features)
    #print(scaler.data_max_)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
    print(train_labels.shape)

    # one hot encoding
    train_labels = train_labels.reshape(-1,1)
    vali_labels = vali_labels.reshape(-1,1)
    test_labels = test_labels.reshape(-1,1)

    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    train_labels = onehot_encoder.fit_transform(train_labels)
    vali_labels = onehot_encoder.transform(vali_labels)
    test_labels = onehot_encoder.transform(test_labels)

    
    print(train_labels.shape)

    # save pickle
    with open('train_data.pickle', 'wb') as f:
        pickle.dump((train_features,train_labels), f)

    with open('vali_data.pickle', 'wb') as f:
        pickle.dump((vali_features,vali_labels), f)

    with open('test_data.pickle', 'wb') as f:
        pickle.dump((test_features,test_labels), f)

