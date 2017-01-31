import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from augment_data import apply_random_augmentation
import os

debug = False

def aug_data_generator(df, batch_size):
    """Generator to train the keras model. Takes a pandas
    dataframe and batch size as inputs. Spits out images
    and steering angles as outputs. Output size is batch size."""
    df_shuffle = shuffle(df)
    df_len = len(df)
    batch_idx = 0
    while True:
        X, y = [], []
        for num_item in range(batch_idx, batch_idx + batch_size):
            im, st_angle = apply_random_augmentation(df_shuffle.iloc[num_item], False)
            if debug and num_item % 50 == 0:
                plt.figure()
                plt.imshow(im)
                plt.title(str(st_angle))
                plt.show()
            X.append(im)
            y.append(st_angle)
        X_train = np.array(X).astype(np.float32)
        y_train = np.array(y).astype(np.float64)
        batch_idx += batch_size
        if (batch_idx + batch_size) > df_len:
            batch_idx = 0
            df_shuffle = shuffle(df)
        yield X_train, y_train

def valid_data_generator(df, batch_size):
    """Data generator for validation data. No augmentation is done"""
    df_shuffle = shuffle(df)
    df_len = len(df)
    batch_idx = 0
    while True:
        X, y = [], []
        for num_item in range(batch_idx, batch_idx + batch_size):
            # No augmentation will be done on this data since valid_flag is being set to True,
            im, st_angle = apply_random_augmentation(df_shuffle.iloc[num_item], True)
            X.append(im)
            y.append(st_angle)
        X_valid = np.array(X).astype(np.float32)
        y_valid = np.array(y).astype(np.float64)
        batch_idx += batch_size
        if (batch_idx + batch_size) > df_len:
            batch_idx = 0
            df_shuffle = shuffle(df)
        yield X_valid, y_valid

def get_dataframe(fnames_list, filepath):
    """Read csv files as pandas dataframes"""
    col = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    df_out = pd.DataFrame()
    for fname in fnames_list:
        df = pd.read_csv(fname)
        df.columns = col
        df['center'] = [filepath + os.path.basename(i) for i in df['center']]
        df['left'] = [filepath + os.path.basename(i) for i in df['left']]
        df['right'] = [filepath + os.path.basename(i) for i in df['right']]
        df_out = df_out.append(df, ignore_index=True)
    return df_out

def prepare_data(df):
    """Method to balance the data"""
    print("Num rows is ", df.shape[0])
    # remove excess zeros
    df = remove_zero_ind(df)
    # boost some lower represented samples
    df = boost_samples(df)
    print("Num rows after zero ind removal ", df.shape[0])
    return df

def remove_zero_ind(df):
    """Method to perfrom undersampling by removing every third
    row with zero steering angle value"""
    st_angles = df['steering']
    zero_ind = np.where(np.logical_and(st_angles >= -0.03, st_angles <= 0.03))  # -0.05 to 0.05
    #print("orig zero ind shape is", len(zero_ind[0]))
    zero_ind_list = zero_ind[0].tolist()
    # throw away half - preferrably alternate value
    del zero_ind_list[1::3]
    #print("discarded zero ind len is", len(zero_ind_list))
    # remove these items from data frame
    df = df.drop(df.index[zero_ind_list])
    num_rows = df.shape[0]
    df.index = range(num_rows)
    return df


def boost_samples(df):
    """Method to perform oversampling by boosting under represented data"""
    st_angles = df['steering']
    oversamp_ind1 = np.where(np.logical_and(st_angles >= -0.5, st_angles < -0.03))
    oversamp_ind2 = np.where(np.logical_and(st_angles > 0.03, st_angles <= 0.5))
    oversamp_ind = oversamp_ind1[0].tolist() + oversamp_ind2[0].tolist()
    pd_oversamp = df.iloc[oversamp_ind, :]
    for i in range(2):
        df = df.append(pd_oversamp)
    num_rows = df.shape[0]
    df.index = range(num_rows)
    # extra boost super low entries
    st_angles = df['steering']
    oversamp_ind1 = np.where(st_angles < -0.5)
    oversamp_ind2 = np.where(st_angles > 0.5)
    oversamp_ind = oversamp_ind1[0].tolist() + oversamp_ind2[0].tolist()
    pd_oversamp = df.iloc[oversamp_ind, :]
    for i in range(4):
        df = df.append(pd_oversamp)
    num_rows = df.shape[0]
    df.index = range(num_rows)
    return df

if debug:
    filename1 = 'train_data/driving_log1.csv'
    filename2 = 'train_data/driving_log2.csv'
    filename3 = 'train_data/driving_log3.csv'
    filename4 = 'train_data/driving_log4.csv'
    filenames_list = [filename1, filename2, filename3, filename4]
    filepath = 'train_data/IMG/'
    df = get_dataframe(filenames_list, filepath)
    #df = get_dataframe(filename1, filename2, filepath)#, filename3)#, filename2, filename3)
    df = prepare_data(df)
    X, y = next(aug_data_generator(df, 256))
    print(X.shape, y.shape)
    print("min max img vals",np.min(X), np.max(X))
    import seaborn as sns
    sns.distplot(y, kde=False, bins=20)
    plt.show()


# print(im1.shape)
# plt.imshow(im1)
# plt.show()