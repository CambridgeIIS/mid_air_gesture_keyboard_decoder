import numpy as np
import tensorflow as tf
from utils import *

def getDataTrain1(label_pad,max_length,num_classes):
    input_data = np.load('data/data_mlgk_train_50_2.npy', allow_pickle=True)
    label_data = np.load('data/label_mlgk_train_50_2.npy', allow_pickle=True)

    padded_input_data = []
    seq_len_list = []

    padded_label_data = []
    label_data_length = []

    for i, v in enumerate(input_data):
        seq_len_list.append(np.shape(v[:int(max_length)])[0])
        v = list(v[:int(max_length)])
        while len(v) < max_length:
            v.append(np.zeros(np.shape(v)[1]))
        padded_input_data.append(np.asarray(v))

        v = label_data[i]
        v = np.array(v)[:int(label_pad)-2]+1
        label_data_length.append(len(v)+2)
        residual = int(label_pad) - v.shape[0]-2
        space_array = np.ones([1]) * (num_classes - 2)
        padding_array = np.zeros([int(residual)])
        padded_label_data.append(
            np.concatenate((space_array,v,space_array, padding_array), axis=0))

    seq_len_list = np.stack(seq_len_list)
    label_data_length = np.stack(label_data_length)

    padded_input_data = np.stack(padded_input_data)
    padded_label_data = np.stack(padded_label_data)

    seq_len_list = np.asarray(seq_len_list).astype(np.int32)
    label_data_length = np.asarray(label_data_length).astype(np.int32)

    padded_input_data = np.asarray(padded_input_data).astype(np.float32)
    padded_label_data =  np.asarray(padded_label_data).astype(np.int32)

    for _ in range(4):
        shuffled_indexes = np.random.permutation(padded_input_data.shape[0])
        padded_input_data = padded_input_data[shuffled_indexes]
        padded_label_data = padded_label_data[shuffled_indexes]
        seq_len_list = seq_len_list[shuffled_indexes]
        label_data_length = label_data_length[shuffled_indexes]


    return [padded_input_data, padded_label_data,seq_len_list,label_data_length]

def tfdata1( data,batch_size, size):
    padded_input_data, label_data, seq_len_list, label_data_length = data[0],data[1],data[2],data[3]

    padded_input_data = padded_input_data[:size]
    label_data = label_data[:size]
    seq_len_list = seq_len_list[:size]
    label_data_length = label_data_length[:size]

    train1_data, valid1_data = np.split(
        padded_input_data, [np.shape(padded_input_data)[0] * 9 // 10])
    train1_label, valid1_label = np.split(
        label_data, [np.shape(label_data)[0] * 9 // 10])
    train1_seq_len_list, valid1_seq_len_list = np.split(
        seq_len_list, [np.shape(seq_len_list)[0] * 9 // 10])
    train1_label_data_length, valid1_label_data_length = np.split(
        label_data_length, [np.shape(label_data_length)[0] * 9 // 10])


    train1Data =  tf.data.Dataset.from_tensor_slices(
        (train1_data, train1_label,train1_seq_len_list,train1_label_data_length)
    )
    train1Data = train1Data.shuffle(buffer_size=np.shape(train1_data)[0])
    train1Data = train1Data.batch(batch_size).prefetch(buffer_size=1)

    valid1Data =  tf.data.Dataset.from_tensor_slices(
        (valid1_data, valid1_label,valid1_seq_len_list,valid1_label_data_length)
    )
    valid1Data = valid1Data.shuffle(buffer_size=np.shape(valid1_data)[0])
    valid1Data = valid1Data.batch(batch_size).prefetch(buffer_size=1)

    return train1Data,valid1Data

def multiple(padded_input_data_,label_data_,multipler):
    padded_input_data = padded_input_data_
    label_data = label_data_
    for i in range(multipler):
        padded_input_data = np.concatenate((padded_input_data, padded_input_data_))
        label_data = np.concatenate((label_data, label_data_))


    return padded_input_data, label_data

def shuffle(padded_input_data, padded_label_data):
    shuffled_indexes = np.random.permutation(padded_input_data.shape[0])
    padded_input_data = padded_input_data[shuffled_indexes]
    padded_label_data = padded_label_data[shuffled_indexes]
    return padded_input_data, padded_label_data


def getDataTest(batch_size,label_pad, max_length, num_classes):
    input_data = np.load('data/data_mlgk_valid_50_2.npy', allow_pickle=True)
    label_data = np.load('data/label_mlgk_valid_50_2.npy', allow_pickle=True)

    padded_input_data = []
    seq_len_list = []

    padded_label_data = []
    label_data_length = []

    for i, v in enumerate(input_data):

        seq_len_list.append(np.shape(v[:int(max_length)])[0])
        v = list(v[:int(max_length)])
        while len(v) < max_length:
            v.append(np.zeros(np.shape(v)[1]))
        padded_input_data.append(np.asarray(v))

        v = label_data[i]
        v = np.array(v)[:int(label_pad)-2]+1
        label_data_length.append(len(v)+2)
        residual = int(label_pad) - v.shape[0]-2
        space_array = np.ones([1]) * (num_classes - 2)
        padding_array = np.zeros([int(residual)])
        padded_label_data.append(
            np.concatenate((space_array,v,space_array, padding_array), axis=0))

    seq_len_list = np.stack(seq_len_list)
    label_data_length = np.stack(label_data_length)

    padded_input_data = np.stack(padded_input_data)
    padded_label_data = np.stack(padded_label_data)

    seq_len_list = np.asarray(seq_len_list).astype(np.int32)
    label_data_length = np.asarray(label_data_length).astype(np.int32)
    padded_input_data = np.asarray(padded_input_data).astype(np.float32)
    padded_label_data =  np.asarray(padded_label_data).astype(np.int32)

    return padded_input_data, padded_label_data,seq_len_list,label_data_length

