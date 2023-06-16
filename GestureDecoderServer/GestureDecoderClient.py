import numpy as np
import os
import socket
import sys
import time
from jiwer import wer
import re

###################
print('\n==========================\nGestureDecoder Test Client\n==========================\n')

###
# DEV TEST DATA


import pandas as pd
fname = './data/data_all.pkl'
output = pd.read_pickle(fname)
subject_id_list = output['subject id'].to_numpy()
coords_list = output['lift on coords'].to_numpy()
phrase_list = output['phrase'].to_numpy()
word_suggestions_list = output['word suggestions'].to_numpy()
candidate_id_list = output['candidate id'].to_numpy()
time_list = output['time'].to_numpy()
timesteps_list = output['lift on time steps'].to_numpy()

error_index = []
for i in range(np.shape(phrase_list)[0]):
    num_phrase = np.shape(phrase_list[i])[0]
    num_suggestions = np.shape(word_suggestions_list[i])[0]
    if num_phrase != num_suggestions:
        error_index.append(i)

coords_list = np.delete(coords_list, [error_index])
phrase_list = np.delete(phrase_list, [error_index])
word_suggestions_list = np.delete(word_suggestions_list, [error_index])
candidate_id_list = np.delete(candidate_id_list, [error_index])
subject_id_list = np.delete(subject_id_list, [error_index])


###################
# Get server details
ip_address = input("Enter an ip address: ")
port = input("Enter port: ")

# Create a TCP/IP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = (ip_address, int(port))
print('Connect to server at %s port %s' % server_address)
s.connect(server_address)
s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 100)

class MessageType:
    Invalid, Acknowledge, Goodbye, DataSample, PredictedStr = range(5)

# Buffer size
buffer_size = 50960

try:
    while True:
        # Wait for a message    
        msg = input("Enter message: ")
        
        if (msg == "exit"):
            msg = "2"
            s.send(msg.encode('utf-8'))
            break         
        elif (msg == "connection_test"):
            print('\nSend connection test msg.')
            len_msg = "%s\t%d\t" % (str(int(MessageType.Acknowledge)), 123)            
            s.send(len_msg.encode('utf-8'))
            time.sleep(.100)
            print('Received: ', s.recv(buffer_size))
        elif (msg == "goodbye"):
            print('\nSend goodbye msg.')
            len_msg = "%s\t%d\t" % (str(int(MessageType.Goodbye)), 123)
            s.send(len_msg.encode('utf-8'))
            time.sleep(.100)
            print('Received: ', s.recv(buffer_size))
        elif (msg == "gd_test"):
            # Send meta data
            print('\nSend test data sample.')
            wer_list = []
            print('length of the phrase lit:  {}'.format(len(phrase_list)))
            for i, phrase in enumerate(phrase_list):
                candidate_id = str(candidate_id_list[i])
                if int(candidate_id) !=47:
                    continue
                true_phrase = ' '.join(phrase)
                true_phrase = re.sub(' +', ' ', true_phrase)
                print('True Phrase: [{}]\n'.format(true_phrase))
                predicted_phrase = []

                pre_predict = []
                for j, word in enumerate(phrase):
                    data_sample =np.array(coords_list[i][j])[:,:2]
                    data_sample_str = np.array2string(data_sample, precision=10,threshold=np.inf, max_line_width=np.inf, separator=',')
                    data_sample_str = data_sample_str.replace('[','')
                    data_sample_str = data_sample_str.replace(']','')



                    pre_predict.append(word)
                    data_sample_msg = "%s\t%s\t%s\t%s" % (str(int(MessageType.DataSample)), data_sample_str,candidate_id,' '.join(pre_predict))
                    s.send(data_sample_msg.encode('utf-8'))

                    prediction_msg = s.recv(buffer_size)
                    if prediction_msg:
                        split_data = prediction_msg.decode().split('\t')
                        if len(split_data) >= 1:
                            message_type = int(split_data[0])
                            if message_type == MessageType.PredictedStr:
                                split_payload = split_data[1]
                                predicted_phrase.append(str(split_payload.split('\n')[0]))
                predicted_phrase_string = ' '.join(predicted_phrase)
                print('Predicted Phrase: [{}]\n'.format(predicted_phrase_string))
                wer_ = wer(true_phrase,predicted_phrase_string)
                print('wer: {}'.format(wer_))
                wer_list.append(wer_)
            print('average wer: {}'.format(np.mean(wer_list)))
finally:    
    # Clean up the connection
    print('close connection')
    s.close()
