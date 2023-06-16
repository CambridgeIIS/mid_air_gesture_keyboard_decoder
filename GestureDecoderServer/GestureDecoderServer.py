import numpy as np
import socket
import time
import os


###################
import GestureDecoder as gd
###################
print('\n=====================\nGestureDecoder Server\n=====================\n')

model_path = './20230616-164057/'
gDecoder = gd.GestureDecoder(model_path,2.5,5)

pre_predict = ''

# Output directory for logged data
output_dir = "logs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
window_width = 50

data_buffer = np.empty((0,2), int)

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind the socket to the port
server_address = ('192.168.1.76', 11001)
#server_address = ('localhost', 11001)
print('\nStarting server on %s port %s' % server_address)
sock.bind(server_address)
ip_address = socket.gethostbyname(socket.gethostname())
print('ip address: %s' % ip_address)

class MessageType:
    Invalid, Acknowledge, Goodbye, DataSample, PredictedStr = range(5)

# Listen for incoming connections
sock.listen(1)
buffer_size = 50960

while True:
    # Wait for a connection
    print('\nWaiting for a connection...')
    connection, client_address = sock.accept()
    
    try:
        print('connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(buffer_size)

            if data:
                print(data.decode())
                splitData = data.decode().split('\t')
                if len(splitData) >= 1:

                    try:

                        messageType = int(splitData[0])

                        if messageType == MessageType.Acknowledge:
                            print('\nReceived ACK message: %s' % splitData[1])
                            connection.sendall(str(int(MessageType.Acknowledge)).encode())

                        elif messageType == MessageType.DataSample:

                            data_sample_str = splitData[1]
                            data_sample = np.fromstring(data_sample_str, dtype=float, sep=",")
                            print('length of data: {}'.format(len(data_sample)))

                            candidate_id = splitData[2]
                            pre_predict = splitData[3]
                            data_frame = data_sample.reshape(-1, 3)

                            predict_gd = gDecoder.predict(data_frame,int(candidate_id),pre_predict)
                            if predict_gd!=1:
                                prediction_msg = "%s\t%s\n" % (str(int(MessageType.PredictedStr)), predict_gd)
                                connection.sendall(prediction_msg.encode())




                        # Goodbye
                        elif messageType == MessageType.Goodbye:
                            print("Goodbye message, saving buffer")
                            output_filename = '{}/data_buffer_{}.csv'.format(output_dir, time.strftime("%Y%m%d_%H%M%S"))
                            np.savetxt(output_filename, data_frame, fmt='%1.4f', delimiter=",")
                            # Reset data buffer
                            connection.sendall(str(int(MessageType.Acknowledge)).encode())
                            break
                    except:
                        messageType = MessageType.DataSample
                        predict_gd = '' + '\n' + '' + '\n' + '' + '\n' + ''
                        prediction_msg = "%s\t%s\n" % (str(int(MessageType.PredictedStr)), predict_gd)
                        connection.sendall(prediction_msg.encode())

            else:
                print('no more data from', client_address)
                break

    finally:
        # Clean up the connection
        connection.close()
