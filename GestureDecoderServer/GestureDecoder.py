from GestureDecoderUtils import *
# Reduce tensorflow verbosity
import os
# Cut back on TF output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
from autocorrect import Speller
import arpa  # Python 3.4+
import numpy as np
import time

models = arpa.loadf("./lm_mobile_word_3gram_tiny.arpa")
lm_mix = models[0]  # ARPA files may contain several models.
pre_predict = ''

key_locations, tolerence, df, new_key_width, new_key_height = get_location_df(169)

class GestureDecoder:
    def __init__(self, model_path, alpha, num_selections):
        self.spell = Speller()
        self.alpha = alpha
        self.num_selections = num_selections
        start_time = time.time()
        self.model = keras.models.load_model(model_path,
                                        custom_objects={'tf': tf, 'CTCLoss': CTCLoss, 'EditDistance': EditDistance})
        print('Model loaded from: {}'.format(model_path))
        print('model loading time take: {}'.format(time.time() - start_time))

        self.spell = Speller()

        print(tf.config.list_physical_devices('GPU'))        
        # Run a dummy predict, as Keras otherwise slow on first call when using CUDA
        dummy_data_frame = np.zeros((40,2))

        self.predict(dummy_data_frame,169,pre_predict)

    @tf.function
    def serve(self,x):
        return self.model(x, training=False)

    def bs_decode(self, logits_op, seq_len):
        seq_len = tf.cast(seq_len, tf.int32)
        decoded_bs, log_probability = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits_op, perm=[1, 0, 2]), seq_len, beam_width=100, top_paths=self.num_selections)

        return decoded_bs, log_probability

    def predict(self, coords,candidate_id,pre_predict):
        start_time_ = time.time()
        start_time = time.time()

        coords = denoise(coords)
        coords = interpolate(coords, 50)
        start_time_gl = time.time()

        print('get location take: {}'.format(time.time() - start_time_gl))
        location = get_location_fst(coords, key_locations, tolerence )
        coords = normalize_bo(coords, candidate_id)
        features = add_features(coords[:, :2])
        coords = np.concatenate((coords, location, features), axis=1)
        print('data process time take: {}'.format(time.time() - start_time))

        if np.shape(coords)[0] <5:
            return 1
        else:
            post_data = np.expand_dims(coords, 0)
            seq_len = np.expand_dims(50, 0)
            logits_op = self.serve(post_data)
            decoded_bs, log_probability = self.bs_decode(logits_op, seq_len)
            predict_list = []
            log_probability_list = []
            log_probabilities_lm = []
            start_time_lm_auto = time.time()

            for i in range(self.num_selections):
                start_time_td = time.time()
                dense_decoded = to_dense(decoded_bs[i])
                predic_ = ''.join(
                    [letter_table[x] for x in dense_decoded[0]])
                print('to dense take: {}'.format(time.time() - start_time_td))

                start_time_lm = time.time()
                print('raw prediction: {}'.format(predic_))
                predict = self.spell(predic_)
                print('corrected prediction: {}'.format(predict))
                if predict in predict_list:
                    continue
                if len(predict)==0:
                    continue
                predict_list.append(predict)
                log_probability_list.append(log_probability[0][i])
                typed_sentence = pre_predict + ' ' + predict
                #print('typed sentence: {}'.format(typed_sentence))
                log_probabilities_lm.append(lm_mix.log_p(typed_sentence))
                print('auto correction +lm time take: {}'.format(time.time() - start_time_lm))
                #print('=========================\n')

            print('auto correction + lm time total taken: {}'.format(time.time() - start_time_lm_auto))


            start_time_1 = time.time()
            probabilities_bs = np.exp(log_probability_list)
            probabilities_lm = np.exp(log_probabilities_lm)
            # probabilities_combine = alpha*probabilities_bs+beta*probabilities_lm
            probabilities_combine = np.power(np.multiply(probabilities_bs,np.power(probabilities_lm,self.alpha)),(1/(1+self.alpha)))
            zipped_lists = zip(probabilities_combine, predict_list)

            sorted_zipped_lists = sorted(zipped_lists, reverse=True)
            sorted_list1 = [element for _, element in sorted_zipped_lists]
            if len(sorted_list1) == 0:
                sorted_list_join = '' + '\n' + '' + '\n' + '' + '\n' +  ''
            else:
                if sorted_list1[0] == '':
                    sorted_list1[0] = sorted_list1[1]
                if len(sorted_list1)==3:
                    sorted_list_join = sorted_list1[0]+'\n'+sorted_list1[1]+'\n'+sorted_list1[2]+'\n'+''
                elif len(sorted_list1)==2:
                    sorted_list_join = sorted_list1[0] + '\n' + sorted_list1[1] + '\n' +'' + '\n' + \
                                       ''
                elif len(sorted_list1) == 1:
                    sorted_list_join = sorted_list1[0] + '\n' + ''+ '\n' + '' + '\n' + \
                                                         ''
                else:
                    sorted_list_join = '\n'.join(sorted_list1[:4])


            print('sort time take: {}'.format(time.time() - start_time_1))
            print('total time take: {}'.format(time.time() - start_time_))

            return sorted_list_join


