from train import *
import tensorflow as tf
import numpy as np
from jiwer import wer
from tensorflow import keras

def eval(SAVE_PATH,batch_size, log,label_pad, max_length, num_classes):
    input_data, label_data,seq_len_list,label_data_length = getDataTest(batch_size,label_pad, max_length, num_classes)

    model = keras.models.load_model(SAVE_PATH,
                                    custom_objects={'tf': tf, 'CTCLoss': CTCLoss, 'EditDistance': EditDistance})
    val_logits = model(input_data)

    dist_list_gd = []
    WER_list_gd = []

    dist_list_bs = []
    WER_list_bs = []

    decoded_gd, _ = tf.nn.ctc_greedy_decoder(
        tf.transpose(val_logits, perm=[1, 0, 2]), seq_len_list, merge_repeated=True
    )
    dense_decoded_gd = to_dense(decoded_gd[0])

    decoded_bs, _ = tf.nn.ctc_beam_search_decoder(
        tf.transpose(val_logits, perm=[1, 0, 2]), seq_len_list, beam_width=10)
    dense_decoded_bs = to_dense(decoded_bs[0])

    for i in range(len(label_data)):
        target = ''.join(
            [letter_table[x] for x in label_data[i]])

        predict_gd = ''.join(
            [letter_table[x] for x in dense_decoded_gd[i]])
        predict_bs = ''.join(
            [letter_table[x] for x in dense_decoded_bs[i]])

        dist_gd = levenshteinDistance(target, predict_gd)
        WER_gd = wer(target, predict_gd)

        dist_bs = levenshteinDistance(target, predict_bs)
        WER_bs = wer(target, predict_bs)

        log.info('Target                : "{}"'.format(target))
        log.info('Predict Greedy Search : "{}"'.format(predict_gd))
        log.info('Predict Beam Search : "{}"'.format(predict_bs))

        log.info('CER Greedy Search :  {}\n'.format(dist_gd))
        log.info('WER Greedy Search :  {}\n'.format(WER_gd))

        log.info('CER Beam Search :  {}\n'.format(dist_bs))
        log.info('WER Beam Search :  {}\n'.format(WER_bs))

        dist_list_gd.append(dist_gd)
        WER_list_gd.append(WER_gd)

        dist_list_bs.append(dist_bs)
        WER_list_bs.append(WER_bs)

    log.info('Mean CER Greedy Search: {}'.format(np.mean(np.array(dist_list_gd))))
    log.info('Mean WER Greedy Search:  {}'.format(np.mean(np.array(WER_list_gd))))


    log.info('Mean CER Beam Search: {}'.format(np.mean(np.array(dist_list_bs))))
    log.info('Mean WER Beam Search:  {}'.format(np.mean(np.array(WER_list_bs))))

    return [np.mean(np.array(dist_list_gd)),np.mean(np.array(WER_list_gd)),
            np.mean(np.array(dist_list_bs)),np.mean(np.array(WER_list_bs))]

