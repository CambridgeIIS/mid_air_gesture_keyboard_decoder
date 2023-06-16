import tensorflow as tf
from model import rnn_att_model, CTCLoss
from tensorflow import keras
from metrics import EditDistance
from getData import tfdata1, getDataTest
import numpy as np
import os
import time
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

letter_table = ['','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','','']
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]/max(len(s1), len(s2))

# @tf.function
def to_dense(tensor):
    tensor = tf.sparse.to_dense(tensor, default_value=0)
    tensor = tf.cast(tensor, tf.int32).numpy()
    return tensor


# @tf.function
def loss(model, x, y,seq_len_list,label_data_length, training):
  y_ = model(x, training=training)
  loss_object = CTCLoss()

  return loss_object(y_true = y,y_pred = y_,data_sequence = seq_len_list,label_seq = label_data_length)

# @tf.function
def grad(model, inputs, targets,seq_len_list,label_data_length):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, seq_len_list,label_data_length,training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# @tf.function
def train_step(epoch, train1Data,model, optimizer,epoch_loss_avg, epoch_accuracy, log):
    # log.info('---------------------------- Start Train Epoch {} ---------------------------------\n'.format(epoch))
    for step, (x_batch_train, y_batch_train,seq_len_list,label_data_length) in enumerate(train1Data):
        loss_value, grads = grad(model, x_batch_train, y_batch_train,seq_len_list,label_data_length)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(loss_value)
        y_pred = model(x_batch_train, training=True)
        epoch_accuracy.update_state(y_batch_train, y_pred,seq_len_list,label_data_length)

        if step % 50 == 0:
            log.info("Step {:03d}: Loss: {:.3f}, Editdistance: {:.3}".format(step,
                                                                             epoch_loss_avg.result(),
                                                                             epoch_accuracy.result()))
    log.info('---------------------------- Finish Train Epoch {} ---------------------------------\n'.format(epoch))


def test(batch_size,model, log,label_pad, max_length, num_classes):
    input_data, label_data,seq_len_list,label_data_length = getDataTest(batch_size,label_pad, max_length, num_classes)
    val_logits = model(input_data)
    for _ in range(10):
        i = int(np.random.randint(0,len(label_data)))
        target = ''.join(
            [letter_table[x] for x in label_data[i]])

        sequence_length = seq_len_list

        decoded_gd, _ = tf.nn.ctc_greedy_decoder(
            tf.transpose(val_logits, perm=[1, 0, 2]), sequence_length, merge_repeated=True
        )
        dense_decoded = to_dense(decoded_gd[0])
        predict_gd = ''.join(
            [letter_table[x] for x in dense_decoded[i]])

        decoded_bs, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(val_logits, perm=[1, 0, 2]), sequence_length, beam_width=10)
        dense_decoded = to_dense(decoded_bs[0])
        predict_bs = ''.join(
            [letter_table[x] for x in dense_decoded[i]])

        dist = levenshteinDistance(target, predict_gd)

        log.info('Target                : "{}"'.format(target))
        log.info('Predict Greedy Search : "{}"'.format(predict_gd))
        log.info('Levenshtein Distance  :  {}\n'.format(dist))
        log.info('Predict Beam Search   : "{}"\n'.format(predict_bs))

# @tf.function
def validation(valid1Data,model, val_accuracy, log, start_time):
    for x_batch_val, y_batch_val, seq_len_list,label_data_length in valid1Data:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_accuracy.update_state(y_batch_val, val_logits, seq_len_list,label_data_length)
    val_acc = val_accuracy.result()
    val_accuracy.reset_states()
    log.info("Validation Editdistance: %.4f" % (float(val_acc)))
    # log.info("Time taken: %.2fs\n" % (time.time() - start_time))

def checkpoint_save(log, checkpoint, manager):
    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % 1 == 0:
        save_path = manager.save()
        log.info("Saved checkpoint for ckpt step {}: {}".format(int(checkpoint.step), save_path))


def train(input_shape2, num_classes, learning_rate,data,
          batch_size, size, EPOCHS, SAVE_PATH,
          restore,log,max_length,label_pad,
          model_size,layer_size,drop_out):

    train1Data,valid1Data = tfdata1(data, batch_size, size)

    number_train = len(train1Data)
    number_valid = len(valid1Data)

    log.info('number of train samples {}'.format(number_train))
    log.info('number of valid samples {}'.format(number_valid))

    # purpose: save and restore models
    checkpoint_path = SAVE_PATH +"training_checkpoints/"

    model = rnn_att_model(input_shape2 = input_shape2, output_shape = num_classes, dropout = drop_out,num_units = model_size,num_layers=layer_size)
    model.summary()
    optimizer = keras.optimizers.RMSprop(learning_rate)

    checkpoint = tf.train.Checkpoint(step = tf.Variable(1),
                                     model = model,
                                     generator_optimizer=optimizer)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        log.info('check point not exist {}'.format(checkpoint_path))
    else:
        log.info('check point {}'.format(checkpoint_path))
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)
    if restore:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            log.info("Restored from {}".format(manager.latest_checkpoint))
        else:
            log.info("Initializing from scratch.")

    #------------------ inspection train mode ------------------
    train_loss_results = []
    train_accuracy_results = []
    log.info('Save Path : {}'.format(SAVE_PATH))
    for epoch in range(EPOCHS):

        start_time = time.time()

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = EditDistance()
        val_accuracy = EditDistance()
        train_step(epoch, train1Data,model, optimizer,epoch_loss_avg, epoch_accuracy, log)

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        log.info("Epoch {:03d}: Loss: {:.3f}, Editdistance: {:.3}".format(epoch,epoch_loss_avg.result(),
                                                                         epoch_accuracy.result()))
        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()
        checkpoint_save(log, checkpoint, manager)
        validation(valid1Data, model, val_accuracy, log, start_time)
        log.info("Time taken: %.2fs\n" % (time.time() - start_time))
        if epoch%20 == 0:
            test(batch_size, model, log,label_pad, max_length, num_classes)

    model.save(SAVE_PATH)

