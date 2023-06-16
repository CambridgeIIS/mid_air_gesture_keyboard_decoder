import tensorflow as tf
from tensorflow import keras

class CTCLoss(keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=28,
                 reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def __call__(self, y_true, y_pred,data_sequence,label_seq):
        y_true = tf.cast(y_true, tf.int32)
        label_length = label_seq
        logit_length = data_sequence

        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.reduce_mean(loss)

def rnn_att_model(input_shape2,
                  output_shape,
                  cnn_features=10,
                  rnn='LSTM',
                  multi_rnn=True,
                  attention=True,
                  dropout=0.2,
                  num_units = 400,
                  num_layers= 2,
                  bidirectional = True):

    # Fetch input
    input_shape = (None, input_shape2)
    inputs = tf.keras.Input(shape=input_shape)

    # LSTM Layer
    if rnn not in ['LSTM', 'GRU']:
        raise ValueError(
            'rnn should be equal to LSTM or GRU. No model generated...')
    if bidirectional:
        if rnn == 'LSTM':
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                num_units, return_sequences=True, dropout=dropout))(inputs)
            if multi_rnn:
                for _ in range(num_layers - 1):
                    layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                        num_units, return_sequences=True, dropout=dropout))(layer_out)

            # GRU Layer
        if rnn == 'GRU':
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                num_units, return_sequences=True, dropout=dropout))(inputs)
            if multi_rnn:
                for _ in range(num_layers - 1):
                    layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                        num_units, return_sequences=True, dropout=dropout))(layer_out)

    else:
        if rnn == 'LSTM':
            layer_out = tf.keras.layers.LSTM(
                num_units, return_sequences=True, dropout=dropout)(inputs)
            if multi_rnn:
                for _ in range(num_layers-1):
                    layer_out =tf.keras.layers.LSTM(
                        num_units, return_sequences=True, dropout=dropout)(layer_out)

        # GRU Layer
        if rnn == 'GRU':
            layer_out =tf.keras.layers.GRU(
                num_units, return_sequences=True, dropout=dropout)(inputs)
            if multi_rnn:
                for _ in range(num_layers-1):
                    layer_out =tf.keras.layers.GRU(
                        num_units, return_sequences=True, dropout=dropout)(layer_out)

    if attention:
        query, value = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=2))(layer_out)
        layer_out, attention_score = tf.keras.layers.Attention(name='Attention')([query, value],return_attention_scores = True)

    outputs = tf.keras.layers.Dense(output_shape)(layer_out)

    # Output Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model