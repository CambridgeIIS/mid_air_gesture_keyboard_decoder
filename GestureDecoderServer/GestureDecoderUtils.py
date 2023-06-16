
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow import keras


candidate_list1 = np.linspace(0.375, 1, 20)
candidate_list2 = np.linspace(0.167, 1, 20)
candidate_list = np.concatenate((candidate_list1, candidate_list2), axis=0)
candidate_list = np.reshape(candidate_list, (2, 20))
candidate_list[0, :] = candidate_list[0, :] * 2
candidate_list[1, :] = candidate_list[1, :] * 3

candidate_final_list = []
for i, hw in enumerate(candidate_list[0, :]):
    for j, w in enumerate(candidate_list[1, :]):
        candidate_final_list.append([hw, w])

alphabet = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']

letter_table = ['','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','','']
def to_dense(tensor):
    tensor = tf.sparse.to_dense(tensor, default_value=0)
    tensor = tf.cast(tensor, tf.int32).numpy()
    return tensor



def get_location_df(candidate_id):
    keys = []
    x_pos = []
    y_pos = []
    key_width = []
    key_height = []
    f = open("./kb-layout.txt", "r")
    lines = f.readlines()
    line_num = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34]

    hw, w = candidate_final_list[int(candidate_id) - 1]

    original_key_width = 50
    original_key_height = 50
    for a, line in enumerate(lines):
        if a in line_num:
            new_line = line
            new_line_split = new_line.split('\t')

            original_x = new_line_split[3]
            original_y = new_line_split[4]

            h = hw * w

            new_x = int(original_x) * w * 10
            new_y = int(original_y) * h * 10

            new_key_width = original_key_width * w
            new_key_height = original_key_height * h

            keys.append(new_line_split[0])
            x_pos.append(new_x)
            y_pos.append(new_y)
            key_width.append(new_key_width)
            key_height.append(new_key_height)

    df = pd.DataFrame({
        'keys': keys,
        'x_pos': x_pos,
        'y_pos': y_pos,
        'key width': key_width,
        'key height': key_height
    })
    df = df.set_index('keys')

    # return df,new_key_width,new_key_height

    zipped_lists = zip(keys, x_pos)

    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list_x = [element for _, element in sorted_zipped_lists]

    zipped_lists = zip(keys, y_pos)

    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list_y = [element for _, element in sorted_zipped_lists]

    key_locations = np.transpose([sorted_list_x,sorted_list_y])
    tolerence = np.sqrt(np.square(new_key_width) + np.square(new_key_width)) / 2


    return key_locations, tolerence,df,new_key_width,new_key_height


def get_location(sequence,df,new_key_width,new_key_height):

    location = np.zeros((len(sequence), len(alphabet)))
    for i, point in enumerate(sequence):
        for j, char in enumerate(alphabet):
            loc = [float(df.loc[char][0]), float(df.loc[char][1])]
            if np.linalg.norm(point - loc) < np.sqrt(np.square(new_key_width) + np.square(new_key_width)) / 2:
                location[i][j] = 1

    return location


def get_location_fst(sequence, key_locations, tolerence):
    def select_negative(number):
        if number != 0:
            return 1
        else:
            return 0

    key_locations_ = [key_locations] * len(sequence)
    print(np.shape(key_locations_))

    coord_0 = [sequence] * len(key_locations)
    coord_ = np.transpose(coord_0, (1, 0, 2))
    print(np.shape(coord_))

    distance = np.linalg.norm(coord_ - key_locations_, axis=2)
    distance_ = distance - tolerence

    mapped_location_ = [list(map(lambda x: min(x, 0), x)) for x in distance_]
    mapped_location = [list(map(lambda x: select_negative(x), x)) for x in mapped_location_]

    return mapped_location

def normalize_bo(coords, candidate_id):
    original_width = 620
    original_height = 170
    hw, w = candidate_final_list[int(candidate_id) - 1]

    h = hw * w
    new_width = int(original_width) * w
    new_height = int(original_height) * h
    coords[:, 0] = coords[:, 0] / new_width
    coords[:, 1] = coords[:, 1] / new_height
    return coords

def denoise(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    x_new = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    y_new = savgol_filter(coords[:, 1], 7, 3, mode='nearest')
    stroke = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return stroke

def interpolate(stroke, length):
    """
    interpolates strokes using cubic spline
    """
    xy_coords = stroke[:, :2]

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='linear')
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='linear')

        xx = np.linspace(0, len(stroke) - 1, length)
        yy = np.linspace(0, len(stroke) - 1, length)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords




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


class EditDistance(keras.metrics.Metric):
    def __init__(self, name='edit_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sum_distance = self.add_weight(name='sum_distance',
                                            initializer='zeros')

    def update_state(self, y_true, y_pred, seq_len_list, label_data_length, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        # logit_length = tf.fill([batch_size], y_pred_shape[1])
        logit_length = seq_len_list
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        sum_distance = tf.math.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int64), tf.cast(tf.sparse.from_dense(
            y_true), tf.int64)))
        batch_size = tf.cast(batch_size, tf.float32)
        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return self.sum_distance / self.total

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)



def add_features(sequence):
    sequence = np.asarray(sequence)
    next_seq = np.append(sequence[1:, :], [sequence[-1, :]], axis=0)
    prev_seq = np.append([sequence[0, :]], sequence[:-1, :], axis=0)

    # compute gradient
    gradient = np.subtract(sequence, prev_seq)

    #compute curvature
    vec_1 = np.multiply(gradient, -1)
    vec_2 = np.subtract(next_seq, sequence)
    cos = np.divide(np.sum(vec_1*vec_2, axis=1),
                      np.linalg.norm(vec_1, 2, axis=1)*np.linalg.norm(vec_2, 2, axis=1))

    angle = np.arccos(cos)
    curvature = np.column_stack((np.cos(angle), np.sin(angle)))


    #compute vicinity (5-points) - curliness/linearity
    padded_seq = np.concatenate(([sequence[0]], [sequence[0]], sequence, [sequence[-1]], [sequence[-1]]), axis=0)
    aspect = np.zeros(len(sequence))
    slope = np.zeros((len(sequence), 2))
    curliness = np.zeros(len(sequence))
    linearity = np.zeros(len(sequence))
    for j in range(2, len(sequence)+2):
        vicinity = np.asarray([padded_seq[j-2], padded_seq[j-1], padded_seq[j], padded_seq[j+1], padded_seq[j+2]])
        delta_x = max(vicinity[:, 0]) - min(vicinity[:, 0])
        delta_y = max(vicinity[:, 1]) - min(vicinity[:, 1])

        # delta_x = vicinity[-1, 0] - vicinity[0, 0]
        # delta_y = vicinity[-1, 1] - vicinity[0, 1]
        slope_vec = vicinity[-1] - vicinity[0]

        #aspect of trajectory
        aspect[j-2] = (delta_y - delta_x) / (delta_y + delta_x)

        #cos and sin of slope_angle of straight line from vicinity[0] to vicinity[-1]
        slope_angle = np.arctan(np.abs(np.divide(slope_vec[1], slope_vec[0]))) * np.sign(np.divide(slope_vec[1], slope_vec[0]))
        slope[j-2] = [np.cos(slope_angle), np.sin(slope_angle)]

        #length of trajectory divided by max(delta_x, delta_y)
        curliness[j-2] = np.sum([np.linalg.norm(vicinity[k+1] - vicinity[k], 2) for k in range(len(vicinity)-1)]) / max(delta_x, delta_y)

        #avg squared distance from each point to straight line from vicinity[0] to vicinity[-1]
        linearity[j-2] = np.mean([np.power(np.divide(np.cross(slope_vec, vicinity[0] - point), np.linalg.norm(slope_vec, 1)), 2) for point in vicinity])

    vicinity_features = np.column_stack((aspect, slope, curliness, linearity))

    # add features to signal
    offsets = coords_to_offsets(sequence)

    result = np.nan_to_num(np.concatenate((offsets, gradient, curvature, vicinity_features), axis=1)).tolist()

    return result



def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = coords[1:, :2] - coords[:-1, :2]
    offsets = np.concatenate([np.array([[0, 0]]), offsets], axis=0)
    # offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    # offsets = np.concatenate([np.array([[0, 0, 0]]), offsets], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    convert from offsets to coordinates
    """
    return np.cumsum(offsets[:, :2], axis=0)

def softmax(mat):
	"calc softmax such that labels per time-step form probability distribution"
	maxT, _ = mat.shape # dim0=t, dim1=c
	res = np.zeros(mat.shape)
	for t in range(maxT):
		y = mat[t, :]
		e = np.exp(y)
		s = np.sum(e)
		res[t, :] = e/s
	return res
