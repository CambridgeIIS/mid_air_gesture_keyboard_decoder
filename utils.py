import csv
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from collections import defaultdict
import pandas as pd

alphabet = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']

letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z']

alphabet_ord = list(map(ord, alphabet))
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
num_to_alpha = dict(enumerate(alphabet_ord))

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


def get_location_df(candidate_id):
    keys = []
    x_pos = []
    y_pos = []
    key_width = []
    key_height = []
    f = open("kb-layout.txt", "r")
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

def normalize(offsets):
    """
    normalizes strokes to median unit norm
    """
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets


def interpolate_cons(stroke, length=240):
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


def get_location_gan(sequence, candidate_id):
    keys = []
    x_pos = []
    y_pos = []
    key_width = []
    key_height = []
    f = open("kb-layout.txt", "r")
    lines = f.readlines()
    line_num = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34]

    hw, w = candidate_final_list[int(candidate_id) - 1]
    _,new_width,new_height =  normalize_bo(sequence,candidate_id)
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

            x_pos.append(new_x/new_width)
            y_pos.append(new_y/new_height)
            key_width.append(new_key_width/new_width)
            key_height.append(new_key_height/new_height)

    df = pd.DataFrame({
        'keys': keys,
        'x_pos': x_pos,
        'y_pos': y_pos,
        'key width': key_width,
        'key height': key_height
    })
    df = df.set_index('keys')
    location = np.zeros((len(sequence), len(alphabet)))
    for i, point in enumerate(sequence):
        for j, char in enumerate(alphabet):
            loc = [float(df.loc[char][0]), float(df.loc[char][1])]
            if np.linalg.norm(point - loc) < np.sqrt(np.square(new_key_width) + np.square(new_key_width)) / 2:
                location[i][j] = 1

    return location, df



def get_location(sequence, candidate_id):
    keys = []
    x_pos = []
    y_pos = []
    key_width = []
    key_height = []
    f = open("kb-layout.txt", "r")
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
    location = np.zeros((len(sequence), len(alphabet)))
    for i, point in enumerate(sequence):
        for j, char in enumerate(alphabet):
            loc = [float(df.loc[char][0]), float(df.loc[char][1])]
            if np.linalg.norm(point - loc) < np.sqrt(np.square(new_key_width) + np.square(new_key_width)) / 2:
                location[i][j] = 1

    return location, df



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


def get_ascii_sequences(phrases):
    lines = encode_ascii(phrases)
    return lines


def l2_norm(phrases,tsteps_ascii,text_line_data, fake_coords):
    dist1 = np.linalg.norm(text_line_data[:,0]-fake_coords[:,0])
    dist2 = np.linalg.norm(text_line_data[:,1]-fake_coords[:,1])
    dist = (dist1+dist2)/(2*len(phrases)*tsteps_ascii)
    return dist



def get_stroke_sequence(fname, phrase, max_c_length, tsteps_asii):
    coords = []

    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row == []:
                continue

            if row[0] == '':
                continue

            coords.append([float(row[0]),
                           float(row[1]),
                           ])
    coords = np.array(coords)
    coords = np.reshape(coords, [-1, 2])

    coords = denoise(coords)
    coords_ = interpolate(coords, len(phrase), tsteps_asii)
    return coords, coords_



def encode_ascii(ascii_string):
    """
    encodes ascii string to array of ints
    """
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string)))

def self_interpolate(stroke_all, scale):
    """
    interpolates strokes using cubic spline
    """
    stroke_all_new = []
    for stroke in stroke_all:
        xy_coords = stroke[:, :2]

        if len(stroke) > 3:
            f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='linear')
            f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='linear')

            xx = np.linspace(0, len(stroke) - 1, int(len(stroke)/scale))
            yy = np.linspace(0, len(stroke) - 1, int(len(stroke)/scale))

            x_new = f_x(xx)
            y_new = f_y(yy)

            xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

        stroke_all_new.append(xy_coords)


    return stroke_all_new


def denoise_3d(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    x_new = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    y_new = savgol_filter(coords[:, 1], 7, 3, mode='nearest')
    z_new = savgol_filter(coords[:, 2], 7, 3, mode='nearest')
    stroke = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1), z_new.reshape(-1, 1)])

    return stroke

def interpolate_3d(stroke, max_length):
    """
    interpolates strokes using cubic spline
    """
    xyz_coords = stroke

    if len(stroke) > 3:

        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0])
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1])
        f_z = interp1d(np.arange(len(stroke)), stroke[:, 2])

        xx = np.linspace(0, len(stroke) - 1, max_length)
        yy = np.linspace(0, len(stroke) - 1, max_length)
        zz = np.linspace(0, len(stroke) - 1, max_length)

        x_new = f_x(xx)
        y_new = f_y(yy)
        z_new = f_z(zz)

        xyz_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1), z_new.reshape(-1, 1)])

    return xyz_coords



def interpolate(stroke, max_length):
    """
    interpolates strokes using cubic spline
    """

    xy_coords = stroke[:, :2]

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='linear')
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='linear')

        xx = np.linspace(0, len(stroke) - 1, max_length)
        yy = np.linspace(0, len(stroke) - 1, max_length)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords


def denoise(coords):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    x_new = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    y_new = savgol_filter(coords[:, 1], 7, 3, mode='nearest')
    stroke = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return stroke

def add_noise(coords, scale=0.05):
    """
    adds gaussian noise to strokes
    """
    coords = np.copy(coords)
    coords[1:, :] += np.random.normal(loc=0.0, scale=scale, size=coords[1:, :].shape)
    return coords



def normalize(coords):
    """
    normalizes strokes to median unit norm
    """
    coords_ = np.copy(coords)
    coords_[:, :] /= np.median(np.linalg.norm(coords[:, :], axis=1))
    return coords_

def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = coords[1:, :2] - coords[:-1, :2]
    offsets = np.concatenate([np.array([[0, 0]]), offsets], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    convert from offsets to coordinates
    """
    return np.cumsum(offsets[:, :2], axis=0)




def interpolate_linear(stroke, len_ascii, tsteps_ascii):
    xy_coords = stroke

    if len(stroke) > 3:
        f_x = interp1d(np.arange(len(stroke)), stroke[:, 0])
        f_y = interp1d(np.arange(len(stroke)), stroke[:, 1])

        xx = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_ascii)
        yy = np.linspace(0, len(stroke) - 1, len_ascii * tsteps_ascii)

        x_new = f_x(xx)
        y_new = f_y(yy)

        xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

    return xy_coords



