import pandas as pd
import numpy as np
import collections
from utils import *
from multiprocessing import Process


def f():
    for type_ in [2]:
        fname = 'data/data_all.pkl'
        output = pd.read_pickle(fname)
        coords_list = output['lift on coords'].to_numpy()
        phrase_list = output['phrase'].to_numpy()
        word_suggestions_list = output['word suggestions'].to_numpy()
        candidate_id_list = output['candidate id'].to_numpy()

        # check for number of words and number of suggestions
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

        word_list = []
        candidate_id_word_list = []
        for i, phrase in enumerate(phrase_list):
            for word in enumerate(phrase):
                word_list.append(word)
                candidate_id_word_list.append(candidate_id_list[i])

        word_suggestion_flat = []
        for i, word_suggestions in enumerate(word_suggestions_list):
            for word_suggestion in enumerate(word_suggestions):
                word_suggestion_flat.append(word_suggestion[1])

        word_coords_list = []
        for coords in coords_list:
            for word_coords in coords:
                word_coords_list.append(word_coords)

        coords_all = []
        label_all = []
        word_all = []
        char_len = []
        stroke_len = []
        for i, coords in enumerate(word_coords_list):
            if word_list[i][1] in word_suggestion_flat[i]:
                print('index: {} out of {}'.format(i,len(word_coords_list)))
                if len(coords)<4:
                    print('empty or less than 3')
                    continue
                coords = np.array(coords)[:,:2]
                candidate_id = candidate_id_word_list[i]
                word = word_list[i][1]
                coords = denoise(coords)
                coords = interpolate(coords, 50)
                stroke_len.append(len(coords))

                if type_==0:
                    coords = normalize_bo(coords, candidate_id)
                elif type_ ==1:
                    key_locations, tolerence, df, new_key_width, new_key_height = get_location_df(candidate_id)
                    location = get_location_fst(coords[:,:2], key_locations, tolerence )
                    coords = normalize_bo(coords, candidate_id)
                    coords = np.concatenate((coords, location), axis=1)
                elif type_ ==2:
                    key_locations, tolerence, df, new_key_width, new_key_height = get_location_df(candidate_id)
                    location = get_location_fst(coords[:,:2], key_locations, tolerence )
                    coords = normalize_bo(coords, candidate_id)
                    features = add_features(coords[:,:2])
                    coords = np.concatenate((coords,location,features),axis = 1)

                label = get_ascii_sequences(word)
                char_len.append(len(label))
                word_all.append(word)
                coords_all.append(coords)
                label_all.append(label)


        d = collections.OrderedDict()
        for i, v in enumerate(word_all):
            d[v] = i

        index_list = np.array(list(d.values()))

        print('stroke : max {}, min {}, mean {}'.format(np.max(stroke_len), np.min(stroke_len), np.mean(stroke_len)))
        print('char : max {}, min {}, mean {}'.format(np.max(char_len),np.min(char_len),np.mean(char_len)))
        print('input data with shape : {}'.format(np.shape(coords_all)))
        print('label data with shape : {}'.format(np.shape(label_all)))
        print('number of unique words: {}'.format(len(index_list)))
        print('number of trajectories: {}'.format(len(coords_all)))
        print('shape of one sequence : {}'.format(np.shape(coords_all[0])))
        coords_all = np.array(coords_all, dtype='object')
        label_all = np.array(label_all, dtype='object')

        shuffled_indexes = np.random.permutation(coords_all.shape[0])
        coords_all = coords_all[shuffled_indexes]
        label_all = label_all[shuffled_indexes]


        train_data, valid_data = np.split(
            coords_all, [np.shape(coords_all)[0] * 19 // 20])
        train_label, valid_label = np.split(
            label_all, [np.shape(label_all)[0] * 19 // 20])

        np.save("data/data_mlgk_train_50_{}".format(type_), train_data)
        np.save("data/label_mlgk_train_50_{}".format(type_), train_label)

        np.save("data/data_mlgk_valid_50_{}".format(type_), valid_data)
        np.save("data/label_mlgk_valid_50_{}".format(type_), valid_label)

        print("Successfully saved!")


if __name__ == '__main__':
    p = Process(target=f)
    p.start()
    p.join()