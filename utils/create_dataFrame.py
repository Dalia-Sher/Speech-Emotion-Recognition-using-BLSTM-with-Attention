
# Import relevant libraries
import os
import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# Function to create dataframe with paths to the files
def create_IEMOCAP_name_files(args, SPLIT, ts, d):
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    path, emotions, names_train, names_test = [], [], [], []

    for sess in range(1, 6):
        emo_evaluation_dir = '/home/dsi/shermad1/Emotion_Recognition/Data/IEMOCAP/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(
            sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                _, wav_file_name, emotion, _ = line.strip().split('\t')

                if args.reverberation == 'YES':
                    print('you chose args.reverberation == "YES"')
                    # path_new = '/home/dsi/shermad1/Emotion_Recognition/Data/Reverberation_data/rever_data_IEMOCAP/'
                    path_new = '/dsi/gannot-lab/datasets/Ohad/IEMOCAP_rev_16k/ALL/ALL/'

                elif args.reverberation == 'NO':
                    path_new = "/home/dsi/shermad1/Emotion_Recognition/Data/IEMOCAP_flat/ALL/"

                if emotion == 'neu':
                    emotions.append(0)
                    path.append(
                        path_new + wav_file_name + ".wav")
                if emotion == 'ang':
                    emotions.append(1)
                    path.append(
                        path_new + wav_file_name + ".wav")
                elif emotion == 'hap' or emotion == 'exc':
                    emotions.append(2)
                    path.append(
                        path_new + wav_file_name + ".wav")
                elif emotion == 'sad':
                    emotions.append(3)
                    path.append(
                        path_new + wav_file_name + ".wav")

    labels = np.asarray(emotions).ravel()

    paths_train, paths_test, emotions_train, emotions_test = train_test_split(path, labels, test_size=ts,
                                                                              random_state=6)
    for i in range(len(paths_test)):
        full_name = paths_test[i].split('/')[8]
        names_test.append(full_name)

    for i in range(len(paths_train)):
        full_name = paths_train[i].split('/')[8]
        names_train.append(full_name)


    d = {0: 'neutral', 1: 'angry', 2: 'happy', 3: 'sad'}
    emotions_train_name = (pd.Series(emotions_train)).map(d)
    emotions_train_name = list(emotions_train_name)

    emotions_test_name = (pd.Series(emotions_test)).map(d)
    emotions_test_name = list(emotions_test_name)

    temp_df = pd.DataFrame()

    train_df = pd.concat([temp_df,
                          pd.DataFrame(paths_train, columns=['path']),
                          pd.DataFrame(['Train'] * len(paths_train), columns=['split']),
                          pd.DataFrame(emotions_train_name, columns=['emotion']),
                          pd.DataFrame(names_train, columns=['name'])], axis=1)

    test_df = pd.concat([temp_df, pd.DataFrame(paths_test, columns=['path']),
                         pd.DataFrame(['Test'] * len(paths_test), columns=['split']),
                         pd.DataFrame(emotions_test_name, columns=['emotion']),
                         pd.DataFrame(names_test, columns=['name'])], axis=1)

    df_iemocap = pd.concat([train_df, test_df], axis=0)
    df_iemocap = df_iemocap.reset_index(drop=True)

    labels_train = np.asarray(emotions_train).ravel()
    labels_test = np.asarray(emotions_test).ravel()

    return df_iemocap, labels_train, labels_test


def create_RAV_name_files(args, SPLIT, ts):
    if args.reverberation == 'YES':
        print('you chose args.reverberation == "YES"')
        path = "/home/dsi/shermad1/Emotion_Recognition/Data/Reverberation_data/rever_data_RAV_new/"
    elif args.reverberation == 'NO':
        path = '/home/dsi/shermad1/Emotion_Recognition/Data/SER/'            #RAVDESS
        # path = '/home/dsi/shermad1/Emotion_Recognition/Data/RAV_16k/'         #RAVDESS 16k
    dir_list = os.listdir(path)
    dir_list.sort()

    emotions = []
    paths = []

    if SPLIT == 'True':
        print('SPLIT')
        split = []
        for i in dir_list:
            fname = os.listdir(path + i)
            for f in fname:
                part = f.split('.')[0].split('-')
                if int(part[2]) == 2:
                    emotions.append(1)
                else:
                    emotions.append(int(part[2]))

                act = part[-1]
                if int(act[0]) == 0 and int(act[1]) < 5:
                    split.append('Test')
                else:
                    split.append('Train')

                paths.append(path + i + '/' + f)

        RAV_df = pd.DataFrame()
        RAV_df = pd.concat([RAV_df, pd.DataFrame(paths, columns=['path']), pd.DataFrame(split, columns=['split'])], axis=1)

        labels = np.asarray(emotions).ravel()
        labels_train = labels[240:]
        labels_test = labels[:240]

    else:
        print('ALL acts')
        for i in dir_list:
            fname = os.listdir(path + i)
            for f in fname:
                part = f.split('.')[0].split('-')
                if int(part[2]) == 2:
                    emotions.append(1)
                else:
                    emotions.append(int(part[2]))
                paths.append(path + i + '/' + f)

        paths_train, paths_test, emotions_train, emotions_test = train_test_split(paths, emotions, test_size=ts,
                                                                                  random_state=6)

        d = {1: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
        emotions_train_name = (pd.Series(emotions_train)).map(d)
        emotions_train_name = list(emotions_train_name)

        emotions_test_name = (pd.Series(emotions_test)).map(d)
        emotions_test_name = list(emotions_test_name)

        temp_df = pd.DataFrame()

        train_df = pd.concat([temp_df,
                              pd.DataFrame(paths_train, columns=['path']),
                              pd.DataFrame(['Train'] * len(paths_train), columns=['split']),
                              pd.DataFrame(emotions_train_name, columns=['emotion'])], axis=1)

        test_df = pd.concat([temp_df, pd.DataFrame(paths_test, columns=['path']),
                             pd.DataFrame(['Test'] * len(paths_test), columns=['split']),
                             pd.DataFrame(emotions_test_name, columns=['emotion'])], axis=1)

        RAV_df = pd.concat([train_df, test_df], axis=0)
        RAV_df = RAV_df.reset_index(drop=True)

        labels_train = np.asarray(emotions_train).ravel()
        labels_test = np.asarray(emotions_test).ravel()

    return RAV_df, labels_train, labels_test

