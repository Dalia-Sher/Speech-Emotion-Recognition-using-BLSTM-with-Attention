import shutil
import glob
import pickle
import librosa
from scipy.io.wavfile import write
import numpy as np
import pandas as pd


# Move IEMOCAP files all to one directory
new_path = "/home/dsi/shermad1/Emotion_Recognition/Data/IEMOCAP_flat/ALL/"

for file in glob.glob("/home/dsi/shermad1/Emotion_Recognition/Data/IEMOCAP/IEMOCAP_full_release/*/sentences/wav/*/*.wav"):
    print(file)
    shutil.move(file, new_path)

# Convert RAVDESS from 48k to 16k
with open('data/RAV_df.pickle', 'rb') as f:
    df = pickle.load(f)
path = "/home/dsi/shermad1/Emotion_Recognition/Data/Reverberation_data/rever_data_RAV_16k"

for file in df.path:
    y, _ = librosa.load(file, sr=48000, mono=True, duration=5)
    x_rever = librosa.resample(y.astype(np.float64), orig_sr=48000, target_sr=16000)

    print(file.split("/")[8])
    new_path = path + "/" + file.split("/")[8]
    write(new_path, 16000, x_rever)


# Create 16k RAVDESS dataframe splitted to train and test as original
with open("data/RAV_df.pickle", "rb") as f:
    object = pickle.load(f)

df = pd.DataFrame(object)
length = len(df.path)
separator = '/'

for i in range(length):
    old_list = df.path[i].split('/')
    new_path = separator.join(old_list[:6])+'/RAV_16k/rever_data_RAV_16k/'+old_list[8]
    df.at[i, 'path'] = new_path


df.to_csv('data_RAV_16k/RAV_df_16k_good_split.csv')
with open('data_RAV_16k/RAV_df_16k_good_split.pickle', 'wb') as f:
    pickle.dump(df, f)
