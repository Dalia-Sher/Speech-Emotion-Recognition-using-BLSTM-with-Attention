import numpy as np
import scipy
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import pickle


def normalize_signal(y):
    g_x = max(abs(y))
    return 0.5 * y / g_x


def create_rever_wav(file, mat, sr, duration, path, length_h):
    y, _ = librosa.load(file, sr=sr, mono=True, duration=duration)
    x = normalize_signal(y)

    mat_rir = mat['impulse_response'][:, 0]
    mat_rir = mat_rir[0:length_h]

    y_rever = np.convolve(x, mat_rir, mode='valid')
    x_norm = normalize_signal(y_rever)
    x_rever = librosa.resample(x_norm.astype(np.float64), orig_sr=sr, target_sr=16000)

    print(file.split("/")[8])
    new_path = path + "/" + file.split("/")[8]
    write(new_path, 16000, x_rever)

    # data, sampling_rate = librosa.load(path, sr=sr)
    # librosa.display.waveshow(data, sampling_rate)
    # plt.show()


dataset = 'RAVDESS'
sr = 48000  # Sample rate (16.0 kHz)
duration = 5  # Max pad length (5.0 sec)
length_h = 16000  # Length of impulse response
mat = scipy.io.loadmat(
    '/home/dsi/shermad1/Emotion_Recognition/Data/Reverberation_data/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.610s)_3-3-3-8-3-3-3_2m_090.mat')

if dataset == 'RAVDESS':
    with open('data/RAV_df.pickle', 'rb') as f:
        df = pickle.load(f)
    path = "/home/dsi/shermad1/Emotion_Recognition/Data/Reverberation_data/rever_data_RAV_new/ALL"

if dataset == 'IEMOCAP':
    with open('data/IEMOCAP_df.pickle', 'rb') as f:
        df = pickle.load(f)
    path = "/home/dsi/shermad1/Emotion_Recognition/Data/Reverberation_data/rever_data_IEMOCAP_new"

for file in df.path:
    create_rever_wav(file, mat, sr, duration, path, length_h)
