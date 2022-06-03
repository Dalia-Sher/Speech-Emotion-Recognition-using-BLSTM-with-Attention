
# Import relevant libraries
from create_dataFrame import create_IEMOCAP_name_files, create_RAV_name_files
import pickle
from collections import Counter
import numpy as np
import librosa
import math
from keras import backend as K
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


# Function to create signal
def create_signal(df, sr, duration):
    signal_train = []
    wavs_train = []
    signal_test = []
    wavs_test = []

    for index, path in enumerate(df.path):
        if df.split[index] == 'Train':
            y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
            wavs_train.append(y)
            signal_train = pad_sequences(wavs_train, maxlen=sr * duration, dtype="float32")
        else:
            y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
            wavs_test.append(y)
            signal_test = pad_sequences(wavs_test, maxlen=sr * duration, dtype="float32")

    return signal_train, signal_test


nb_aug = 5


# Function to noisy signal
def noisy_signal(signal, snr_low=10, snr_high=15, nb_augmented=nb_aug):
    # Signal length
    signal_len = len(signal)

    # Generate White noise
    noise = np.random.normal(size=(nb_augmented, signal_len))

    # Compute signal and noise power
    s_power = np.sum((signal / (2.0 ** 15)) ** 2) / signal_len
    n_power = np.sum((noise / (2.0 ** 15)) ** 2, axis=1) / signal_len

    # Random SNR: Uniform [15, 30]
    snr = np.random.randint(snr_low, snr_high)

    # Compute K coeff for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- snr / 10))
    K = np.ones((signal_len, nb_augmented)) * K

    # Generate noisy signal
    return signal + K.T * noise


# Function to extract features
def extract_mel_spect(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', fmax=4000):
    # Compute stft
    stft = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute log-mel spectrogram - 128
    mel_spect = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=128, fmax=fmax)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return mel_spect


def extract_get_audio_features(y, sr=16000, frame_length=512, duration=5):
    N_FRAMES = math.ceil(sr * duration / frame_length)
    frames = []
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=frame_length)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=frame_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=frame_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=frame_length)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=frame_length)[0]
    S, phase = librosa.magphase(librosa.stft(y=y, hop_length=frame_length))
    rms = librosa.feature.rms(y=y, hop_length=frame_length, S=S)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=frame_length)
    mfcc_der = librosa.feature.delta(mfcc)
    for i in range(N_FRAMES):
        f=[]
        f.append(spectral_centroid[i])
        f.append(spectral_contrast[i])
        f.append(spectral_bandwidth[i])
        f.append(spectral_rolloff[i])
        f.append(zero_crossing_rate[i])
        f.append(rms[i])
        for m_coeff in mfcc[:,i]:
            f.append(m_coeff)
        for m_coeff_der in mfcc_der[:, i]:
            f.append(m_coeff_der)
        frames.append(f)
    return frames


# Split spectrogram into frames
def frame(x, win_step=64, win_size=128):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:, t, :, :] = np.copy(x[:, :, (t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames


# Function to extract train and test data
def extract_train_test(args, signal_train, signal_test, labels_train, labels_test):

    print("Extracting Features: START")
    if args.dataset == 'IEMOCAP':
        mel_spect_train = np.asarray(list(map(extract_get_audio_features, signal_train)))  #IEMOCAP
        mel_spect_test = np.asarray(list(map(extract_get_audio_features, signal_test)))  # IEMOCAP

    elif args.dataset == 'RAVDESS':
        mel_spect_train = np.asarray(list(map(extract_mel_spect, signal_train)))         #RAVDESS
        mel_spect_test = np.asarray(list(map(extract_mel_spect, signal_test)))          #RAVDESS

    print('mel_spect_train.shape', mel_spect_train.shape)
    print('mel_spect_test.shape', mel_spect_test.shape)
    print("Extracting Features: END!")

    X_train = mel_spect_train
    y_train = labels_train

    # Build test set
    X_test = mel_spect_test
    y_test = labels_test

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    print('X_train.shape before mean:', X_train.shape)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print('X_train.shape bafore framing:', X_train.shape)

    if args.dataset == 'RAVDESS':
    # Frame for TimeDistributed model
        hop_ts = 64
        win_ts = 128
        X_train = frame(X_train, hop_ts, win_ts)     #RAVDESS
        X_test = frame(X_test, hop_ts, win_ts)       #RAVDESS

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(np.ravel(y_train)))
    y_test = np_utils.to_categorical(lb.transform(np.ravel(y_test)))

    if args.dataset == 'RAVDESS':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)  #RAVDESS
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)        #RAVDESS

    elif args.dataset == 'IEMOCAP':
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], 1)           #IEMOCAP
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], 1)                #IEMOCAP

    K.clear_session()

    return X_train, X_test, y_train, y_test


def create_data(dic, args):
    # Creating dataframe with paths
    SPLIT = 'False'
    ts = 0.2

    if args.dataset == 'IEMOCAP':
        df, labels_train, labels_test = create_IEMOCAP_name_files(SPLIT, ts, dic)  #IEMOCAP

    elif args.dataset == 'RAVDESS':
        df, labels_train, labels_test = create_RAV_name_files(SPLIT, ts)     #RAVDESS

    print('dictionary:', dic)
    print('num of train labels:', Counter(labels_train))
    print('num of test labels:', Counter(labels_test))

    # Creating a signal
    sr = 16000  # Sample rate (16.0 kHz)
    duration = 5  # Max pad length (5.0 sec)
    print("Creating Signal: START")
    signal_train, signal_test = create_signal(df, sr, duration)
    print("Creating Signal: END!")
    print('signal_train.shape:', signal_train.shape)
    print('signal_test.shape:', signal_test.shape)

    # Extracting train and test
    X_train, X_test, y_train, y_test = extract_train_test(args, signal_train, signal_test, labels_train, labels_test)

    def save_train_test(args):
        if args.dataset == 'IEMOCAP':
            with open('data/X_train_1_IEMOCAP.pickle', 'wb') as f:
                pickle.dump(X_train[:int(X_train.shape[0]/2), :, :, :, :], f)
            with open('data/X_train_2_IEMOCAP.pickle', 'wb') as f:
                pickle.dump(X_train[int(X_train.shape[0]/2):, :, :, :, :], f)
            with open('data/y_train_IEMOCAP.pickle', 'wb') as f:
                pickle.dump(y_train, f)
            with open('data/X_test_IEMOCAP.pickle', 'wb') as f:
                pickle.dump(X_test, f)
            with open('data/y_test_IEMOCAP.pickle', 'wb') as f:
                pickle.dump(y_test, f)

        elif args.dataset == 'RAVDESS':
            with open('data/X_train_1_RAVDESS.pickle', 'wb') as f:
                pickle.dump(X_train[:int(X_train.shape[0]/2),:, :, :, :], f)
            with open('data/X_train_2_RAVDESS.pickle', 'wb') as f:
                pickle.dump(X_train[int(X_train.shape[0]/2):,:, :, :, :], f)
            with open('data/y_train_RAVDESS.pickle', 'wb') as f:
                pickle.dump(y_train, f)
            with open('data/X_test_RAVDESS.pickle', 'wb') as f:
                pickle.dump(X_test, f)
            with open('data/y_test_RAVDESS.pickle', 'wb') as f:
                pickle.dump(y_test, f)

    if args.create_data == 'YES':
        save_train_test(args)

    return X_train, y_train, X_test, y_test




