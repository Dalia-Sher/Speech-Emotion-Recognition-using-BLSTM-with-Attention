
from model import create_model
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
ax = plt.axes(projection='3d')
from pylab import *
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def extract_mel_spect_TSNE(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', fmax=4000):
    # Compute stft
    stft = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute log-mel spectrogram - 128
    mel_spect = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=128, fmax=fmax)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return np.mean(mel_spect,1)


def extract_get_audio_features_TSNE(y, sr=16000, frame_length=512):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=frame_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=frame_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=frame_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=frame_length)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=frame_length)
    S, phase = librosa.magphase(librosa.stft(y=y, hop_length=frame_length))
    rms = librosa.feature.rms(y=y, hop_length=frame_length, S=S)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=frame_length)
    mfcc_der = librosa.feature.delta(mfcc)

    feature_vector = np.concatenate((np.mean(spectral_centroid,1), np.mean(spectral_contrast,1), np.mean(rms,1),
                                     np.mean(spectral_bandwidth,1), np.mean(spectral_rolloff,1), np.mean(mfcc,1),
                                     np.mean(zero_crossing_rate,1), np.mean(mfcc_der,1)))

    return feature_vector


# TSNE - for features
def TSNE_features(args):

    if args.dataset == 'IEMOCAP':
        print('you chose args.dataset == "IEMOCAP"')
        with open('data/IEMOCAP_df.pickle', 'rb') as f:
            df = pickle.load(f)
        colors = ['blue', 'red', 'green', 'orange']
        palette = sns.color_palette(colors)    #IEMOCAP

    elif args.dataset == 'RAVDESS':
        print('you chose args.dataset == "RAVDESS"')
        with open('data/RAV_df.pickle', 'rb') as f:
            df = pickle.load(f)
        palette = sns.color_palette("bright", 7)  # RAV

    print(df.head())

    if args.create_TSNE_data == 'YES':
        feature_vectors = []
        sound_paths = []
        for i, f in enumerate(df.path):
            if i % 100 == 0:
                print("get %d of %d = %s" % (i + 1, len(df.path), f))
            y, sr = librosa.load(f)

            if args.dataset == 'IEMOCAP':
                feat = extract_get_audio_features_TSNE(y)
            elif args.dataset == 'RAVDESS':
                feat = extract_mel_spect_TSNE(y)

            feature_vectors.append(feat)
            sound_paths.append(f)

        if args.dataset == 'IEMOCAP':
            with open('data/feature_vectors_IEMOCAP.pickle', 'wb') as f:
                pickle.dump(feature_vectors, f)

        elif args.dataset == 'RAVDESS':
            with open('data/feature_vectors_RAV.pickle', 'wb') as f:
                pickle.dump(feature_vectors, f)

    elif args.create_TSNE_data == 'NO':
        if args.dataset == 'IEMOCAP':
            with open('data/feature_vectors_IEMOCAP.pickle', 'rb') as f:
                feature_vectors = pickle.load(f)
        elif args.dataset == 'RAVDESS':
            with open('data/feature_vectors_RAV.pickle', 'rb') as f:
                feature_vectors = pickle.load(f)

    print("calculated %d feature vectors" % len(feature_vectors))

    # 2D
    model = TSNE(n_components=2, learning_rate=150, perplexity=30, verbose=2, angle=0.1, random_state=1).fit_transform(feature_vectors)
    x_axis = model[:, 0]
    y_axis = model[:, 1]
    plt.figure()
    sns.scatterplot(x_axis, y_axis, hue=df.emotion, legend='full', palette=palette)
    plt.legend(fontsize='x-large', title_fontsize='40')
    plt.savefig(f'/home/dsi/shermad1/PycharmProjects/SER_git/results/%s_features.png' % args.dataset)
    plt.show()


# TSNE - for network
def TSNE_model(args):
    # Opening the data files
    if args.dataset == "IEMOCAP":
        print('you chose args.dataset == "IEMOCAP"')
        with open('data/X_train_1_IEMOCAP.pickle', 'rb') as f:
            X_train_1 = pickle.load(f)
        with open('data/X_train_2_IEMOCAP.pickle', 'rb') as f:
            X_train_2 = pickle.load(f)
        with open('data/y_train_IEMOCAP.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('data/X_test_IEMOCAP.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test_IEMOCAP.pickle', 'rb') as f:
            y_test = pickle.load(f)
        X_train = np.concatenate((X_train_1, X_train_2))

    elif args.dataset == "RAVDESS":
        print('you chose args.dataset == "RAVDESS"')
        with open('data/X_train_1_RAVDESS_new.pickle', 'rb') as f:
            X_train_1 = pickle.load(f)
        with open('data/X_train_2_RAVDESS_new.pickle', 'rb') as f:
            X_train_2 = pickle.load(f)
        with open('data/y_train_RAVDESS_new.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('data/X_test_RAVDESS_new.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test_RAVDESS_new.pickle', 'rb') as f:
            y_test = pickle.load(f)
        X_train = np.concatenate((X_train_1, X_train_2))

    # load Model
    model_for_tsne = create_model(X_train, y_train, args.model_type, last_layer=False)
    print(model_for_tsne.summary())

    # Loads the weights
    if args.dataset == "IEMOCAP":
        model_for_tsne.load_weights('models/best_%s_model_%s.h5' % (args.model_type, args.dataset), by_name=True)  # IEMOCAP
        colors = ['blue', 'red', 'green', 'orange']  # IEMOCAP
        palette = sns.color_palette(colors)  # IEMOCAP
        d = {0: 'neutral', 1: 'angry', 2: 'happy', 3: 'sad'}  # IEMOCAP

    elif args.dataset == "RAVDESS":
        model_for_tsne.load_weights('models/best_%s_model_%s.h5' % (args.model_type, args.dataset), by_name=True) #RAVSESS
        palette = sns.color_palette("bright", 7)    #RAV
        d = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 4: 'fear', 5: 'disgust', 6: 'surprise'} #RAVDESS


    ax = plt.axes(projection='3d')
    sns.set(rc={'figure.figsize': (11.7, 8.27)})

    # TSNE to all data
    X_train = np.concatenate((X_train, X_test))
    z = model_for_tsne.predict(X_train)
    y_train = np.concatenate((y_train, y_test))
    y_test_arr = np.array([np.where(r == 1)[0][0] for r in y_train])
    emotions_name = (pd.Series(y_test_arr)).map(d)
    emotions_name = list(emotions_name)
    print(emotions_name)
    z_list = z.tolist()

    model = TSNE(n_components=2, learning_rate=150, perplexity=30, verbose=2, angle=0.1, random_state=2).fit_transform(
        z_list)
    x_axis = model[:, 0]
    y_axis = model[:, 1]
    plt.figure()
    sns.scatterplot(x_axis, y_axis, hue=emotions_name, legend='full', palette=palette)
    plt.legend(fontsize='x-large', title_fontsize='40')
    plt.savefig(f'/home/dsi/shermad1/PycharmProjects/SER_git/results/%s_network.png' % args.dataset)
    plt.show()