from numpy.compat import os_PathLike

from preprocessing import create_data
from model import create_model
from TSNE import TSNE_features, TSNE_model
from plotting import plotting_function, compare_results
from reverberation import create_rever_wav
from pylab import *

from argparse import ArgumentParser

# experiment reproducibility
# Seed value
seed_value = 1

# 1. Set 'PYTHONHASHSEED' environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 2. Set 'python' built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set 'numpy' pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the 'tensorflow' pseudo-random generator at a fixed value
import tensorflow as tf
# tf.random.set_seed(seed_value)   # module 'tensorflow._api.v1.random' has no attribute 'set_seed'
tf.compat.v1.random.set_random_seed(seed_value)
# tf.random.set_random_seed(seed_value) # The name tf.random.set_random_seed is deprecated. Please use tf.compat.v1.random.set_random_seed instead

# 5. Configure a new global 'tensorflow' session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#test
print("test")
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Import relevant libraries
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans


def main(args):

    # Dictionary of emotion coding
    if args.dataset == "IEMOCAP":
        print('you chose args.dataset == "IEMOCAP"')
        dic = {0: 'neutral', 1: 'angry', 2: 'happy', 3: 'sad'}  #IEMOCAP
    elif args.dataset == "RAVDESS":
        print('you chose args.dataset == "RAVDESS"')
        dic = {1: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}  # RAVDESS

    # Creating the data pickle files
    if args.create_data == 'YES':
        print('you chose args.create_data == "YES"')
        X_train, y_train, X_test, y_test = create_data(dic, args)

    # Opening the data files
    elif args.create_data == 'NO' and args.dataset == "IEMOCAP" and args.reverberation == 'NO':
        print('you chose args.create_data == "NO"')
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

    elif args.create_data == 'NO' and args.dataset == "IEMOCAP" and args.reverberation == 'YES':
        print('you chose args.create_data == "NO"')
        with open('data_rev_ohad/X_train_1_IEMOCAP.pickle', 'rb') as f:
            X_train_1 = pickle.load(f)
        with open('data_rev_ohad/X_train_2_IEMOCAP.pickle', 'rb') as f:
            X_train_2 = pickle.load(f)
        with open('data_rev_ohad/y_train_IEMOCAP.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('data_rev_ohad/X_test_IEMOCAP.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data_rev_ohad/y_test_IEMOCAP.pickle', 'rb') as f:
            y_test = pickle.load(f)
        X_train = np.concatenate((X_train_1, X_train_2))

    elif args.create_data == 'NO' and args.dataset == "RAVDESS" and args.reverberation == 'NO':
        print('you chose args.create_data == "NO"')
        with open('data_RAV_16k/X_train_1_RAVDESS.pickle', 'rb') as f:
            X_train_1 = pickle.load(f)
        with open('data_RAV_16k/X_train_2_RAVDESS.pickle', 'rb') as f:
            X_train_2 = pickle.load(f)
        with open('data_RAV_16k/y_train_RAVDESS.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('data_RAV_16k/X_test_RAVDESS.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data_RAV_16k/y_test_RAVDESS.pickle', 'rb') as f:
            y_test = pickle.load(f)
        X_train = np.concatenate((X_train_1, X_train_2))

    elif args.create_data == 'NO' and args.dataset == "RAVDESS" and args.reverberation == 'YES':
        print('you chose args.create_data == "NO"')
        with open('data_RAV_rev/X_train_1_RAVDESS.pickle', 'rb') as f:
            X_train_1 = pickle.load(f)
        with open('data_RAV_rev/X_train_2_RAVDESS.pickle', 'rb') as f:
            X_train_2 = pickle.load(f)
        with open('data_RAV_rev/y_train_RAVDESS.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('data_RAV_rev/X_test_RAVDESS.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data_RAV_rev/y_test_RAVDESS.pickle', 'rb') as f:
            y_test = pickle.load(f)
        X_train = np.concatenate((X_train_1, X_train_2))

    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    print('X_test.shape:', X_test.shape)
    print('y_test.shape:', y_test.shape)

    # Creating model
    model = create_model(X_train, y_train, args.model_type)
    print(model.summary())

    if args.model_load == 'NO':
        # Start training
        print("START training:")

        # Early stopping
        # val_accuracy
        if args.reverberation == 'NO':
            callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=30),
                ModelCheckpoint(filepath=f'models/best_%s_model_%s_16k_good_s_5conv.h5' % (args.model_type, args.dataset), monitor='val_loss', save_best_only=True)]

        elif args.reverberation == 'YES':
            callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=30),
                ModelCheckpoint(filepath=f'models/best_%s_model_%s_rev.h5' % (args.model_type, args.dataset), monitor='val_loss', save_best_only=True)]
        
        # Fit the model
        history = model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_test, y_test),
                            callbacks=[callbacks])

        print("Training is DONE!")
        
        if args.reverberation == 'NO':
            model.save_weights(f'models/last_%s_model_%s_16k_good_s_5conv.h5' % (args.model_type, args.dataset))
            
        elif args.reverberation == 'YES':
            model.save_weights(f'models/last_%s_model_%s_rev.h5' % (args.model_type, args.dataset))

        # Make predictions
        preds = model.predict(X_test)
        preds = preds.argmax(axis=1)

    elif args.model_load == 'YES':
        # load Model
        model_best = create_model(X_train, y_train, args.model_type, last_layer=True)

        # Loads the weights
        model_best.load_weights(f'models/best_%s_model_%s_16k_good_s.h5' % (args.model_type, args.dataset))
        #model_best.load_weights('/home/dsi/shermad1/PycharmProjects/SER/models/best_Attention_model_RAV.h5')

        # Make predictions
        preds = model_best.predict(X_test)
        preds = preds.argmax(axis=1)

    y_test_arr = np.array([np.where(r == 1)[0][0] for r in y_test])

    # Plot relevant graphs
    model_dir = r'/home/dsi/shermad1/Emotion_Recognition/Models/project2_models/'
    if args.model_load == 'NO':
        plotting_function(args, model, model_dir, history, y_test_arr, preds, dic, args.model_type, args.dataset)
    if args.model_load == 'YES':
        plotting_function(args, model, model_dir, model_dir, y_test_arr, preds, dic, args.model_type, args.dataset)

    # compare_results(y_test_arr, preds, dic)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='RAVDESS', type=str, help='dataset name: IEMOCAP or RAVDESS')
    parser.add_argument('--create_data', default='NO', type=str, help='choosing to use existing data or create it: YES or NO')
    parser.add_argument('--model_type', default='Attention', type=str, help='choosing model type: LSTM or BLSTM or Attention')
    parser.add_argument('--model_load', default='YES', type=str, help='creating to load model: YES or NO')
    parser.add_argument('--create_TSNE_data', default='NO', type=str, help='choosing to use existing data or create it: YES or NO')
    parser.add_argument('--TSNE', default='YES', type=str, help='creating TSN: YES or NO')
    parser.add_argument('--reverberation', default='NO', type=str, help='using reverberation data: YES or NO')

    args = parser.parse_args()

    if args.TSNE == 'NO':
        main(args)

    elif args.TSNE == 'YES':
        TSNE_features(args)
        TSNE_model(args)
