
# Import relevant libraries
import keras
from keras.regularizers import l1
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, Dot, Softmax, Lambda
from tensorflow.keras.optimizers import Adam


# Function to create the model
def create_model(X_train, y_train, model_type, last_layer=True):

    input_y = Input(shape=X_train.shape[1:], name='Input_MELSPECT')

    # First LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT')(input_y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

    # Second LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT')(y)

    # Third LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT')(y)

    # Fourth LFLB (local feature learning block)
    y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT')(y)
    y = TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT')(y)
    y = TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT')(y)
    y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT')(y)
    y = TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT')(y)

    # # Fifth LFLB (local feature learning block)
    # y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_5_MELSPECT')(y)
    # y = TimeDistributed(BatchNormalization(), name='BatchNorm_5_MELSPECT')(y)
    # y = TimeDistributed(Activation('elu'), name='Activ_5_MELSPECT')(y)
    # y = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_5_MELSPECT')(y)
    # y = TimeDistributed(Dropout(0.2), name='Drop_5_MELSPECT')(y)

    # Flattening
    y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)

    if model_type == 'Attention':
        # Apply BLSTM with attention layer
        states, forward_h, _, backward_h, _ = Bidirectional(LSTM(256, return_sequences=True,
                                                return_state=True, activity_regularizer=l1(0.001),name='BLSTM_1'))(y)
        last_state = Concatenate()([forward_h, backward_h])
        hidden = Dense(256, activation="tanh", use_bias=False,
                       kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.))(states)
        out = Dense(1, activation='linear', use_bias=False,
                    kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.))(hidden)
        flat = Flatten()(out)
        energy = Lambda(lambda x: x / np.sqrt(256))(flat)
        normalize = Softmax()
        normalize._init_set_name("alpha")
        alpha = normalize(energy)
        context_vector = Dot(axes=1)([states, alpha])
        context_vector = Concatenate()([context_vector, last_state])

    elif model_type =='BLSTM':
        # Apply BLSTM layer
        context_vector = Bidirectional(LSTM(256, return_sequences=False, activity_regularizer=l1(0.001), name='BLSTM_1'))(y)

    elif model_type == 'LSTM':
        # Apply LSTM layer
        context_vector = LSTM(256, return_sequences=False, activity_regularizer=l1(0.001), name='BLSTM_1')(y)

    # FC layer
    if last_layer:
        y = Dense(y_train.shape[1], activation='softmax', name='FC')(context_vector)
    else:
        y = context_vector

    # Build final model
    model = Model(inputs=input_y, outputs=y)
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.99)
    # opt = RMSprop(learning_rate=0.0001, rho=0.9, momentum=0.0, decay=1e-6)
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

