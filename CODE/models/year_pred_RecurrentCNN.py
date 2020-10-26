from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, GRU, Lambda, concatenate
from tensorflow.keras import regularizers, optimizers
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os
import sys



save_dir = '../../saved_models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Auxillary Function for getting model name throughout validation
def get_model_name():
    return save_dir + '/RCNN.h5'


def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['accurracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



batch_size = 20
num_classes = 6
nb_filters1 = 16
nb_filters2 = 32
nb_filters3 = 64
nb_filters4 = 64
nb_filters5 = 64
ksize = (3, 1)
pool_size_1 = (2, 2)
pool_size_2 = (4, 4)
pool_size_3 = (4, 2)

dropout_prob = 0.20
dense_size1 = 128
lstm_count = 64
num_units = 120

BATCH_SIZE = 64
EPOCH_COUNT = 150
L2_regularization = 0.001

def RCNN_model(model_input):
    print('Building model...')
    layer = model_input

    ### Convolutional blocks
    conv_1 = Conv2D(filters=nb_filters1, kernel_size=ksize, strides=1,
                    padding='valid', activation='relu', name='conv_1')(layer)
    print(conv_1)
    pool_1 = MaxPooling2D(pool_size_1)(conv_1)
    print(pool_1)
    conv_2 = Conv2D(filters=nb_filters2, kernel_size=ksize, strides=1,
                    padding='valid', activation='relu', name='conv_2')(pool_1)
    print(conv_2)
    pool_2 = MaxPooling2D(pool_size_1)(conv_2)
    print(pool_2)
    conv_3 = Conv2D(filters=nb_filters3, kernel_size=ksize, strides=1,
                    padding='valid', activation='relu', name='conv_3')(pool_2)
    print(conv_3)
    pool_3 = MaxPooling2D(pool_size_1)(conv_3)
    print(pool_3)
    conv_4 = Conv2D(filters=nb_filters4, kernel_size=ksize, strides=1,
                    padding='valid', activation='relu', name='conv_4')(pool_3)
    print(conv_4)
    pool_4 = MaxPooling2D(pool_size_2)(conv_4)

    print(pool_4)

    flatten1 = Flatten()(pool_4)
    ### Recurrent Block

    # Pooling layer
    pool_lstm1 = MaxPooling2D(pool_size_3, name='pool_lstm')(layer)

    squeezed = Lambda(lambda x: K.squeeze(x, axis=-1))(pool_lstm1)
    #     flatten2 = K.squeeze(pool_lstm1, axis = -1)
    #     dense1 = Dense(dense_size1)(flatten)

    # Bidirectional GRU
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  # default merge mode is concat

    # Concat Output
    concat = concatenate([flatten1, lstm], axis=-1, name='concat')

    ## Softmax Output
    output = Dense(num_classes, activation='softmax', name='preds')(concat)

    model_output = output
    model = Model(model_input, model_output)

    #     opt = Adam(lr=0.001)
    opt = optimizers.Adam(lr=0.001)  # Optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    print(model.summary())
    return model


# Possible changes:
    # Optim to 0.0001
    # More Dropout / Batchnorm
    # More Data
    # Change NN width/ length again :)


df = pd.read_csv('../../data_library/preprocessing_data/url_data_CNNfiftotwen3000.csv')

columns = ["Year", "ID", "Artist", "Title", "URL"]
df.columns = columns
df["Specs"] = df["Artist"] + " -- " + df["Title"] + ".png"

df = df.values
DIR = '../../melspectograms/400dpi'

df_new = []

for j in range(len(df)):

    filename = df[j][2] + " -- " + df[j][3] + ".png"
    if os.path.isfile(os.path.join(DIR, filename)):
        df_new.append(df[j])

columns.append("Specs")
df_new = pd.DataFrame(df_new, columns=columns)

for x in range(len(df_new)):
    if int(df_new.at[x, "Year"]) < 1930:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "twenties"
    elif int(df_new.at[x, "Year"]) < 1940 and int(df_new.at[x, "Year"]) >= 1930:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "thirties"
    elif int(df_new.at[x, "Year"]) < 1950 and int(df_new.at[x, "Year"]) >= 1940:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "fourties"
    elif int(df_new.at[x, "Year"]) < 1960 and int(df_new.at[x, "Year"]) >= 1950:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "fifties"
    elif int(df_new.at[x, "Year"]) < 1970 and int(df_new.at[x, "Year"]) >= 1960:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "sixties"
    elif int(df_new.at[x, "Year"]) < 1980 and int(df_new.at[x, "Year"]) >= 1970:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "seventies"
    elif int(df_new.at[x, "Year"]) < 1990 and int(df_new.at[x, "Year"]) >= 1980:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "eigties"
    elif int(df_new.at[x, "Year"]) < 2000 and int(df_new.at[x, "Year"]) >= 1990:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "nineties"
    else:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "millenial"



df_new = df_new[["Year", "Specs"]]
print(df_new)

datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)

train_generator = datagen.flow_from_dataframe(
    dataframe=df_new,
    directory="../../melspectograms/400dpi",
    x_col="Specs",
    y_col="Year",
    color_mode="grayscale",
    subset="training",
    batch_size=round(len(df_new)*0.75),
    seed=None,
    shuffle=False,
    save_to_dir="../../melspectograms/400dpi/TimeDistributed",
    class_mode="categorical",
    target_size=(128, 128))

valid_generator = datagen.flow_from_dataframe(
    dataframe=df_new,
    directory="../../melspectograms/400dpi",
    x_col="Specs",
    y_col="Year",
    color_mode="grayscale",
    subset="validation",
    batch_size=round(len(df_new)*0.25),
    shuffle=False,
    seed=False,
    class_mode="categorical",
    target_size=(128, 128))


train_generator.reset()
x_train,y_train=train_generator.next()
x_train=np.array(x_train)
y_train=np.array(y_train)
print(x_train.shape)
print(y_train.shape)

valid_generator.reset()
x_val,y_val=valid_generator.next()
x_val=np.array(x_val)
y_val=np.array(y_val)
print(x_val.shape)
print(y_val.shape)


input_shape = (128, 128, 1)
model_input = Input(input_shape, name='input')
print(model_input)
model = RCNN_model(model_input)
print(model.summary())

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint(get_model_name(), monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
callbacks = [es, mc]

#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks)

show_summary_stats(history)