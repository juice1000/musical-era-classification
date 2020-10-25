from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys


save_dir = '../../saved_models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Auxillary Function for getting model name througout validation
def get_model_name():
    return save_dir + '/model.h5'


def CNN_model(filters, depth):
    model = Sequential()
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                     input_shape=(64, 64, 3), activation="relu"))
    for i in range(depth):
        model.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    for i in range(depth):
        model.add(Conv2D(filters=filters*2, kernel_size=(3, 3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation="relu")) # Dense layers upon individual change!
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizers.Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

    # Possible changes:
    # Optim to 0.0001
    # More Data
    # Change NN width/ length again :)

def data_preprocessing():

    df = pd.read_csv('../../data_library/preprocessing_data/url_data_CNN.csv')

    columns = ["Year", "ID", "Artist", "Title", "URL"]
    df.columns = columns
    df["Specs"] = df["Artist"] + " -- " + df["Title"] + ".png"

    df = df.values
    DIR = '../../melspectograms'

    df_new = []

    for j in range(len(df)):

        filename = df[j][2] + " -- " + df[j][3] + ".png"
        if os.path.isfile(os.path.join(DIR, filename)):
            df_new.append(df[j])

    columns.append("Specs")
    df_new = pd.DataFrame(df_new, columns=columns)
    print(len(df_new))

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

    pd.get_dummies(df['country'], prefix='country')
    datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)


    train_generator=datagen.flow_from_dataframe(
        dataframe=df_new,
        directory="../../melspectograms/400dpi",
        x_col="Specs",
        y_col="Year",
        subset="training",
        batch_size=20,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64, 64))

    valid_generator=datagen.flow_from_dataframe(
        dataframe=df_new,
        directory="../../melspectograms/400dpi",
        x_col="Specs",
        y_col="Year",
        subset="validation",
        batch_size=20,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64, 64))

    return train_generator, valid_generator


if __name__ == "__main__":

    print("\n INITIALIZE TRAINING")

    with tf.device("/cpu:0"):
        df = pd.DataFrame(index=["2 Conv2D", "3 Conv2D"], columns=["32 Filters", "64 Filters", "128 Filters"])
        for i in range(5, 8):
            filters = 2**i
            train_generator, valid_generator = data_preprocessing()

            for depth in range(2, 4):
                scores = []
                losses = []
                model = CNN_model(filters, depth)

                #Fitting keras model, no test gen for now
                STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
                STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
                #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
                es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
                mc = ModelCheckpoint(get_model_name(), monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
                callbacks = [es, mc]

                with tf.device("/gpu:0"):
                    model.fit_generator(generator=train_generator,
                                        steps_per_epoch=STEP_SIZE_TRAIN,
                                        validation_data=valid_generator,
                                        validation_steps=STEP_SIZE_VALID,
                                        epochs=200,
                                        callbacks=callbacks
                    )


                score = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID
                )
                score = np.array(score)
                print("EVALUATION - %s: %.2f%%, loss:" % (model.metrics_names[1], score[1] * 100), score[0])

                row = "{} Conv2D".format(depth)
                column = "{} Filters".format(filters)
                df.loc[row, column] = score

            df.to_csv("../../NN_eval/CNN_eval_librosa_audiofeatures4.csv")
