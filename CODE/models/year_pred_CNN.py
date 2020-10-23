from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import numpy as np
import pandas as pd
import os
import sys


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
    elif int(df_new.at[x, "Year"]) < 1990 and int(df_new.at[x, "Year"]) >= 1930:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "eigties"
    elif int(df_new.at[x, "Year"]) < 2000 and int(df_new.at[x, "Year"]) >= 1930:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "nineties"
    else:
        df_new["Year"] = df_new["Year"].astype(str)
        df_new.at[x, "Year"] = "millenial"

datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)


train_generator=datagen.flow_from_dataframe(
    dataframe=df_new,
    directory="../../melspectograms/400dpi",
    x_col="Specs",
    y_col="Year",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(128,128))

valid_generator=datagen.flow_from_dataframe(
    dataframe=df_new,
    directory="../../melspectograms/400dpi",
    x_col="Specs",
    y_col="Year",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(128,128))




model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#model.add(Conv2D(128, (3, 3), padding='same'))
#model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3),  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))
model.compile(optimizers.Adam(lr=0.0005, decay=1e-6),loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150
)
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID
)