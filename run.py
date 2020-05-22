import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

from sklearn.model_selection import train_test_split

import os
import cv2
import shutil  
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from matplotlib import image

# memory fix
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)


PATH = os.getcwd()
DATASET = os.path.join(PATH, "dataset")
CATEGORIES = [folder for folder in os.listdir(DATASET)]
INDEX_VALUES = [i for i in range(30)]


def create_training_data():
    _images = []
    _labels = []
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        new_path = os.path.join(DATASET, category)
        for img in os.listdir(new_path):
            img_array = cv2.imread(os.path.join(new_path, img), cv2.IMREAD_GRAYSCALE)
            _images.append(img_array)
            _labels.append(class_num)
    return (_images, _labels)

(training_images, training_labels) = create_training_data()
training_images = np.array(training_images)
training_images = training_images / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    training_images, training_labels, test_size=0.3, random_state=42)

EPOCHS = hp.HParam('epochs', hp.IntInterval(30, 100))
NEURONS = hp.HParam('num_units', hp.Discrete([128, 256]))
DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['Adam', 'NAdam', 'SGD']))

METRIC_ACCURACY = 'accuracy'


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[NEURONS, DROPOUT, OPTIMIZER, EPOCHS],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams): 
    log_dir = "logs\\fit\\" \
    + "neurons-" + str(hparams[NEURONS]) + " " \
    + "dropout-" + str(hparams[DROPOUT]) + " " \
    + "epochs-" + str(hparams[EPOCHS]) + " " \
    + str(hparams[OPTIMIZER]) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[NEURONS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[DROPOUT]),
        tf.keras.layers.Dense(30, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer=hparams[OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(np.array(X_train), np.array(y_train), epochs=hparams[EPOCHS])
    _, accuracy = model.evaluate(np.array(X_train), np.array(y_train)
    ,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir),  
        hp.KerasCallback(log_dir, hparams), 
    ],)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for epoch in range(EPOCHS.domain.min_value, EPOCHS.domain.max_value):
    for num_units in NEURONS.domain.values:
        for dropout_rate in (DROPOUT.domain.min_value, DROPOUT.domain.max_value):
            for optimizer in OPTIMIZER.domain.values:
                hparams = {
                    NEURONS: num_units,
                    DROPOUT: dropout_rate,
                    OPTIMIZER: optimizer,
                    EPOCHS: epoch
                }
                run_name = f"run-{session_num}_{epoch}_{num_units}_{dropout_rate}_{optimizer}"
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1