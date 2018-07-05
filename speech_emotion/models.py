import os
import keras
import h5py
import glob
import time
import librosa
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


class KerasModel(object):
    def __init__(self):
        self.model = None
        self.trained = False
        # model preprocess params
        self.train_X_factor = None
        # other params
        self.class_names = None

    def train(self, train_X, train_y, valid_X, valid_y, n_class, model_path, class_names):
        # model's data preprocess
        self.class_names = class_names
        print("preprocessing features for the model...")
        self._update_preprocess_params(train_X)
        train_X = self._preprocess_data(train_X)
        valid_X = self._preprocess_data(valid_X)

        d = train_X.shape[1]
        print("training on {} data ... valid on {} data... dimension is {}...".format(train_X.shape[0],
                                                                                      valid_X.shape[0],
                                                                                      d))
        self.model = self._build_model_1dcnn(d, n_class)
        ckpt = keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', mode='max', save_best_only=True)
        history = self.model.fit(train_X, train_y, batch_size=32, epochs=20,
                                 validation_data=(valid_X, valid_y),
                                 callbacks=[ckpt, ])
        self.trained = True

        # save preprocess params to .hdf5
        with h5py.File(model_path, mode='a') as f:
            f.attrs['train_X_factor'] = self.train_X_factor
            f.attrs['class_names'] = [n.encode('utf-8', 'ignore') for n in self.class_names]

        # DEBUG
        pred_valid_y = self.model.predict_classes(valid_X, )
        print("accuracy on validation set is {}...".format(accuracy_score(valid_y, pred_valid_y)))
        cnf_matrix = confusion_matrix(valid_y, pred_valid_y)
        print(cnf_matrix)

    def load(self, model_path):
        # load encoders

        # keras save both model graph and weights now
        self.model = keras.models.load_model(model_path)
        # load preprocess params
        with h5py.File(model_path, mode='r') as f:
            self.train_X_factor = f.attrs['train_X_factor']
            self.class_names = f.attrs['class_names']
            self.class_names = [n.decode('utf-8', 'ignore') for n in self.class_names]
        self.trained = True

    def predict(self, X, verbose=1, predict_label=False):
        assert self.trained is True, "call train() or load() before predict()"

        # model data preprocess
        X = self._preprocess_data(X)

        # predict
        output = self.model.predict_classes(X, verbose=verbose)
        if predict_label is True:
            output = [self.class_names[o] for i, o in enumerate(output)]
        return output

    def _update_preprocess_params(self, train_X):
        # use empirical std to standardize data
        self.train_X_factor = np.std(abs(train_X), axis=0)

    def _preprocess_data(self, X):
        processed_X = X / self.train_X_factor
        processed_X = np.expand_dims(processed_X, 2)
        return processed_X

    @staticmethod
    def _build_model_1dcnn(d, n_class):
        model = Sequential()
        model.add(Conv1D(128, 5, padding='same',
                         input_shape=(d, 1), activation='relu'))
        model.add(Conv1D(128, 5, padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPool1D(pool_size=(8,)))
        model.add(Conv1D(128, 5, padding='same', activation='relu'))
        model.add(Conv1D(128, 5, padding='same', activation='relu'))
        model.add(Conv1D(128, 5, padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv1D(128, 5, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(n_class, activation='softmax'))
        # opt = keras.optimizers.rmsprop(lr=1e-4, decay=1e-6)
        opt = keras.optimizers.Adam(1e-3)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model