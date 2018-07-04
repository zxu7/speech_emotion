"""
data pipeline for loading ravdess, yscz data
model
save weights

"""

import os
import keras
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

T_CLASSIFY = 2
CHUNK = 1024
# FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
N_TIME_TEST = 20
DATA_TO_USE = 'yscz'
FEATURES_TO_USE = "mfcc-on-feature"
AVAILABLE_FEATURES_TO_USE = ("mfcc-on-data", "mfcc-on-feature",)
AVAILABLE_DATA = ("yscz", "ravdess",)
if DATA_TO_USE == 'ravdess':
    DATA_PATH = '/Users/harryxu/school/data/Audio_Speech_Actors_01-24' if os.environ['PWD'].startswith('/Users/harryxu') \
        else '/data/harry/speech_emotion/Audio_Speech_Actors_01-24'
elif DATA_TO_USE == 'yscz':
    DATA_PATH = '/Users/harryxu/school/data/yscz-sound' if os.environ['PWD'].startswith('/Users/harryxu') \
        else '/data/lhy/sound'
CLS_LABEL_DICT = {
    'neutral-normal': '正常说话',
    'calm-normal': '正常说话',
    'happy-normal': '正常说话',
    'surprise-normal': '正常说话',
    'calm-strong': '窃窃私语',
    'sad-normal': '窃窃私语',
    'sad-strong': '窃窃私语',
    'fearful-normal': '窃窃私语',
    'happy-strong': '大声争吵',
    'angry-normal': '大声争吵',
    'angry-strong': '大声争吵',
    'disgust-strong': '大声争吵',
}
assert FEATURES_TO_USE in AVAILABLE_FEATURES_TO_USE, "{} not in {}!".format(FEATURES_TO_USE,
                                                                            AVAILABLE_FEATURES_TO_USE)
assert DATA_TO_USE in AVAILABLE_DATA, "{} not in {}!".format(DATA_TO_USE,
                                                             AVAILABLE_DATA)


def process_data_yscz(path, t=2, n_samples=50):
    """construct dataset X, y from RAVDESS, each RAVDESS wavefile is sampled
    n_samples times at t seconds; Meanwhile, a meta dictionary is built which has
     key: wavefile basename, values: {'X', 'y', 'path'}; 'X' is stacked numpy array
     of normalized wave amplitude samples, y is a list of string labels,
     path is path/to/wavefile"""
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    meta_dict = {}

    LABEL_DICT = {
        '小声': '窃窃私语',
        '正常': '正常说话',
        '大声': '大声争吵',
    }

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(wav_files)):
        label = LABEL_DICT[os.path.basename(wav_file)[:2]]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        for index in np.random.choice(range(len(wav_data) - t*RATE), n_samples, replace=False):
            X1.append(wav_data[index : (index+t*RATE)])
            y1.append(label)
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    X = []
    y = []
    for k in meta_dict:
        X.append(meta_dict[k]['X'])
        y += meta_dict[k]['y']
    X = np.row_stack(X)
    y = np.array(y)
    assert len(X) == len(y), "X length and y length must match! X shape: {}, y length: {}".format(X.shape, y.shape)
    return X, y


def process_data_ravdess(path, t=2, n_samples=3):
    """construct dataset X, y from yscz dataset, each wavefile is sampled
        n_samples times at t seconds; Meanwhile, a meta dictionary is built which has
         key: wavefile basename, values: {'X', 'y', 'path'}; 'X' is stacked numpy array
         of normalized wave amplitude samples, y is a list of string labels,
         path is path/to/wavefile"""
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*/*.wav')
    meta_dict = {}

    LABEL_DICT1 = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    LABEL_DICT2 = {
        '01': 'normal',
        '02': 'strong',
    }

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(wav_files)):
        label1 = LABEL_DICT1[str(os.path.basename(wav_file).split('-')[2])]
        label2 = LABEL_DICT2[str(os.path.basename(wav_file).split('-')[3])]
        label = label1 + '-' + label2
        if label not in CLS_LABEL_DICT.keys():
            continue
        label = CLS_LABEL_DICT[label]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        for index in np.random.choice(range(len(wav_data) - t * RATE), n_samples, replace=False):
            X1.append(wav_data[index: (index + t * RATE)])
            y1.append(label)
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    X = []
    y = []
    for k in meta_dict:
        X.append(meta_dict[k]['X'])
        y += meta_dict[k]['y']
    X = np.row_stack(X)
    y = np.array(y)
    assert len(X) == len(y), "X length and y length must match! X shape: {}, y length: {}".format(X.shape, y.shape)
    return X, y


class FeatureExtractor(object):
    """return train_X_features, valid_X_features"""
    def __init__(self, train_X_data, valid_X_data, rate):
        self.train_X = train_X_data
        self.valid_X = valid_X_data
        self.rate = rate

    def get_mfcc_across_data(self, n_mfcc=13):
        """get mean of mfcc features across frame"""
        print("building mfcc features...")
        train_X_features = np.apply_along_axis(lambda x: np.mean(librosa.feature.mfcc(x, sr=self.rate,
                                                                                      n_mfcc=n_mfcc),
                                                                 axis=0),
                                               1, self.train_X)
        valid_X_features = np.apply_along_axis(lambda x: np.mean(librosa.feature.mfcc(x, sr=self.rate,
                                                                                      n_mfcc=n_mfcc),
                                                                 axis=0),
                                               1, self.valid_X)
        return train_X_features, valid_X_features

    def get_mfcc_across_features(self, n_mfcc=13):
        """get mean, variance, max, min of mfcc features across feature"""
        def _get_mfcc_features(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            mean = np.mean(mfcc_data, axis=1)
            var = np.var(mfcc_data, axis=1)
            maximum = np.max(mfcc_data, axis=1)
            minimum = np.min(mfcc_data, axis=1)
            out = np.array(list(mean) + list(var) + list(maximum) + list(minimum))
            return out

        train_X_features = np.apply_along_axis(_get_mfcc_features, 1, self.train_X)
        valid_X_features = np.apply_along_axis(_get_mfcc_features, 1, self.valid_X)
        return train_X_features, valid_X_features


def train():
    # process data to X, Y format
    if DATA_TO_USE == 'ravdess':
        X, y = process_data_ravdess(DATA_PATH)
    elif DATA_TO_USE == 'yscz':
        X, y = process_data_yscz(DATA_PATH)
    lb_encoder = LabelEncoder()
    y = lb_encoder.fit_transform(y)
    print(X.shape)
    print(y.shape)

    # split data to train/valid
    print("splitting data to train/valid...")
    n = len(X)
    train_indices = list(np.random.choice(range(n), int(n*0.9), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    train_X = X[train_indices]
    train_y = y[train_indices]
    valid_X = X[valid_indices]
    valid_y = y[valid_indices]

    # extract features: mfccs
    feature_extractor = FeatureExtractor(train_X, valid_X, rate=RATE)
    if FEATURES_TO_USE in ('mfcc-on-data', ):
        train_X_features, valid_X_features = feature_extractor.get_mfcc_across_data(n_mfcc=13)
    elif FEATURES_TO_USE in ('mfcc-on-feature', ):
        train_X_features, valid_X_features = feature_extractor.get_mfcc_across_features(n_mfcc=13)

    # normalize for easier CNN training
    train_X_features_factors = np.std(abs(train_X_features), axis=0)
    train_X_features /= train_X_features_factors
    valid_X_features /= train_X_features_factors
    train_X_features = np.expand_dims(train_X_features, 2)
    valid_X_features = np.expand_dims(valid_X_features, 2)
    n_class = len(lb_encoder.classes_)
    d = train_X_features.shape[1]
    print("training on {} data ... valid on {} data... dimension is {}...".format(train_X_features.shape[0],
                                                                                  valid_X_features.shape[0],
                                                                                  d))

    # build model and fit
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
    opt = keras.optimizers.rmsprop(lr=1e-4, decay=1e-6)
    opt = keras.optimizers.Adam(1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    ckpt = keras.callbacks.ModelCheckpoint('experiments/mfcc_model.hdf5', save_best_only=True)
    history = model.fit(train_X_features, train_y, batch_size=32, epochs=200,
                        validation_data=(valid_X_features, valid_y),
                        callbacks=[ckpt, ])

    # report
    pred_valid_y = model.predict_classes(valid_X_features, )
    print("accuracy on validation set is {}...".format(accuracy_score(valid_y, pred_valid_y)))
    cnf_matrix = confusion_matrix(valid_y, pred_valid_y)
    print(cnf_matrix)
    print("class names are {}...".format(lb_encoder.classes_))

    print("timing {} prediction...".format(N_TIME_TEST))
    indices = np.random.choice(list(range(train_X_features.shape[0])), N_TIME_TEST, replace=False)
    times = []
    for i in range(N_TIME_TEST):
        start = time.time()
        sample_data = np.expand_dims(train_X_features[indices[i]], 0)
        model.predict(sample_data)
        times.append(time.time() - start)
    print("average prediction time for each sample is {}...".format(np.mean(times)))


if __name__ == '__main__':
    train()