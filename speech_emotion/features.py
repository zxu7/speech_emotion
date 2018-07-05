import librosa
import numpy as np


class FeatureExtractor(object):
    """return X_features"""
    def __init__(self, rate):
        self.rate = rate

    def get_mfcc_across_data(self, X, n_mfcc=13):
        """get mean of mfcc features across frame"""
        print("building mfcc features...")
        X_features = np.apply_along_axis(lambda x: np.mean(librosa.feature.mfcc(x, sr=self.rate,
                                                                                      n_mfcc=n_mfcc),
                                                                 axis=0),
                                               1, X)
        return X_features

    def get_mfcc_across_features(self, X, n_mfcc=13):
        """get mean, variance, max, min of mfcc features across feature"""
        def _get_mfcc_features(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            mean = np.mean(mfcc_data, axis=1)
            var = np.var(mfcc_data, axis=1)
            maximum = np.max(mfcc_data, axis=1)
            minimum = np.min(mfcc_data, axis=1)
            out = np.array(list(mean) + list(var) + list(maximum) + list(minimum))
            return out

        X_features = np.apply_along_axis(_get_mfcc_features, 1, X)
        return X_features

    # def get_mfcc_across_data(self, train_X, valid_X, n_mfcc=13):
    #     """get mean of mfcc features across frame"""
    #     print("building mfcc features...")
    #     train_X_features = np.apply_along_axis(lambda x: np.mean(librosa.feature.mfcc(x, sr=self.rate,
    #                                                                                   n_mfcc=n_mfcc),
    #                                                              axis=0),
    #                                            1, train_X)
    #     valid_X_features = np.apply_along_axis(lambda x: np.mean(librosa.feature.mfcc(x, sr=self.rate,
    #                                                                                   n_mfcc=n_mfcc),
    #                                                              axis=0),
    #                                            1, valid_X)
    #     return train_X_features, valid_X_features
    #
    # def get_mfcc_across_features(self, train_X, valid_X, n_mfcc=13):
    #     """get mean, variance, max, min of mfcc features across feature"""
    #     def _get_mfcc_features(x):
    #         mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
    #         mean = np.mean(mfcc_data, axis=1)
    #         var = np.var(mfcc_data, axis=1)
    #         maximum = np.max(mfcc_data, axis=1)
    #         minimum = np.min(mfcc_data, axis=1)
    #         out = np.array(list(mean) + list(var) + list(maximum) + list(minimum))
    #         return out
    #
    #     train_X_features = np.apply_along_axis(_get_mfcc_features, 1, train_X)
    #     valid_X_features = np.apply_along_axis(_get_mfcc_features, 1, valid_X)
    #     return train_X_features, valid_X_features