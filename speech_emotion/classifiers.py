import os
import numpy as np
from .features import FeatureExtractor


class StreamChunkClassifier(object):
    """
    An classifier Object  that reads in audio data and output a predicted label.
    The Object uses rule-based methods to classifiy sound amplitudes.
    声强数据的分类器。现阶段仅支持rule1。
    rule1解释：classify方法读入list/numpy array形式的audio_data数据并存储在类变量
    self.collected_samples中。分类逻辑仅使用self.collected_samples中的后samples_required个数据，
    若self.collected_samples中的数据量不够samples_required，则不进行分析。
    audio data来源可以是麦克风，比如，采样率44100的麦克风以CHUNK=1024读入数据的话，
    每个audio data为长度1024的list/numpy array。
    """
    def __init__(self, sample_rate=44100):
        self.SAMPLE_RATE = sample_rate
        self.collected_samples = []
        self.feature_extractor = None

    def classify(self, audio_data, rule_params, classify_func='predict_by_rule1'):
        """
        read new chunk of audio data and classify
        :param: audio_data: list/numpy array
        :param: rule_params: dict;
        :return: None or label
        """
        out = None
        ACCEPTED_RULES = ('predict_by_rule1', 'predict_by_rule2')
        assert classify_func in ACCEPTED_RULES, "{} not understood!".format(classify_func)

        self.collected_samples += list(audio_data)

        if classify_func in ('predict_by_rule1',):
            out = self.predict_by_rule1(**rule_params)
        elif classify_func in ('predict_by_rule2',):
            out = self.predict_by_rule2(**rule_params)

        return out

    def predict_by_rule1(self, thresh1=0.5, thresh2=0.2, t1=1, t2=1, t3=5):
        """
        1、 大声吵闹: 该一秒(t2)的平均声强比上一秒(t1)增加50%(thresh1)以上, 并在后面连续5秒(t3)以上在此位置震荡，不会低于增加后的声强的20%(thresh2)
        2、 低声私语: 该一秒(t2)的平均声强比上一秒(t1)减少50%(thresh1)以上, 并在后面连续5秒(t3)以上在此位置震荡，不会高于增加后的声强的20%(thresh2)
        3、 正常说话，除以上场景外
        :param thresh1: float; 在上述例子中为50% (0.5)
        :param thresh2: float; 在上述例子中为20% (0.2)
        :param t1: float; 在上述例子中为上一秒 (1)
        :param t2: float; 在上述例子中为该一秒 (1)
        :param t3: float; 在上述例子中为后面连续五秒 (5)
        :return pred_label: None or str; return None if not enough samples are provided
        """
        time_required = t1 + t2 + t3
        samples_required = self.SAMPLE_RATE * time_required

        if len(self.collected_samples) < samples_required:
            # not enough samples to analyze
            return None
        else:
            # truncate list
            self.collected_samples = self.collected_samples[-samples_required:]

            samples_t1 = self.collected_samples[0:int(self.SAMPLE_RATE * t1)]
            samples_t2 = self.collected_samples[int(self.SAMPLE_RATE * t1):int(self.SAMPLE_RATE * (t1 + t2))]
            samples_t3 = self.collected_samples[
                         int(self.SAMPLE_RATE * (t1 + t2)):int(self.SAMPLE_RATE * (t1 + t2 + t3))]
            avg_amp_t1 = np.mean(samples_t1)
            avg_amp_t2 = np.mean(samples_t2)
            # avg amps over t3
            avg_amps_t3 = [np.mean(samples_t3[i * self.SAMPLE_RATE:(i + 1) * self.SAMPLE_RATE]) for i in range(t3)]

            # 并在后面连续5秒(t3)以上在此位置震荡，不会低于增加后的声强的20%(thresh2)
            lower_rule_truefalse = [(avg_amp_t2 - avg_amp_t3) / avg_amp_t2 < thresh2 for avg_amp_t3 in avg_amps_t3]
            # 并在后面连续5秒(t3)以上在此位置震荡，不会高于增加后的声强的20%(thresh2)
            higher_rule_truefalse = [(avg_amp_t3 - avg_amp_t2) / avg_amp_t2 < thresh2 for avg_amp_t3 in avg_amps_t3]

            # 1、 大声吵闹
            if (avg_amp_t2 - avg_amp_t1) / avg_amp_t1 >= thresh1 and np.all(lower_rule_truefalse):
                pred_label = "大声吵闹"
            # 2、 低声私语
            elif (avg_amp_t1 - avg_amp_t2) / avg_amp_t1 >= thresh1 and np.all(lower_rule_truefalse):
                pred_label = "低声私语"
            # 3、 正常说话
            else:
                pred_label = "正常说话"

            return pred_label

    def predict_by_rule2(self, time_required, model, model_name, normalize=True, rate=None, interval=1):
        """
        KerasModel based classifier
        :param time_required: int; number of seconds required to input to model
        :param model: KerasModel Object;
        :param model_name: str; path/to/model
        :param normalize: bool; normalize amplitude data by dividing by (2**15-1)
        :param rate: int; sample rate
        :param interval: int; number of seconds between predictions, this number MUST be smaller than time_required
        :return:
        """
        # TODO: give warning if model is trained on a different sample rate from self.SAMPLE_RATE
        sr, dataset, t_required, features_to_use, _, _ = os.path.basename(model_name).split('_')
        samples_required = self.SAMPLE_RATE * time_required
        interval_samples = self.SAMPLE_RATE * interval
        if len(self.collected_samples) < samples_required:
            # not enough samples to analyze
            return None
        else:
            # truncate list
            self.collected_samples = self.collected_samples[-samples_required:]
            if normalize:
                collected_samples = np.expand_dims(np.array(self.collected_samples) / (2**15 - 1), 0)
            else:
                collected_samples = np.expand_dims(np.array(self.collected_samples), 0)
            # extract features
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor(rate=rate)
            collected_samples_features = self.feature_extractor.get_features(features_to_use,
                                                                             collected_samples,
                                                                             n_mfcc=13)
            pred_label = model.predict(collected_samples_features, verbose=0, predict_label=True)
            # truncate list to interval
            self.collected_samples = self.collected_samples[-interval_samples:]
            return pred_label
