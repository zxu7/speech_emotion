"""test"""
import pyaudio
import numpy as np
from speech_emotion.classifiers import StreamChunkClassifier
from speech_emotion.models import KerasModel


CHUNK = 1024  # ADC information [8 bit = 2⁸ steps]
FORMAT = pyaudio.paInt16  # 16 bit encoding
CHANNELS = 1
RATE = 44100  # sample rate
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "output.wav"
RULE = 'predict_by_rule2'

# set params here
RULE1_PARAMS = {
    "thresh1": 0.5,
    "thresh2": 0.2,
    "t1": 1,
    "t2": 1,
    "t3": 5}

RULE2_PARAMS = {
    "model_name": 'experiments/44100_yscz_2_mfcc-on-feature_cnn_model.hdf5',
    "time_required": 2,
    "rate": RATE
}

if RULE in ('predict_by_rule1',):
    RULE_PARAMS = RULE1_PARAMS
elif RULE in ('predict_by_rule2',):
    RULE_PARAMS = RULE2_PARAMS
    model = KerasModel()
    model.load(RULE2_PARAMS["model_name"])
    RULE_PARAMS["model"] = model


def main():
    label = None
    p = pyaudio.PyAudio()
    RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
    print("DEBUG: default sample_rate of device is: ", RATE)
    # update all RATE in params
    if 'RATE' in RULE_PARAMS:
        RULE_PARAMS['RATE'] = RATE

    classifier = StreamChunkClassifier(sample_rate=RATE)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=0,
                    frames_per_buffer=CHUNK)  # buffer
    print("*recording")

    frames = np.array([], dtype=np.int16)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = np.fromstring(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        frames = np.concatenate((frames, data))  # 2 bytes(16 bits) per channel
        new_label = classifier.classify(data, rule_params=RULE_PARAMS, classify_func=RULE)
        if label != new_label and new_label is not None:
            print(new_label)
            label = new_label

    print("*done recording")


if __name__ == "__main__":
    main()
