"""test"""
import pyaudio
import numpy as np
from speech_emotion.classifiers import RuleBasedClassifier


CHUNK = 1024  # ADC information [8 bit = 2⁸ steps]
FORMAT = pyaudio.paInt16  # 16 bit encoding
CHANNELS = 1
RATE = 16000  # sample rate
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "output.wav"


def main():
    label = None
    p = pyaudio.PyAudio()
    RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
    print("DEBUG: default sample_rate of device is: ", RATE)
    classifier = RuleBasedClassifier(sample_rate=RATE)

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
        new_label = classifier.classify(data)
        if label != new_label:
            print(new_label)
            label = new_label

    print("*done recording")


if __name__ == "__main__":
    main()
