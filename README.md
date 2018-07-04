# Speech Emotion Recognition

## Requirements
* Python3
* pyaudio (lastest)
* numpy (latest)
* keras (latest)
* tensorflow (latest)
* librosa (latest)
* tqdm (latest)

## About
Rule-based approach to identify emotion in speech. 

#### Rule1:

```angular2html
# 1、 大声吵闹: 该一秒(t2)的平均声强比上一秒(t1)增加50%(thresh1)以上, 并在后面连续5秒(t3)以上在此位置震荡，不会低于增加后的声强的20%(thresh2)
# 2、 低声私语: 该一秒(t2)的平均声强比上一秒(t1)减少50%(thresh1)以上, 并在后面连续5秒(t3)以上在此位置震荡，不会高于增加后的声强的20%(thresh2)
# 3、 正常说话，除以上场景外
```

    
## Examples
checkout `main.py` for an example using data from microphone.
