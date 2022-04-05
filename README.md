# Cross Domain Slot Filling with BERT

* Data: [SNIPS Dataset](https://github.com/sonos/nlu-benchmark) preprocessed by [zliucr](https://github.com/zliucr/coach)
* Download data from [here](https://drive.google.com/drive/folders/1ydalMtB-hpfS3SIEaR5UbRfEe2m8bFcj) and save it in <b>./data</b> diretory.

## Model Configuration
huggingface에서 제공하는 pretrained BERT의 기본 configuration을 따릅니다. 

## How to run
다음과 같이 실행하면 됩니다.
```
python main.py config.json
```

## Options
다음은 주로 사용하는 option입니다. config.json 파일에 option을 주면 됩니다.</br>
config.py 파일을 참고하면 어떤 option이 어떻게 사용되는 지 설명을 명시하였습니다.</br>


* target_domain: test의 대상(target)으로 할 domain, train data는 해당 domain을 제외한 나머지 domain으로 구성
* n_samples: <i>few shot learning</i> 시 target domain으로부터 얼만큼의 data를 train data와 함께 사용할 것인지 지정. 0으로 지정하면 <i>zero-shot learning</i>으로 수행
* learning_rate: learning rate
* dropout_rate: BERT output hidden에서 적용할 dropout rate
* max_steps: 최대 minibatch 학습 step
* eval_steps: 몇 step마다 evaluation 수행할 지
* early_stopping_patience: 가장 좋은 model parameter 발견 후 학습 종료할 patience steps
* run_mode: train 및 test option을 줄 수 있음

## Sample Output
**experiments/BookRestaurant/Sample0** 폴더에 *test output(json 형식)의 sample*을 포함하였습니다.