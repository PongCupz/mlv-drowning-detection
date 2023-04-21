# Deep learning and vision-based early drowning detection

###Using 5 models
*   ResNet50
*   AlexNet
*   GoogleNet
*   Squeezenet
*   ShuffleNet

###To Train
```
python3 train.py -d {dataset} -m {model_name} -t resnet -e 20
python3 train.py -d {dataset} -m {model_name} -t alexnet -e 20
python3 train.py -d {dataset} -m {model_name} -t googlenet -e 20
python3 train.py -d {dataset} -m {model_name} -t shufflenet -e 20
python3 train.py -d {dataset} -m {model_name} -t squeezenet -e 20
```

###To Test
```
python3 test.py -d {dataset} -m {model_name} -t resnet
python3 test.py -d {dataset} -m {model_name} -t alexnet
python3 test.py -d {dataset} -m {model_name} -t googlenet
python3 test.py -d {dataset} -m {model_name} -t shufflenet
python3 test.py -d {dataset} -m {model_name} -t squeezenet
```