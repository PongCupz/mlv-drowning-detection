from utility.resnest50 import ResNet50
from utility.alexnet import AlexNet
from utility.squeezenet import SqueezeNet
from utility.googlenet import GoogleNet
from utility.shufflenet import ShuffleNet

rest_net = ResNet50.ResNet50()
rest_net.test(test_set="../data/mlv/test",model_name="resnet")

alexnet = AlexNet.AlexNet()
alexnet.test(test_set="../data/mlv/test",model_name="alexnet")

squeezenet = SqueezeNet.SqueezeNet()
squeezenet.test(test_set="../data/mlv/test",model_name="squeezenet")

googlenet = GoogleNet.GoogleNet()
googlenet.test(test_set="../data/mlv/test",model_name="googlenet")

shufflenet = ShuffleNet.ShuffleNet()
shufflenet.test(test_set="../data/mlv/test",model_name="shufflenet")