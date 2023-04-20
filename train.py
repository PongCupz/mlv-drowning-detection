from utility.resnest50 import ResNet50
from utility.alexnet import AlexNet
from utility.squeezenet import SqueezeNet
from utility.googlenet import GoogleNet
from utility.shufflenet import ShuffleNet
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help=".path to output dataset")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-t", "--type", required=True,help="model type")
args = vars(ap.parse_args())

if args["type"] == 'resnet' :
    rest_net = ResNet50.ResNet50()
    rest_net.train(input_path=args["dataset"],model_name=args["model"],batch_size=50,epochs=20)

if args["type"] == 'alexnet' :
    alexnet = AlexNet.AlexNet()
    alexnet.train(input_path=args["dataset"],model_name=args["model"],batch_size=50,epochs=20)

if args["type"] == 'squeezenet' :
    squeezenet = SqueezeNet.SqueezeNet()
    squeezenet.train(input_path=args["dataset"],model_name=args["model"],batch_size=50,epochs=20)


if args["type"] == 'googlenet' :
    googlenet = GoogleNet.GoogleNet()
    googlenet.train(input_path=args["dataset"],model_name=args["model"],batch_size=50,epochs=1)

if args["type"] == 'shufflenet' :
    shufflenet = ShuffleNet.ShuffleNet()
    shufflenet.train(input_path=args["dataset"],model_name=args["model"],batch_size=50,epochs=20)