from utility.resnest50 import ResNet50
from utility.alexnet import AlexNet
from utility.squeezenet import SqueezeNet
from utility.googlenet import GoogleNet
from utility.shufflenet import ShuffleNet
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=False,help="epochs")
args = vars(ap.parse_args())

epochs = int(args["epochs"]) if args["epochs"] else 20

print('#' * 5)
print(f"# {epochs} epochs")
rest_net = ResNet50.ResNet50()
rest_net.train(input_path="../data/mlv/train",model_name="resnet",batch_size=50,epochs=epochs)
alexnet = AlexNet.AlexNet()
alexnet.train(input_path="../data/mlv/train",model_name="alexnet",batch_size=50,epochs=epochs)
squeezenet = SqueezeNet.SqueezeNet()
squeezenet.train(input_path="../data/mlv/train",model_name="squeezenet",batch_size=50,epochs=epochs)
googlenet = GoogleNet.GoogleNet()
googlenet.train(input_path="../data/mlv/train",model_name="googlenet",batch_size=50,epochs=epochs)
shufflenet = ShuffleNet.ShuffleNet()
shufflenet.train(input_path="../data/mlv/train",model_name="shufflenet",batch_size=50,epochs=epochs)