import torch
import cv2
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datatest", required=True,help="path to input test vdo")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-t", "--type", required=True,help="model type")
args = vars(ap.parse_args())

model_type = args["type"]

data_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

data_resize = {
    'resnet':224,
    'alexnet': 256,
    'squeezenet':224,
    'googlenet': 256,
    'shufflenet':232
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_type == 'resnet' :
    model = models.resnet50(weights='DEFAULT').to(device)
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device)
if model_type == 'alexnet' :
    model = models.alexnet(weights='DEFAULT').to(device)

if model_type == 'squeezenet' :
    model = models.squeezenet1_1(weights='DEFAULT').to(device)

if model_type == 'googlenet' :
    model = models.googlenet(weights='DEFAULT').to(device)

if model_type == 'shufflenet' :
    model = models.shufflenet_v2_x2_0(weights='DEFAULT').to(device)


model_name = args["model"]
model.load_state_dict(torch.load(f"models/{model_name}.h5"))

player = cv2.VideoCapture(args["datatest"])


def read_cam(video_capture):
    if video_capture.isOpened():
        n = 0
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(f'output/vdo-output-{model_name}.avi', four_cc, 20, (x_shape, y_shape))

        x = int(x_shape/2 - 120/2)
        y = int(y_shape/2 - 200/2)
        position = (x, y)
        bgr = (0, 0, 255)

        n = 0
        while True:
            ret, frame = player.read()
            if not ret:
                break

            framex = cv2.resize(frame, (data_resize[model_type], data_resize[model_type]))
            batch_tensor = torch.stack([data_transforms(framex).to(device)])
            pred_logits_tensor = model(batch_tensor)
            pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
            print("{:.0f}% Drowing, {:.0f}% Swimming".format(100*pred_probs[0,0],100*pred_probs[0,1]))
            if pred_probs[0,0] > 0.5 :
                n = n + 1
            else :
                n = 0
            if n > 10 :
                cv2.rectangle(frame, (x - 25 ,y - 30), (x + 180, y + 20), (0,0,255), 1)
                cv2.putText(frame, "Drowning !", position, cv2.FONT_HERSHEY_TRIPLEX, 0.9, bgr, 2)
                
            out.write(frame)


read_cam(player)