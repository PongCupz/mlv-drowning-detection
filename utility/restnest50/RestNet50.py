import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import torchvision

class RestNet50:
    def __init__(self,path):
        self.input_path = path

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        self.data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ]),
            'validation':
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                self.normalize
            ]),
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.image_datasets = {
            'train': 
            datasets.ImageFolder(self.input_path  + 'train', self.data_transforms['train']),
            'validation': 
            datasets.ImageFolder(self.input_path  + 'validation', self.data_transforms['validation'])
        }

        self.dataloaders = {
            'train':
            torch.utils.data.DataLoader(self.image_datasets['train'],
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0),  # for Kaggle
            'validation':
            torch.utils.data.DataLoader(self.image_datasets['validation'],
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=0)  # for Kaggle
        }

        self.dataloaders = {
            'train':
            torch.utils.data.DataLoader(self.image_datasets['train'],
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0),  # for Kaggle
            'validation':
            torch.utils.data.DataLoader(self.image_datasets['validation'],
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=0)  # for Kaggle
        }

        model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(self.device)
            
        for param in model.parameters():
            param.requires_grad = False   
            
        model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2)).to(self.device)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())

        model_trained = self.train_model(model, criterion, optimizer, num_epochs=3)
        torch.save(model_trained.state_dict(), 'models/weights-drowning.h5')

    def test(self,test_set):

        model = models.resnet50().to(self.device)
        model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2)).to(self.device)
        model.load_state_dict(torch.load('models/weights-drowning.h5'))

        d_files = glob.glob(test_set) 
        img_list = [Image.open(img_path) for img_path in d_files]

        validation_batch = torch.stack([self.data_transforms['validation'](img).to(self.device)
                                        for img in img_list])

        pred_logits_tensor = model(validation_batch)
        # pred_logits_tensor

        pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

        fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
        for i, img in enumerate(img_list):
            # ax = axs[i]
            # ax.axis('off')
            # ax.set_title("{:.0f}% Cat, {:.0f}% Dog".format(100*pred_probs[i,0],100*pred_probs[i,1]))
            # ax.imshow(img)
            if pred_probs[i,0] < 0.5 :
                print("{:.0f}% Drowing, {:.0f}% Swimming : ".format(100*pred_probs[i,0],100*pred_probs[i,1]))
                print(d_files[i])


    def train_model(self, model, criterion, optimizer, num_epochs=3):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.image_datasets[phase])
                epoch_acc = running_corrects.double() / len(self.image_datasets[phase])

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
        return model

