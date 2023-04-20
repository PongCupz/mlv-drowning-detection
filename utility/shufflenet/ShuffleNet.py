import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

from utility.utils import train_model, test_model

class ShuffleNet:
    def __init__(self):
        self.data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'test':
            transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),

        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self,input_path,model_name,batch_size=32,epochs=3):
        dataset = datasets.ImageFolder(input_path, self.data_transforms['train'])
        test = int(len(dataset) * 0.8)
        val = len(dataset) - test
        test_data,val_date = torch.utils.data.random_split(dataset,[test,val])
        image_datasets = {
            'train': test,
            'validation': val
        }
        dataloaders = {
            'train':
            torch.utils.data.DataLoader(test_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0),
            'validation':
            torch.utils.data.DataLoader(val_date,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0)
        }

        model = models.shufflenet_v2_x2_0(weights='DEFAULT').to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        model_trained = train_model(
            device=self.device,
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            dataloaders=dataloaders, 
            image_datasets=image_datasets, 
            model_name=model_name,
            num_epochs=epochs,
            title="ShuffleNet"
        )

        torch.save(model_trained.state_dict(), f"models/{model_name}.h5")

    def test(self,test_set,model_name):

        model = models.shufflenet_v2_x2_0(weights='DEFAULT').to(self.device)
        model.load_state_dict(torch.load(f"models/{model_name}.h5"))

        dataset = datasets.ImageFolder(test_set, self.data_transforms['test'])
        testloader = torch.utils.data.DataLoader(dataset, batch_size=50,shuffle=False, num_workers=0)

        test_model(model, testloader,model_name, title="ShuffleNet")

