import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from imblearn.metrics import specificity_score, sensitivity_score

def train_model(device, model, criterion, optimizer, dataloaders, image_datasets, model_name, num_epochs=3, title=""):
    print(title)
    h = {
        "train" : {
            "loss" : [],
            "acc" : [],
        },
        "validation" : {
            "loss" : [],
            "acc" : [],
        }
    }
    t0 = time.time()
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

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / image_datasets[phase]
            epoch_acc = running_corrects.double() / image_datasets[phase]

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            h[phase]["loss"].append(epoch_loss)
            h[phase]["acc"].append(epoch_acc)

    print("Training time: {:.2f}s".format(time.time()-t0))
    print('-' * 10)
    print(' ')
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(1, num_epochs + 1), h["train"]["loss"], label="train_loss")
    plt.plot(np.arange(1, num_epochs + 1), h["validation"]["loss"], label="val_loss")
    plt.plot(np.arange(1, num_epochs + 1), h["train"]["acc"], label="train_acc")
    plt.plot(np.arange(1, num_epochs + 1), h["validation"]["acc"], label="val_acc")
    plt.title(f"{title} Training Loss and Accuracy")

    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(f"models/{model_name}.png")
    return model

def test_model(model, testloader,model_name, title=""):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ('drowning', 'swimming')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title(f"{title} Confusion Matrix",fontsize=17)
    
    plt.savefig(f'output/confusion_matrix_{model_name}.png')

    # Finding precision and recall
    print(title)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy   :", accuracy)
    precision = precision_score(y_true, y_pred)
    print("Precision :", precision)
    recall = recall_score(y_true, y_pred)
    print("Recall    :", recall)
    F1_score = f1_score(y_true, y_pred)
    print("F1-score  :", F1_score)
    MMC_score = matthews_corrcoef(y_true, y_pred)
    print("MMC-score  :", MMC_score)
    SP_score = specificity_score(y_true, y_pred)
    print("Specificity :", SP_score)
    print('-' * 10)
    print(' ')
    