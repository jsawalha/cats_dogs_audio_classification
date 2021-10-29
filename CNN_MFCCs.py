import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns  # for heatmaps
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from PIL import Image
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

#Check the full file path of where these images are

with Image.open(r'/home/jeff/Documents/cats_dogs_audio_signal_processing/cats_dogs/non_pad_MFCC_train/cat/cat_32.jpeg') as im:
    plt.imshow(im)


#Set up top level paths

path = r'/home/jeff/Documents/cats_dogs_audio_signal_processing/cats_dogs/non_pad_MFCC_train'

img_names = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '/' + img)

#Check the length of image
len(img_names)

#These images have a variety of sizes, width and heights, and more complex, later, we need to transform images
#create a data frame to look at sizes of the images


img_sizes = []
rejected = []

for item in img_names:

    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)

    except:
        rejected.append(item)

# Rejected = see if there are difference in image sizes, or no extra files
len(rejected)

#0 = we are good

#Turn to dataframe

df = pd.DataFrame(img_sizes)

#Here is the statistics on the width of the images
df[0].describe()

#Here is the statistics on the width of the images
df[1].describe()

#have to decide what size of images these are going to be, have to standardize all of the shapes
#size is around 1320 x 968

##-------WIDTH--------
# count    5216.000000
# ###mean     1320.610813
# std       355.298743
# min       384.000000
# 25%      1056.000000
# 50%      1284.000000
# 75%      1552.000000
# max      2916.000000

#-------HEIGHT--------
# count    5216.000000
# ###mean      968.074770
# std       378.855691
# min       127.000000
# 25%       688.000000
# 50%       888.000000
# 75%      1187.750000
# max      2663.000000

#Transform

train_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(250),
    transforms.RandomHorizontalFlip(p = 0.3),
    transforms.RandomRotation(degrees= (-10, 10), resample = False, expand = False),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees = (0,0), translate = (0.05, 0.05)),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop(250),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


root = '../cats_dogs/'

train_data = datasets.ImageFolder(os.path.join(root, 'non_pad_MFCC_train'), transform = train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'non_pad_MFCC_test'), transform = test_transform)

torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

class_names = train_data.classes
class_names

# weight = torch.FloatTensor([3876/(1342+3876), 1342/(1342+3876)])

#grab first 10 images, plot them

for images, labels in train_loader:
    break

im = make_grid(images, nrow=5)
plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im.numpy(), (1,2,0)))

#############

class CNN(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(61*61*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
        self.fc4 = nn.Linear(10,2)

    def forward(self, X):
       X = F.relu(self.conv1(X))
       X = F.max_pool2d(X,2,2)
       X = F.relu(self.conv2(X))
       X = F.max_pool2d(X,2,2)
       X = X.view(-1, 61*61*16)
       X = F.relu(self.fc1(X))
       X = F.relu(self.fc2(X))
       X = F.relu(self.fc3(X))
       X = self.fc4(X)
       return F.log_softmax(X, dim = 1)

torch.manual_seed(101)


#Set up model

CNNmodel = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr = 0.001)

CNNmodel


#Training

import time

start_time = time.time()

epochs = 5

max_trn_batch = 800
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):

        #Limit the number of batches
        if b == max_trn_batch:
            break
        b += 1

        # Apply the model
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        # Print interim results

        print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/8000]  loss: {loss.item():10.8f}  \
        accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Limit the number of batches
            # if b == max_tst_batch:
            #     break

            # Apply the model
            y_val = CNNmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend();

print(test_correct)
print(f'Test accuracy: {test_correct[-1].item()*100/3000:.3f}%')

# Create a loader for the entire the test set
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = CNNmodel(X_test)
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()

arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize = (9,6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
plt.show();

print(classification_report(y_test.view(-1), predicted.view(-1)))