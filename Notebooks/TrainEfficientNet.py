
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
#import pydicom
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, roc_curve, auc
from efficientnet_pytorch import EfficientNet


# In[3]:


# upload train dataframe
train_df_allsamples = pd.read_csv("../TrainTestDataFrames/train_concat.csv")
train_df_allsamples.head()


# In[5]:


# create dictionary that maps image name to target
image_names = train_df_allsamples["image_name"].values
targets = train_df_allsamples["target"].values
img_to_target = {image_name:target for image_name, target in zip(image_names, targets)}

percent_tp = sum(targets)/len(targets) * 100
print("{} training samples total.".format(len(targets)))
print("Only {:.3f} percent of training data set is a true positive.".format(percent_tp))
print("Therefore, the baseline accuracy is {:.3f}".format(np.max([percent_tp, 100-percent_tp])))


# In[7]:


# update so that the number of positives balances negatives
train_df_pos = train_df_allsamples.iloc[targets>0, :]
train_df_neg = train_df_allsamples.iloc[targets==0, :]
train_df_negsample = train_df_neg.sample(n=int(train_df_pos.shape[0]))

# concatenate negative and positive samples, then shuffle using .sample()
#train_val_df = pd.concat((train_df_pos, train_df_negsample)).sample(frac=1)
train_val_df = train_df_allsamples.sample(frac=1)

train_val_split = .95
n_train_val = train_val_df.shape[0]
n_train = int(train_val_split*n_train_val)

train_df = train_val_df[:n_train]
val_df = train_val_df[n_train:]

# create dictionary that maps image name to target
image_names = val_df["image_name"].values
val_targets = val_df["target"].values

percent_tp = sum(val_targets)/len(val_targets) * 100
baseline = np.max([percent_tp, 100-percent_tp])

print("{} Training and {} Validation samples".format(n_train, n_train_val-n_train))
print("{:.3f} percent of validation data set is a positive.".format(percent_tp))
print("Baseline validation accuracy is {:.3f}".format(baseline))


# In[8]:


# Device configuration (GPU can be enabled in settings)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)


# In[10]:


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, path_to_files):
        # 1. Initialize file paths or a list of file names.
        self.path = path_to_files
        self.df = df

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        # load X
        img_name = self.df['image_name'].values[index]
        img_path = self.path + img_name + ".jpg"
        img = plt.imread(img_path)

        # load y
        label = self.df["target"].values[index]
        target = torch.tensor(label, dtype=torch.float32)

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        #img = img.resize((256, 256))
        img_processed = transform(img)
        # 3. Return a data pair (e.g. image and label).
        return img_processed, target

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]


# In[12]:


# First, load the EfficientNet with pre-trained parameters
ENet = EfficientNet.from_pretrained('efficientnet-b0').to(device)


# In[13]:


# Hyper parameters
num_epochs = 5
batch_size = 42
learning_rate = 0.001

# Convolutional neural network
class MyENet(nn.Module):
    def __init__(self, Net):
        super(MyENet, self).__init__()
        self.Net = Net
        self.output = nn.Sequential(
            nn.Linear(1000, 1),
            nn.Sigmoid())

    def embedding(self, x):
        out = self.Net(x)
        return out

    def forward(self, x):
        out = self.Net(x)
        out = self.output(out)
        return out

model = MyENet(ENet).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[14]:


def make_weights_for_balanced_classes(df, nclasses=2):
    targets = df["target"].values
    count = [0] * nclasses
    for label in targets:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(targets)
    for idx, label in enumerate(targets):
        weight[idx] = weight_per_class[label]
    return weight


# In[15]:


# Train the model
# Use the prebuilt data loader.
path = "../../data/train/train/"

train_dataset = Dataset(train_df, path)
train_weights = make_weights_for_balanced_classes(train_df)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler = train_sampler)

# evaluate performance on validation data
valid_dataset = Dataset(val_df, path)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size)

# save losses from training
losses = []
val_roc = []
patience = 5
set_patience = 5
best_val = 0

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # set up model for training
        model = model.train()

        images = images.to(device)
        labels = torch.reshape(labels, [len(labels), 1])
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss
        losses.append(loss)

        # Calculate ROC
        predictions = outputs.detach().cpu().numpy().ravel()
        targets = labels.cpu().numpy().ravel()

        fpr, tpr, _ = roc_curve(np.array(targets, np.int), np.array(predictions).ravel())
        train_roc_auc = auc(fpr, tpr)

        # prep model for evaluation
        valid_predictions = []
        valid_targets = []
        model.eval()
        with torch.no_grad():
            for j, (images, labels) in enumerate(valid_loader):
                images = images.to(device)

                labels = torch.reshape(labels, [len(labels), 1])
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate val ROC
                valid_predictions += list(outputs.detach().cpu().numpy().ravel())
                valid_targets += list(labels.cpu().numpy().ravel())

        fpr, tpr, _ = roc_curve(np.array(valid_targets, np.int), np.array(valid_predictions).ravel())
        val_roc_auc = auc(fpr, tpr)
        val_roc.append(val_roc_auc)

        if val_roc_auc >= best_val:
            best_val = val_roc_auc
            patience = set_patience
            torch.save(model.state_dict(), '../Models/ENetmodel.ckpt')
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best validation roc_auc: {:.3f}'.format(best_val))
                break

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train ROC AUC: {:.4f}, Val ROC AUC: {:.4f}'
               .format(epoch+1, num_epochs, i+1, total_step, loss.item(), train_roc_auc, val_roc_auc))

# In[18]:


valid_predictions = []
valid_targets = []

valid_dataset = Dataset(val_df, path)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size)

model.eval() # prep model for evaluation
with torch.no_grad():
    for i, (images, labels) in enumerate(valid_loader):
        images = images.to(device)

        labels = torch.reshape(labels, [len(labels), 1])
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        valid_predictions += list(outputs.detach().cpu().numpy().ravel())
        valid_targets += list(labels.cpu().numpy().ravel())

fpr, tpr, _ = roc_curve(np.array(valid_targets, np.int), np.array(valid_predictions).ravel())
roc_auc = auc(fpr, tpr)

percent_tp = sum(valid_targets)/len(valid_targets) * 100
baseline = np.max([percent_tp, 100-percent_tp])
acc = np.sum(np.round(valid_predictions) == np.array(valid_targets)) / len(valid_targets)

print('\nBaseline classification accuracy: {:.2f}'.format(baseline))
print('\nModel classification accuracy:    {:.2f}'.format(acc))

plt.style.use('seaborn-colorblind')
plt.rcParams.update({'font.size': 16,
                     'legend.framealpha':1,
                     'legend.edgecolor':'inherit'})
plt.figure(figsize=(9, 6))

lw = 2
plt.plot(fpr, tpr,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("Figures/NN_ROC_on_val.png")
plt.close()

# In[21]:


# Save  the entire model.
train_df.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)
torch.save(model.state_dict(), '../Models/ENetmodel.ckpt')
