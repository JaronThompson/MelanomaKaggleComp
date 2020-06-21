
# coding: utf-8

# In[1]:


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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import xgboost as xgb
from efficientnet_pytorch import EfficientNet

from tqdm import tqdm


# In[2]:


# Device configuration (GPU can be enabled in settings)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)


# In[4]:


train_df = pd.read_csv("ENET_train_df_all.csv")
val_df = pd.read_csv("ENET_val_df_all.csv")
path = "../../data-512/512x512-dataset-melanoma/512x512-dataset-melanoma/"

print("Training on {} images, validating on {} images.".format(train_df.shape[0], val_df.shape[0]))


# In[5]:


# First, load the EfficientNet with pre-trained parameters
ENet = EfficientNet.from_pretrained('efficientnet-b0').to(device)

# Convolutional neural network
class MyENet(nn.Module):
    def __init__(self, ENet):
        super(MyENet, self).__init__()
        # modify output layer of the pre-trained ENet
        self.ENet = ENet
        num_ftrs = self.ENet._fc.in_features
        self.ENet._fc = nn.Linear(in_features=num_ftrs, out_features=256, bias=True)
        # map Enet output to melanoma decision
        self.output = nn.Sequential(nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(256, 1),
                                    nn.Sigmoid())

    def embedding(self, x):
        out = self.ENet(x)
        return out

    def forward(self, x):
        out = self.ENet(x)
        out = self.output(out)
        return out

model = MyENet(ENet).to(device)
# Load best model
model.load_state_dict(torch.load('../Models/ENETmodel_all.ckpt'))


# In[6]:


meta_features = ['sex', 'age_approx', 'anatom_site_general_challenge']

encoder = {}
for feature in meta_features:
    # determine unique features
    categories = np.unique(np.array(train_df[feature].values, str))
    for i, category in enumerate(categories):
        if category != 'nan':
            encoder[category] = np.float(i)
encoder['nan'] = np.nan

# no flip or rotation for test/validation data
transform_valid = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, df, path_to_files):
        # 1. Initialize file paths or a list of file names.
        self.path = path_to_files
        self.df = df

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        # load X
        img_name = self.df['image_id'].values[index]
        img_path = self.path + img_name + ".jpg"
        img = plt.imread(img_path)

        # determine meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # load y
        label = self.df["target"].values[index]
        target = torch.tensor(label, dtype=torch.float32)

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        #img = img.resize((256, 256))
        img_processed = transform_valid(img)

        # 3. get meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, target

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]


# In[7]:


# Use the data loader.
'''
batch_size = 12
path = "../../data-512/512x512-dataset-melanoma/512x512-dataset-melanoma/"

train_dataset = ValidDataset(train_df, path)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size)

model = model.eval()
for i, (images, meta_data, labels) in enumerate(tqdm(train_loader)):
    images = images.to(device)

    # Forward pass
    embed = model.embedding(images)
    nn_pred = model.output(embed).detach().cpu().numpy()
    embedding = embed.detach().cpu().numpy()

    # determine NN features for the set of images
    batch_features = np.concatenate((embedding, meta_data, nn_pred), axis=1)

    # append the dataset
    try:
        X = np.concatenate((X, batch_features), 0)
        y = np.append(y, labels.numpy())
    except:
        X = batch_features
        y = labels.numpy()


# In[8]:


# Save X and y in pandas dataframe
XGB_data = pd.DataFrame(data=X)
XGB_data['targets'] = y
XGB_data.to_csv("XGB_ENET_train_all.csv", index=False)
'''
XGB_data = pd.read_csv("XGB_ENET_train_all.csv")

X = np.array(XGB_data.values[:, :-1], np.float32)
y = np.array(XGB_data['targets'].values, np.float32)


# In[9]:


# weight positive examples more heavily
def make_weights(targets, nclasses=2):
    count = [0] * nclasses
    for label in targets:
        count[np.int(label)] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(targets)
    for idx, label in enumerate(targets):
        weight[idx] = weight_per_class[np.int(label)]

    return np.array(weight)

XGB_data = pd.read_csv("XGB_ENET_val_all.csv")
Xval = np.array(XGB_data.values[:, :-1], np.float32)
yval = np.array(XGB_data['targets'].values, np.float32)

dtrain = xgb.DMatrix(X, label=y) #, weight=w)
dval = xgb.DMatrix(Xval, label=yval)


# booster params
param = {'max_depth': 16, 'eta': .05, 'objective': 'binary:logistic'}
param['nthread'] = 8
param['eval_metric'] = 'auc'
#param['subsample'] = .5
#param['gpu_id'] = 0
#param['tree_method'] = 'gpu_hist'

# specify validation set
evallist = [(dval, 'eval')]

# Training
num_round = 5000
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=15)


# In[11]:


bst.save_model("../Models/xgb_ENET_all.model")
#bst = xgb.Booster({'nthread': 8})  # init model
#bst.load_model('../Models/xgb_ENET_all.model')  # load data


# In[12]:


# prediction
ypred = bst.predict(dval)

fpr, tpr, _ = roc_curve(yval, ypred)
roc_auc = auc(fpr, tpr)

# determine NN features for the set of image
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
#plt.show()
plt.savefig("Figures/XGB_AUC_val.png")
plt.close()

# In[15]:


val_acc = accuracy_score(yval, np.round(ypred))
print("validation accuracy: {:.3f}".format(val_acc))


# In[16]:


tn, fp, fn, tp = confusion_matrix(yval, np.round(Xval[:, -1])).ravel()

accuracy = (tp + tn) / len(yval)
# precision is the fraction of correctly identified positive samples
# precision asks:"Of all the samples identified as positives, how many were correct?"
precision = tp / (tp + fp)
# recall is the ability of the model to identify positive samples
# recall asks:"Of all the positive samples in the dataset, how many were identified by the model?"
recall = tp / (tp + fn)

print("CNN Stats:")

print("Model accuracy: {:.2f}".format(accuracy))
print("Model precision: {:.2f}".format(precision))
print("Model recall: {:.2f}".format(recall))

print("\nConfusion Matrix: ")
print(confusion_matrix(yval, np.round(Xval[:, -1])))

tn, fp, fn, tp = confusion_matrix(yval, np.round(ypred)).ravel()

accuracy = (tp + tn) / len(yval)
# precision is the fraction of correctly identified positive samples
# precision asks:"Of all the samples identified as positives, how many were correct?"
precision = tp / (tp + fp)
# recall is the ability of the model to identify positive samples
# recall asks:"Of all the positive samples in the dataset, how many were identified by the model?"
recall = tp / (tp + fn)

print("\nXGBoost Stats:")

print("Model accuracy: {:.2f}".format(accuracy))
print("Model precision: {:.2f}".format(precision))
print("Model recall: {:.2f}".format(recall))

print("\nConfusion Matrix: ")
print(confusion_matrix(yval, np.round(ypred)))

tn, fp, fn, tp = confusion_matrix(yval, np.round(Xval[:, -1]*(2/4) + ypred*(2/4))).ravel()

accuracy = (tp + tn) / len(yval)
# precision is the fraction of correctly identified positive samples
# precision asks:"Of all the samples identified as positives, how many were correct?"
precision = tp / (tp + fp)
# recall is the ability of the model to identify positive samples
# recall asks:"Of all the positive samples in the dataset, how many were identified by the model?"
recall = tp / (tp + fn)

print("\nEnsemble Stats:")

print("Model accuracy: {:.2f}".format(accuracy))
print("Model precision: {:.2f}".format(precision))
print("Model recall: {:.2f}".format(recall))

print("\nConfusion Matrix: ")
print(confusion_matrix(yval, np.round(Xval[:, -1]*(2/4) + ypred*(2/4))))


# In[17]:

'''
bst_feature_dict = bst.get_score(importance_type='gain')
feature_importance = np.argsort(list(bst_feature_dict.values()))[::-1]
top_10 = np.array(list(bst_feature_dict.keys()))[feature_importance][:10]
top_10


# In[19]:


plt.style.use('seaborn-colorblind')
plt.rcParams.update({'font.size': 16,
                     'legend.framealpha':1,
                     'legend.edgecolor':'inherit'})
plt.figure(figsize=(9, 6))

plt.hist(ypred, label='XGBoost')
plt.hist(Xval[:, -1], label='CNN', alpha=.5)
plt.legend()
plt.show()


# In[20]:


# Next try "test time augmentation"

#
transform_TTA = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


class TTADataset(torch.utils.data.Dataset):
    def __init__(self, df, path_to_files):
        # 1. Initialize file paths or a list of file names.
        self.path = path_to_files
        self.df = df

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        # load X
        img_name = self.df['image_id'].values[index]
        img_path = self.path + img_name + ".jpg"
        img = plt.imread(img_path)

        # determine meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # load y
        label = self.df["target"].values[index]
        target = torch.tensor(label, dtype=torch.float32)

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        #img = img.resize((256, 256))
        img_processed = transform_TTA(img)

        # 3. get meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, target

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]

N_TTA = 5
batch_size = 1
valid_dataset = TTADataset(val_df, path)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size)

ypred_TTA = np.zeros((len(ypred), N_TTA))

model = model.eval()
for j in range(N_TTA):
    for i, (images, meta_data, labels) in enumerate(tqdm(valid_loader)):
        images = images.to(device)

        # Forward pass
        embed = model.embedding(images)
        nn_pred = model.output(embed).detach().cpu().numpy()
        embedding = embed.detach().cpu().numpy()

        # determine NN features for the set of images
        batch_features = np.concatenate((embedding, meta_data.numpy(), nn_pred), axis=1)

        ypred_TTA[i, j] = bst.predict(xgb.DMatrix(batch_features))


# In[43]:


ypred_TTA_mean = np.mean(ypred_TTA, 1)


# In[44]:


fpr, tpr, _ = roc_curve(yval, ypred_TTA_mean)
roc_auc = auc(fpr, tpr)

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
plt.show()


# In[45]:


tn, fp, fn, tp = confusion_matrix(yval, np.round(ypred_TTA_mean)).ravel()

accuracy = (tp + tn) / len(yval)
# precision is the fraction of correctly identified positive samples
# precision asks:"Of all the samples identified as positives, how many were correct?"
precision = tp / (tp + fp)
# recall is the ability of the model to identify positive samples
# recall asks:"Of all the positive samples in the dataset, how many were identified by the model?"
recall = tp / (tp + fn)

print("\nTTA Stats:")

print("Model accuracy: {:.2f}".format(accuracy))
print("Model precision: {:.2f}".format(precision))
print("Model recall: {:.2f}".format(recall))

tn, fp, fn, tp = confusion_matrix(yval, np.round(ypred)).ravel()

accuracy = (tp + tn) / len(yval)
# precision is the fraction of correctly identified positive samples
# precision asks:"Of all the samples identified as positives, how many were correct?"
precision = tp / (tp + fp)
# recall is the ability of the model to identify positive samples
# recall asks:"Of all the positive samples in the dataset, how many were identified by the model?"
recall = tp / (tp + fn)

print("\nXGBoost Stats:")

print("Model accuracy: {:.2f}".format(accuracy))
print("Model precision: {:.2f}".format(precision))
print("Model recall: {:.2f}".format(recall))
'''
