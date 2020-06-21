
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


# In[3]:


alldata = pd.read_csv("../TrainTestDataFrames/marking.csv")
test_df = pd.read_csv("../TrainTestDataFrames/test.csv")
path = "../../data-512/512x512-test/512x512-test/"


# In[4]:


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


# In[5]:


test_df.head()


# In[6]:


meta_features = ['sex', 'age_approx', 'anatom_site_general_challenge']

encoder = {}
for feature in meta_features:
    # determine unique features
    categories = np.unique(np.array(alldata[feature].values, str))
    for i, category in enumerate(categories):
        if category != 'nan':
            encoder[category] = np.float(i)
encoder['nan'] = np.nan

# no flip or rotation for test/validation data
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

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

        # determine meta data
        meta = self.df[meta_features].values[index]
        meta_data = np.array([encoder[str(m)] for m in meta])

        # load y
        #label = self.df["target"].values[index]
        #target = torch.tensor(label, dtype=torch.float32)

        # 2. Preprocess the data (e.g. torchvision.Transform).
        img = Image.fromarray(img)
        #img = img.resize((256, 256))
        img_processed = transform(img)
        # 3. Return a data pair (e.g. image and label).
        return img_processed, meta_data, img_name

    def __len__(self):
        # total size of your dataset.
        return self.df.shape[0]


# In[7]:


test_dataset = Dataset(test_df, path)

batch_size = 8
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = model.eval()
for i, (images, meta_data, batch_image_names) in enumerate(tqdm(test_loader)):


    images = images.to(device)

    # Forward pass
    embed = model.embedding(images)
    nn_pred = model.output(embed).detach().cpu().numpy()
    embedding = embed.detach().cpu().numpy()

    # determine NN features for the set of images
    batch_features = np.concatenate((embedding, meta_data.numpy(), nn_pred), axis=1)

    # append the dataset
    try:
        X = np.concatenate((X, batch_features), 0)
        image_names = np.append(image_names, batch_image_names)
    except:
        X = batch_features
        image_names = np.array(batch_image_names)


# In[8]:


bst = xgb.Booster({'nthread': 8})  # init model
bst.load_model('../Models/xgb_ENET_all.model')  # load data


# In[9]:


Dtest = xgb.DMatrix(X)
predictions = bst.predict(Dtest)


# In[20]:


submission = pd.DataFrame()

submission["image_name"] = image_names
submission["target"] = predictions


# In[21]:


submission.to_csv("JT_submission_XGB.csv", index=False)


# In[22]:

'''
submission.head(10)


# In[23]:


plt.style.use('seaborn-colorblind')
plt.rcParams.update({'font.size': 16,
                     'legend.framealpha':1,
                     'legend.edgecolor':'inherit'})
plt.figure(figsize=(9, 6))

plt.hist(predictions, label='XGBoost')
plt.hist(X[:, -1], label='CNN', alpha=.5)
plt.hist(X[:, -1]*(1/2) + predictions*(1/2), label='XGB CNN', alpha=.5)
plt.legend()
plt.show()
'''
