#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:56:14 2019

Extract person features from all subfolder in Sample folder (except ALL folder).
To make features set for recognition or train SVM.

@author: gjz
"""
import os
import numpy as np
import pickle
import yaml

from PIL import Image
from model import PCB, torch
from torchvision import transforms
from torch.autograd import Variable

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


OUTPUT_NAME = 'sample_set1'

# Features extraction model file.
CONFIG_FILE = 'model/PCB01.yaml'
WEIGHT_FILE = 'model/PCB01.pth'

# SVM parameter.
svm = SVC(kernel='linear', C=5, probability=True)

# Images sample folder.
SAMPLE_FOLDER = './Sample'


def clusterPlot(x, y, title='', save=None):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from numpy import arange

    plt.figure(figsize=[8, 6])
    t = arange(len(set(labels))+1)
    if x.shape[1] > 2:
        x_embedded = TSNE(n_components=2).fit_transform(x)
    else:
        x_embedded = x
    for i, t in enumerate(set(y)):
        idx = y == t
        plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], label=t)

    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1))
    if save is not None:
        plt.savefig(save)


with open(CONFIG_FILE, 'r') as stream:
    config = yaml.load(stream)
nclasses = config['nclasses']
model = PCB(nclasses)
model.load_state_dict(torch.load(WEIGHT_FILE))
model = model.eval().cuda()

features = []
labels = []

fol_list = os.listdir(SAMPLE_FOLDER)
for folder in fol_list:
    if folder == 'ALL':
        continue

    fol = os.path.join(SAMPLE_FOLDER, folder)
    if not os.path.isdir(fol):
        continue
    
    print('Folder : ' + folder)
    cls_name = folder
    fil_list = os.listdir(fol)
    for fil in fil_list:
        img = Image.open(os.path.join(fol, fil))
        res_img = transforms.functional.resize(img, (384, 192), interpolation=3)
        res_img = transforms.functional.to_tensor(res_img)
        res_img = transforms.functional.normalize(res_img, [0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
        res_img = Variable(res_img.cuda())

        out = model.predict(res_img.unsqueeze(0))
        out = [o.data.cpu().numpy() for o in out]
        out = np.array(out).transpose(1, 2, 0)
        out = torch.FloatTensor(out)

        fnorm = torch.norm(out, p=2, dim=1, keepdim=True) * np.sqrt(6)
        feature = out.div(fnorm.expand_as(out))
        feature = feature.reshape(feature.size(0), -1)[0]
        feature = feature.numpy()
        
        features.append(feature)
        labels.append(cls_name)
        
cls_names = sorted(list(set(labels)))
SAVE_FILE = os.path.join(SAMPLE_FOLDER, OUTPUT_NAME)
pickle.dump((features, labels, cls_names), open(SAVE_FILE + '.pkl', 'wb'))

print('Cluster Plot...')
clusterPlot(features, labels, OUTPUT_NAME, SAVE_FILE + '.png')

print('Train SVM...')
lb = LabelEncoder().fit(cls_names)
x_train = np.array(features)
y_train = lb.transform(labels)

svm.fit(x_train, y_train)

SVM_FILE = os.path.join('Model', OUTPUT_NAME + '-svm.pkl')
pickle.dump(svm, open(SVM_FILE, 'wb'))
