import torch
import torch.nn.functional as F
import numpy as np
import os

splits = {'ntu': [  
                [2,3,4,8,20], # head interpolate + 7
                [4,5,6,7,8,9,10,11,21,22,23,24], #hands
                [0,1,4,8,12,16,20], # torso  interpolate + 5
                [0,12,13,14,15,16,17,18,19] # feet  interpolate + 3
            ],
          'kinetic':[
              [16,14,15,17,0],
              [1,2,3,4,5,6,7],
              [0,1,2,5,8,11],
              [8,11,9,12,10,13]
          ]}

def joint_interpolate(x):
    x = torch.Tensor(x)
    x2 = F.interpolate(x, (72, 12), mode='bilinear')
    x2 =x2.numpy()
    return x2

def load_labels(root, split, dataloader, model_name):
    cls_num = int(dataloader.split('_')[-1])
    emb_dim = 512
    unseen_inds = np.load(os.path.join(root, 'resources/label_splits/r')+'u'+str(split)+'.npy')
    seen_inds = np.load(os.path.join(root, 'resources/label_splits/r')+'s'+str(cls_num - split)+'.npy')

    cls_labels = np.load(os.path.join(root, 'resources/ntu{}_bpnames.npy'.format(cls_num)),allow_pickle=True)

    return cls_num, emb_dim, unseen_inds, seen_inds, cls_labels