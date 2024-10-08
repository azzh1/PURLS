import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from data.utils import splits, joint_interpolate

backbone_locs = {'shift': 6, 'ctr': 9, 'dg': 8, 'aa': 8, 'c3d': 9}

class Ntu120(data.Dataset):
    def __init__(self, root2, root, dataset, bp_num, t_num, 
                 sample_type='train', data_type = 'common', backbone='shift'):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.cls_num = 120
        self.check_files()
    
    def load_y (self, y):
        return y[1] if len(y) == 2 else y

    def load_x(self, path):
        # pre-process
        if self.sample_type == 'seen_train':
            x = np.load(path + '/train_{}.npy'.format('feature' if self.data_type != 'pimg' else 'pimg'))
        elif self.sample_type == 'seen_val':
            x = np.load(path + '/val_{}.npy'.format('feature' if self.data_type != 'pimg' else 'pimg'))
        elif self.sample_type == 'zsl_val':
            x = np.load(path + '/test_{}.npy'.format('feature' if self.data_type != 'pimg' else 'pimg'))
        
        if self.data_type == 'common':
            x, nonzero_idx = self.preprocess(x)
            return x.mean(axis=(2,3)), nonzero_idx
        elif self.data_type == 'pimg':
            # N, t, p, 3
            x = np.transpose(x, (0, 3, 1, 2))
            return (x-np.min(x))/(np.max(x)-np.min(x)), []
        elif self.data_type in ['partition']:
            x, nonzero_idx = self.preprocess(x) # n, 256, 3, 25
            x1 = x.mean(axis=2) # joint level feature
            x1 = x1.transpose(0, 2, 1) # N, 25, 256
            x2 = x.mean(axis=3) # time level feature
            x2 = x2.transpose(0, 2, 1) # N, 3, 256
            x3 = np.expand_dims(x.mean(axis=(2,3)), axis=1) # N, 1, 256
            xs = []
            for i in range(self.bp_num): # bp
                xs.append(np.expand_dims(x1[:, splits['ntu'][i], :].mean(axis=1), axis=1))
            xs.append(x2)
            xs.append(x3)
            del(x)
            del(x1)
            del(x2)
            del(x3)
            return np.concatenate(xs, axis = 1), nonzero_idx # N, 7, 256
        elif self.data_type == 'full':
            out, nonzero_idx = self.preprocess(x)
            out = out.transpose(0, 2, 3, 1)
            return out, nonzero_idx
    
    # generate confusion matrix test indices
    def generate_cf_samples(self, cls, y, sample_num = 20):
        cf_samples = {}
        for i in range(len(cls)):
            cf_samples[cls[i]] = 0
        enough_samples = 0
        cf_index = []
        for i in range(len(y)):
            # return if enough
            if enough_samples == len(cls):
                return cf_index
            if cf_samples[int(y[i])] >= sample_num:
                enough_samples += 1
                continue
            else: # save data index
                cf_samples[int(y[i])] += 1
                cf_index.append(i)
        return cf_index
                
    def check_files(self):
        # print(self.backbone)
        # exit()
        # get zsl cls number
        path = self.root2 + '/' + self.dataset
        zsl_cls_num = int(path[path.find('shift_')+backbone_locs[self.backbone] : 
            path.find('_', path.find('shift_')+backbone_locs[self.backbone])])
        label_path = self.root2 + '/shift_{}_r'.format(zsl_cls_num)
        seen_cls_num = self.cls_num - zsl_cls_num
        # get learning samples
        if self.sample_type == 'seen_train':
            true_class_ids = np.load('{}/resources/label_splits/rs{}.npy'
                                     .format(self.root, seen_cls_num)).tolist()
            self.x, nonzero_idx = self.load_x(path)
            self.y = self.load_y(np.load(label_path + '/train_labels.npy'))[nonzero_idx]
            # self.i = np.transpose(np.load(path + '/train_pimg.npy'), (0,3,1,2))
        elif self.sample_type == 'seen_val':
            true_class_ids = np.load('{}/resources/label_splits/rs{}.npy'.format(self.root, seen_cls_num)).tolist()
            self.x, nonzero_idx = self.load_x(path)
            self.y = self.load_y(np.load(label_path + '/val_labels.npy'))[nonzero_idx]
            # self.i = np.transpose(np.load(path + '/val_pimg.npy'), (0,3,1,2))
        elif self.sample_type == 'zsl_val':
            true_class_ids = np.load('{}/resources/label_splits/ru{}.npy'.format(self.root, zsl_cls_num)).tolist()
            self.x, nonzero_idx = self.load_x(path)
            self.y = self.load_y(np.load(label_path + '/test_labels.npy'))[nonzero_idx]
            # self.i = np.transpose(np.load(path + '/test_pimg.npy'), (0,3,1,2))
            # select 5 samples for each class
            # self.cf_samples = self.generate_cf_samples(true_class_ids, self.x, self.y)
        
        # reset index
        self.y = self.y.astype(int)
        self.y2 = np.copy(self.y)
        self.y = np.array([true_class_ids.index(i) for i in self.y])
        
    def preprocess(self, x): # (N, 256, 3, 25)
        # process'
        x_in = x.transpose(0, 2, 3, 1) # N, 3, 25, 256
        N, C, P, D = x_in.shape
        
        # remove zero values
        idx_count = x_in.reshape(N, -1)
        nonzero_idx = np.where(np.any(idx_count != 0, axis=1))
        x_out = x_in[nonzero_idx]

        # normalize        
        x_out = x_out.reshape(-1, x_in.shape[-1]) # size * 4, dim
        
        x_out = x_out / np.tile((np.linalg.norm(x_out, axis=1).reshape((-1, 1))), 
                                (1, x_out.shape[-1]))

        out = x_out
        out = out.reshape(-1, C, P, D)
        out = out.transpose(0, 3, 1, 2)

        return out, nonzero_idx
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if hasattr(self, "i"):
            return [self.x[idx], self.y[idx], self.y2[idx], self.i[idx]]
        else:
            return [self.x[idx], self.y[idx], self.y2[idx]]