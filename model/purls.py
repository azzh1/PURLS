import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import clip
from model import Base, init_weights
import numpy as np
from model.utils.losses import contrastive_loss, accuracy
from model.utils.attention import MultiHeadSelfAttention
from model.utils.layers import HiddenLayer
from model.utils.contrastive_learning import get_rank

local_names = ['head', 'hands', 'torso', 'legs', 'start', 'middle', 'end', 'global']

class Purls(Base):
    def __init__(self, num_epoch, input_size=256, 
                 emb_dim = 512, bp_num=4, t_num=3, 
                 proj_hidden_size=1024, n_hidden_layers=1, 
                 n_head=2, k_dim=150, local_type = 'full', cls_labels=None,
                 use_d = False):
        """
        Argument explanations:
        num_epoch: total number of running epochs
        input_size: input skeleton feature size
        emb_dim: text embedding size for descriptions
        bp_num: number of body-part-based representations
        t_num: number of temporal-interval-based representations
        proj_hidden_size: hidden layer output size for skel->txt projection
        n_hidden_layers: number of hidden layers for skel->txt projection
        n_head: number of head for attention module
        k_dim: transformed feature size for attention module
        local_type: types of local/global representation matching ('bp', 'tp', 'g')
        cls_labels: (extended) label names of each action (or its part-based/temporal-based local descriptions)
        use_d: activate weighted learning for each contrastive loss
        """
        
        init_dict = locals().copy()
        init_dict.pop('self')
        super().__init__(**init_dict)
        """
        configs
        """
        self.loss_func = contrastive_loss
        """
        model
        """
        # feature skel extractors for glob & local
        if self.local_type == 'bp': # only learn body-part-based local representations
            encode_input_size = self.input_size * (self.bp_num + 1) # 2048
            encode_output_size = self.emb_dim * (self.bp_num + 1) # 4096
            self.tmps = nn.Parameter(torch.ones([self.bp_num + 1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(self.bp_num + 1,device='cuda'), requires_grad=True)
            
        elif self.local_type == 'tp': # only learn temporal-based local representations
            encode_input_size = self.input_size * (self.t_num + 1) # 2048
            encode_output_size = self.emb_dim * (self.t_num + 1) # 4096
            self.tmps = nn.Parameter(torch.ones([self.t_num + 1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(self.t_num + 1,device='cuda'), requires_grad=True)
            
        elif self.local_type == 'g': # only learn global-descriptiion-based representations
            encode_input_size = self.input_size # 2048
            encode_output_size = self.emb_dim # 4096
            self.tmps = nn.Parameter(torch.ones([1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(1,device='cuda'), requires_grad=True)
            
        else: # learn global-description, body-part-based, temporal-based local representations
            encode_input_size = self.input_size * (self.bp_num + self.t_num + 1) # 2048
            encode_output_size = self.emb_dim * (self.bp_num + self.t_num + 1) # 4096
            self.tmps = nn.Parameter(torch.ones([self.bp_num + self.t_num + 1], dtype=torch.float, device='cuda', requires_grad=True) * np.log(1 / 0.07))
            if self.use_d:
                self.d = nn.Parameter(torch.ones(self.bp_num + self.t_num + 1,device='cuda'), requires_grad=True)
        
        # skeleton representation projection
        self.attention = MultiHeadSelfAttention(self.input_size, self.emb_dim, k_dim, n_head)
        self.skel_encoder = nn.Sequential(
                                        nn.BatchNorm1d(encode_input_size), 
                                        nn.Dropout(.5),
                                        nn.Linear(encode_input_size, self.proj_hidden_size),
                                        nn.SiLU(),
                                        HiddenLayer(self.n_hidden_layers, self.proj_hidden_size),
                                        nn.BatchNorm1d(self.proj_hidden_size),
                                        nn.Dropout(.5),
                                        nn.Linear(self.proj_hidden_size, encode_output_size),
                                        nn.Tanh()
                                    )
        # description encoding
        self.cls_tokens = [clip.tokenize(self.cls_labels[:, i]) for i in range(bp_num + t_num + 1)]
        self.clip, _ = clip.load("ViT-B/32")
        # initialization
        self.attention.apply(init_weights)
        self.skel_encoder.apply(init_weights)
        for name, param in self.named_parameters():
            # if name_to_update not in name:
            if "clip" in name:
                param.requires_grad_(False)
        
    def configure_optimizers(self, monitor1, monitor2, lr=1e-3):
        params = []
        params += self.skel_encoder.parameters()
        params += [self.tmps]
        if self.use_d:
            params += [self.d]
        
        params += self.attention.parameters()
        optimizer = optim.Adam(params, lr=lr)
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, 
                                        cooldown=3, verbose=True)
        scheduler = {'scheduler': scheduler,
                    'monitor': monitor1,
                    'mode': 'min'
                    }
        return [optimizer], [scheduler]
    
    def loss(self, vis_emb, target, target2, # constant variables
             cls_idx, curr_epoch, is_train=False, 
             is_test=False, **params):
        """
        Argument explanations:
        vis_emb: Pre-trained skeleton features of the batch examples from backbones
        target: Label index for the batch examples (re-labeled according to available candidate classes)
        target2: Original label index for the batch examples (i.e. not yet re-labeled)
        cls_idx: All class index for the current task (e.g. For a NTU 55/5 split, cls_idx = [10, 11, 19, 26, 56])
        curr_epoch: Current epoch no.
        is_train: Whether the function is used during the training stage 
        is_test: Whether the function is used during the testing stage. 
            is_train = True -> Training Stage (on seen classes)
            is_train = False, is_test = False -> Validation Stage (on seen classes)
            is_train = False, is_test = True -> Testing Stage (on unseen classes)
        """
        # prepare one-hot contrastive learning labels
        batch_labels = len(vis_emb) * get_rank() + torch.arange(
                                                    len(vis_emb), device= vis_emb.device
                                                )
        
        # text embedding
        text_features = []
        for i in range(self.bp_num + self.t_num + 1):
            
            curr_text_features = self.clip.encode_text(self.cls_tokens[i].to(vis_emb.device)).float()
            curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
            text_features.append(curr_text_features.unsqueeze(0))
        text_features = torch.cat(text_features, dim=0)
        text_features = text_features.permute(1, 0, 2).contiguous() # cls, bp, dim
        class_text_features = text_features[cls_idx]
        bp_emb = class_text_features[target]
        
        # get body part adaptive features
        bs, t, j, d = vis_emb.shape
        vis_emb = vis_emb.view(bs, -1, d) # bs, tp, d
        local_vis_emb = self.attention(vis_emb, bp_emb)
        local_bp_emb = bp_emb
        total_vis_emb = local_vis_emb
        
        # calculate loss
        loss = 0.
        bs, p, d = total_vis_emb.shape
        total_vis_emb = total_vis_emb.view(bs, -1).to(total_vis_emb.device)
        output_emb = self.skel_encoder(total_vis_emb)
        output_emb = output_emb.view(bs, p, -1).to(total_vis_emb.device) # bs, p, 512

        # bs, t, j, d = skel_emb.shape # shape: batchsize, convoluted temporal dim, joint dim, feature dim
        # skel_emb = skel_emb.view(bs, -1, d) # bs, tp, d
        # if self.local_type == 'bp':
        #     local_skel_emb = self.attention(torch.cat((skel_emb[:, :self.bp_num,:], skel_emb[:, -1,:].unsqueeze(1)), dim=1), 
        #                                    torch.cat((desc_emb[:, :self.bp_num,:], desc_emb[:, -1, :].unsqueeze(1)),dim=1))
        #     local_desc_emb = torch.cat((desc_emb[:, :self.bp_num,:], desc_emb[:, -1, :].unsqueeze(1)),dim=1)
        # elif self.local_type == 'tp':
        #     local_skel_emb = self.attention(torch.cat((skel_emb[:, self.bp_num : self.bp_num + self.t_num,:], skel_emb[:, -1,:].unsqueeze(1)), dim=1), 
        #                                    torch.cat((desc_emb[:, self.bp_num : self.bp_num + self.t_num,:], desc_emb[:, -1, :].unsqueeze(1)),dim=1))
        #     local_desc_emb = torch.cat((desc_emb[:, self.bp_num : self.bp_num + self.t_num,:], desc_emb[:, -1, :].unsqueeze(1)),dim=1)
        # elif self.local_type == 'g':
        #     local_skel_emb = self.attention(skel_emb[:, -1,:].unsqueeze(1), desc_emb[:, -1,:].unsqueeze(1))
        #     local_desc_emb = desc_emb[:, -1,:].unsqueeze(1)
        # else:
        #     local_skel_emb = self.attention(skel_emb, desc_emb)
        #     local_desc_emb = desc_emb
        # total_skel_emb = local_skel_emb
        
        # each representation contrastive learning loss (include global)
        for i in range(output_emb.shape[1]):
            curr_output_emb = output_emb[:, i, :]
            curr_att_emb = local_bp_emb[:, i, :]
            curr_loss, _ =  self.loss_func(curr_output_emb, curr_att_emb, None, 
                                                    self.tmps[i], batch_labels) #, reduction='none')
            if self.use_d:
                loss += curr_loss * self.d[i]
            else:
                loss += curr_loss


        # glob_loss, _ = self.loss_func(proj_emb[:, -1, :], local_desc_emb[:, -1, :], img_emb, self.tmps[-1], batch_labels) #, reduction='none')
        # loss += (glob_loss * self.d(desc_emb[:, -1, :]).view(-1)).sum() / len(glob_loss)
        
        # if self.local_type in ['g', 'full']:
        #     if self.use_d:
        #         loss += glob_loss * self.d[-1]
        #     else:
        #         loss += glob_loss
            
        # add regularization or not
            
        # else:
        #     global_loss, logits_per_skel_txt = self.loss_func(proj_emb[:, -1, :], att_emb, None, self.tmps[-1], batch_labels)
        #     loss = global_loss
        global_emb = output_emb[:, -1, :]
        acc, preds = accuracy(global_emb, target, class_text_features[:, -1, :], is_test, False)
        # else:
            # acc, preds = accuracy(global_emb, target, class_emb, None, is_test)
        return {'loss': loss}, \
                {'acc': acc, 'preds': preds}
    