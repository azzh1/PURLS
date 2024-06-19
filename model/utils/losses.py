import torch
import torch.nn.functional as F

def contrastive_loss(m_skel, m_txt, m_img, tmp, target, reduction='mean'):
    '''
    * Copyright (c) 2023, salesforce.com, inc.
    * All rights reserved.
    * SPDX-License-Identifier: BSD-3-Clause
    * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
    * By Le Xue
    '''
    
    logits_per_skel_txt = tmp * (m_skel @ m_txt.t())
    logits_per_txt_skel = tmp * (m_txt @ m_skel.t())
    loss = (F.cross_entropy(logits_per_skel_txt, target, reduction=reduction) + \
            F.cross_entropy(logits_per_txt_skel, target, reduction=reduction))
    loss_len = 1
    if m_img != None:
        logits_per_skel_img = tmp * (m_skel @ m_img.t())
        logits_per_img_skel = tmp * (m_img @ m_skel.t())
        loss += (F.cross_entropy(logits_per_skel_img, target, reduction=reduction) + \
                F.cross_entropy(logits_per_img_skel, target, reduction=reduction))
        loss_len += 1
    
    # loss = loss / loss_len
    return loss, logits_per_skel_txt

def accuracy(input, target, class_emb, is_test = False, count_by_batch=False):
    # bs, 4, 512; 60, 4, 512
    expand_input = input.unsqueeze(1).expand([input.shape[0], class_emb.shape[0]] + 
                                                list(input.shape[1:]))
    expand_class_emb = class_emb.unsqueeze(0).expand([input.shape[0], class_emb.shape[0]] +
                                         list(class_emb.shape[1:])) # bs, cls, (4,) 512
    
    final_scores = torch.sum(expand_input*expand_class_emb, axis=-1) # bs, cls, (4)
    
    while len(final_scores.shape) > 2:
        final_scores = final_scores.mean(-1) #bs, cls
        
    final_scores = F.log_softmax(final_scores, 1)
    preds = torch.argmax(final_scores, axis = 1) # bs
    
    if target != None:
        acc_out = (preds == target)
        acc = torch.sum(acc_out).item()/len(acc_out)
        if is_test:
            # get class of targets
            classes = range(len(class_emb))
            cresults = []
            cresults.append(acc) # average accuracy for all classes
            for c in classes:
                # get indices of data when target == c
                cidx = (target == c).nonzero()
                curr_acc = torch.sum(acc_out[cidx]).item() # get sum accuracy for predicting target c
                cresults.append((c, len(cidx), curr_acc))
            return cresults, preds
        if count_by_batch:
            return (torch.sum(acc_out).item(), len(acc_out)), preds
        else:
            return acc, preds
    else:
        return None, preds