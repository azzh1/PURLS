import inspect
import torch
import importlib
from torch.nn import functional as F
import pytorch_lightning as pl

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.__dict__.update(kwargs['model_args'])
        self.save_hyperparameters()
        # breakpoint()
        
        # get model
        self.load_model()
        self.configure_optimizers()
        
        self.train_accumulate_count = 0
        self.train_acc_num = 0
        self.train_acc_tot = 0
        self.train_acc_prev = 0
        self.val_accumulate_count = 0
        self.val_acc_num = 0
        self.val_acc_tot = 0
        self.val_acc_prev = 0
        # get test accuracy recorder: total accuracy, number of samples
        self.test_stage = 0
        self.total_acc = 0.
        self.total_num = 0
        self.test_acc = {}
        for i in range(self.split):
            self.test_acc[i] = [0., 0]
        
    def log_loss(self, losses, type):
        self.log('{}-loss'.format(type), losses['loss'], 
                 prog_bar=True, logger=False, 
                 on_step=True if type == 'train' else False, 
                 on_epoch =False if type == 'train' else True)
        for key in losses:
            if isinstance(losses[key], list):
                for i in range(len(losses[key])):
                    self.logger.experiment.add_scalars('{}-loss'.format(type), 
                                                       {'{}-{}'.format(key, i): losses[key][i]},
                                                       self.global_step)
            else:
                self.logger.experiment.add_scalars('{}-loss'.format(type), 
                                                   {key: losses[key]},
                                                   self.global_step)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers(monitor1 = 'val-loss', 
                                               monitor2 = 'val-acc',
                                               lr = self.hparams.lr
                                               )
        
    def training_step(self, batch, batch_idx):
        vis_emb, target, target2 = batch # data sample, label, true label
            
        # move to correct device
        vis_emb = vis_emb.to(self.device)
        target = target.to(self.device)
        target2 = target2.to(self.device)
        # loss        
        losses,vals = self.model.loss(vis_emb, 
                                      target = target, 
                                      target2 = target2,
                                      cls_idx = self.seen_inds,
                                      curr_epoch = self.current_epoch,  # configs
                                      is_train=True) # training status
        # log losses
        self.log_loss(losses, 'train')
        # log accuracy
        if 'acc' in vals:
            if self.accumulate_grad_batches > 0:
                self.train_acc_num += vals['acc'][0]
                self.train_acc_tot += vals['acc'][1]
                if self.train_accumulate_count == self.accumulate_grad_batches:
                    self.train_accumulate_count = 0
                    self.train_acc_prev = self.train_acc_num / self.train_acc_tot
                    self.train_acc_num = 0
                    self.train_acc_tot = 0
                else:
                    self.train_accumulate_count += 1
                self.log('train-acc', self.train_acc_prev, prog_bar=True, logger=True, on_step=True, on_epoch =False)
            else:
                self.log('train-acc', vals['acc'], prog_bar=True, logger=True, on_step=True, on_epoch =False)
        
        torch.cuda.empty_cache()
        return losses['loss']
        
    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        vis_emb, target, target2 = batch # data sample, label, true label
        
        vis_emb = vis_emb.to(self.device)
        target = target.to(self.device)
        target2 = target2.to(self.device)
        # loss
        with torch.no_grad():
            losses, vals = self.model.loss(vis_emb, 
                                      target = target, 
                                      target2 = target2,
                                      cls_idx = self.seen_inds,
                                      curr_epoch = self.current_epoch,  # configs
                                      is_train=False,
                                      is_test=False) # training status
        # log losses
        self.log_loss(losses, 'val')
        # log accuracy
        if 'acc' in vals:
            if self.accumulate_grad_batches > 0:
                self.val_acc_num += vals['acc'][0]
                self.val_acc_tot += vals['acc'][1]
                if self.val_accumulate_count == self.accumulate_grad_batches:
                    self.val_accumulate_count = 0
                    self.val_acc_prev = self.val_acc_num / self.val_acc_tot
                    self.val_acc_num = 0
                    self.val_acc_tot = 0
                else:
                    self.val_accumulate_count += 1
                self.log('val-acc', self.val_acc_prev, prog_bar=True, logger=True, on_step=True, on_epoch =False)
            else:
                self.log('val-acc', vals['acc'], prog_bar=True, logger=True, on_step=True, on_epoch =False)
        # update best result
        torch.cuda.empty_cache()
        
    
    def _test_count_reset(self):
        # reset count
        self.total_acc = 0.
        self.total_num = 0
        self.test_acc = {}
    
    def zsl(self):
        # switched to zsl mode
        self._test_count_reset()
        self.test_stage = 0
        for i in range(self.split):
            self.test_acc[i] = [0., 0]
        
    def gzsl(self):
        # switched to gzsl mode
        self._test_count_reset()
        self.test_stage = 1
        for i in range(len(self.cls_labels)):
            self.test_acc[i] = [0., 0]
    
    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        vis_emb, target, target2 = batch # data sample, label, true label
        
        vis_emb = vis_emb.to(self.device)
        
        # check test stage and change
        target = target.to(self.device)
        target2 = target2.to(self.device)
        # loss
        with torch.no_grad():
            losses, vals = self.model.loss(vis_emb, 
                                      target = target, 
                                      target2 = target2,
                                      cls_idx = self.unseen_inds if self.test_stage == 0 else list(range(len(self.cls_labels))),
                                      curr_epoch = self.current_epoch,  # configs
                                      test_stage = self.test_stage,
                                      is_train=False,
                                      is_test=True) # training status
        # log losses
        self.log_loss(losses, 'test')
        
        # single class performance
        for i in vals['acc'][1:]:
            c, l, v = i
            self.test_acc[c][0] += v
            self.test_acc[c][1] += l
            self.total_acc += v
            self.total_num += l
        self.log('test-acc', self.total_acc / self.total_num, prog_bar=True, logger=True, on_step=False, on_epoch =True)    
        torch.cuda.empty_cache()
        # plot confusion matrix
        
    def load_model(self):
        # load main model
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        Model = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
        self.model = self.instancialize(Model)

    def instancialize(self, Model):
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys() 
        model_inkeys = self.hparams.model_args.keys()
        args1 = {}
        for arg in class_args: # update default arg under Minterface
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
            if arg in model_inkeys:
                args1[arg] = self.hparams.model_args[arg]
        return Model(**args1).to(self.device)
