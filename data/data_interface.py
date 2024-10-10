import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split, ConcatDataset
        
class DInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.load_data_module()

    def divide_by_proportions(self, proportions, dataset):
        lengths = [int(p * len(dataset)) for p in proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])
        return lengths

    def setup(self, stage=None):
        # seen class train
        self.seen_train = self.instancialize(sample_type = 'seen_train') # seen class train
        self.seen_val =  self.instancialize(sample_type = 'seen_val') 
        self.zsl_test = self.instancialize(sample_type = 'zsl_val') 
        self.gzsl_test = ConcatDataset([self.seen_val, self.zsl_test])
        
    # training dataloaders
    def seen_train_dataloader(self):
        return DataLoader(self.seen_train, pin_memory=True, batch_size=self.batch_size, shuffle=True)
            
    # validate dataloaders
    def seen_val_dataloader(self):
        return DataLoader(self.seen_val, pin_memory=True, batch_size=self.batch_size, shuffle=True)
    
    def zsl_test_dataloader(self):
        return DataLoader(self.zsl_test, pin_memory=True, batch_size=self.batch_size, shuffle=True)
    
    def gzsl_test_dataloader(self):
        return DataLoader(self.gzsl_test, pin_memory=True, batch_size=self.batch_size, shuffle=True)
    
    def load_data_module(self):
        # decide training type
        name = self.dataloader
        # Change the `snake_case.py` file name to `CamelCase` class name.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        self.data_module = getattr(importlib.import_module(
            '.'+name, package=__package__), camel_name)
    
    def instancialize(self, **other_args):
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.__dict__.keys()
        args1 = {}
        for arg in class_args: # update default arg under Dinterface
            if arg in inkeys:
                args1[arg] = self.__dict__[arg]
        args1.update(other_args) # update specialized arg from functional calls
        return self.data_module(**args1)