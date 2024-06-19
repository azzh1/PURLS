import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        
class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        
    def configure_optimizers(self, monitor1, monitor2, lr=1e-4, **kwargs):
        """
        Monitor1: Validation Loss,
        Monitor2: Validation Accuracy
        """
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=25, cooldown=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler, # Changed scheduler to lr_scheduler
                "monitor": monitor2,
                "mode": "max"
            }
        }
        