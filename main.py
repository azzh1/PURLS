import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import yaml
from model import MInterface
from data import DInterface
from data.utils import load_labels
from utils import str2bool, TBLogger
import sys
import numpy as np
import torch
import warnings
import shutil
warnings.filterwarnings("ignore")
import copy

def load_callbacks(args):
    callbacks = []
    """
    early stop callback
    """
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val-loss',
    #     mode='min',
    #     patience=10,
    #     min_delta=0.001
    # ))  
    
    """
    checkpoint callback
    """ 
    callbacks.append(plc.ModelCheckpoint(
        dirpath=args.work_dir,
        monitor='val-loss',  # use valid set's acc
        filename='best',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs = args.save_interval,
    ))

    return callbacks

def get_trainer(args):
    logger = TBLogger(save_dir = args.tb_folder, 
                        name=args.work_dir_name, 
                        default_hp_metric=False)
            
    args.callbacks = load_callbacks(args)
    args.logger =logger
    
    return Trainer(
                devices=args.gpus if args.gpus > 0 else None,
                accelerator = 'gpu' if args.gpus > 0 else 'cpu',
                strategy = 'ddp_find_unused_parameters_true', 
                callbacks = args.callbacks,
                logger = logger,
                max_epochs = args.num_epoch,
                check_val_every_n_epoch = args.eval_interval,
                num_sanity_val_steps=0,
            )

def main(args):
    hparams = copy.deepcopy(args)
    """
    pre-process arguments
    """
    hparams.tb_folder = '/'.join((hparams.work_dir).split('/')[:-1]) + '/tensorboard'
    hparams.work_dir_name = (hparams.work_dir).split('/')[-1]
    
    print("Current Experiment Configs: {}".format(hparams))
    """
    init environment
    """
    if hparams.activate_train: # start a new training
        if hparams.resume: # resume if work_dir exists, otherwise create
            if not os.path.exists(hparams.work_dir):
                print("Create working dir {}".format(hparams.work_dir))
                os.mkdir(hparams.work_dir)
                hparams.resume = False
            else:
                # check checkpoint file
                if not os.path.exists('{}/last.ckpt'.format(hparams.work_dir)):
                    hparams.resume = False
        else: # if work_dir exists, create new; otherwise directly create
            if os.path.exists(hparams.work_dir):
                shutil.rmtree(hparams.work_dir)
            print("Create working dir {}".format(hparams.work_dir))
            os.mkdir(hparams.work_dir)
    # constant seed
    pl.seed_everything(hparams.seed)
    
    """
    Pre-known dataset configs
    """
    hparams.num_class, hparams.emb_dim, hparams.unseen_inds, \
    hparams.seen_inds, hparams.cls_labels \
        = load_labels(hparams.root, hparams.split, hparams.dataloader, hparams.model_name)

    hparams.seen_labels = [hparams.cls_labels[i] for i in hparams.seen_inds]
    hparams.unseen_labels = [hparams.cls_labels[i] for i in hparams.unseen_inds]
    hparams.bp_num = 4
    hparams.t_num = 3
    
    """
    data module
    """
    print("Load data module.")
    print(hparams.backbone)
    data_module = DInterface(**vars(hparams))
    data_module.setup()
    
    """
    model module
    """
    print("Load model module.")
    model = MInterface(**vars(hparams))
    
    """
    Processor
    """
    trainer = get_trainer(hparams)
    # zsl training
    if hparams.activate_train:
        trainer.fit(model, data_module.seen_train_dataloader(), 
                        data_module.seen_val_dataloader(),
                        ckpt_path = '{}/last.ckpt'.format(hparams.work_dir) 
                                    if hparams.resume else None)
    # unseen test - pure zsl
    model.zsl()
    trainer.test(model, data_module.zsl_test_dataloader(), 
                ckpt_path = '{}/best.ckpt'.format(hparams.work_dir))
    for i in model.test_acc:
        acc = np.sum(np.array(model.test_acc[i][0]))
        acclen = np.sum(np.array(model.test_acc[i][1]))
        model.test_acc[i] = acc / acclen
    print(model.test_acc)

if __name__ == '__main__':
    parser = ArgumentParser()
    """
    Basic arguments
    """
    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    """
    Processor
    """
    parser.add_argument('--gpus', type=str2bool, default=-1, help='use GPUs or not')
    parser.add_argument('--num_epoch', type=int, help='stop training in which epoch')

    """
    Visulize and debug
    """
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')

    """
    Data
    """
    parser.add_argument('--root', help='root repo to load data')
    parser.add_argument('--root2', help='feature data path')
    parser.add_argument('--dataset', help='type of dataset: shift_5_r')
    parser.add_argument('--dataloader', help='class of the dataloader: ntu60')
    parser.add_argument('--data_type', help='how to process the input skeleton data')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('-ss', '--split', type=int, help='Which split to use: 5 or 12')
    parser.add_argument('-b', '--backbone', default='shift', help='encoder backbone')
    

    """
    Model
    """
    # configs
    parser.add_argument('--model_name', help='select the training config for a specific model')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_args', default=dict(), help='model configs')
    
    """
    Ablation
    """
    parser.add_argument('--activate_train', type=str2bool, default=False, help='activate training process')
    parser.add_argument('--test_p', default=False, type=str2bool, help='activate hyperparameter testing')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume training')
    parser.add_argument('--accumulate_grad_batches', default=0, help='accumulate grad batches')
    
    
    """
    Main process
    """    
    # Reset Some Default Trainer Arguments' Default Values
    p = parser.parse_args(sys.argv[1:])
    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            input_args = yaml.load(f, Loader=yaml.FullLoader)

        # update parser from config file
        key = vars(p).keys()
        for k in input_args.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key

        parser.set_defaults(**input_args) # assign arg values
    args = parser.parse_args() # update args with hand input
    
    if not args.test_p:
        """
        run
        """
        args.prompt_args['prompt_depth'] = 3
        main(args)
    else:
        """
        do parameter tests
        """
        pass