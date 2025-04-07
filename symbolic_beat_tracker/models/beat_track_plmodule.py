# Remove tempo estimation.
# Change to feed beat into downbeat GRU.

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import ReduceLROnPlateau

from models.crnn_beat_model import CRNNBeatModel


class BeatTrackPLModule(LightningModule):

    def __init__(self,
        hidden_size: int = 64,
        num_layers_convs: int = 5,
        num_layers_gru: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.15,
        lr: float = 1e-3,
        monitor: str = 'val_f1',
        mode: str = 'max',
        patience: int = 20,
        model_state_dict_filename: str = '{epoch}-{val_loss:.4f}-{val_f1:.4f}',
        tasks: list = ['beat', 'downbeat'],  # what tasks to train/finetune        
    ):
        
        super().__init__()

        self.model = CRNNBeatModel(
            hidden_size = hidden_size,
            num_layers_convs = num_layers_convs,
            num_layers_gru = num_layers_gru,
            kernel_size = kernel_size,
            dropout = dropout,
        )

        self.lr = lr
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.model_state_dict_filename = model_state_dict_filename
        self.tasks = tasks

    def forward(self, x):
        return self.model(x)
    
    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=1,
            filename=self.model_state_dict_filename,
            save_last=True,
        )
        earlystop_callback = EarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
        )
        return [checkpoint_callback, earlystop_callback]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            monitor=self.monitor,
            mode=self.mode,
            factor=0.2,
            patience=10,
            min_lr=1e-10,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.monitor,
        }

    def training_step(self, batch, batch_idx):
        
        # Get batch
        x, (y_b, y_db, y_ibi), length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        # y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat = self(x)

        # Mask out the padding part
        mask = torch.ones(y_b_hat.shape).to(y_b_hat.device)
        for i in range(y_b_hat.shape[0]):
            mask[i, length[i]:] = 0
        y_b_hat = y_b_hat * mask
        y_db_hat = y_db_hat * mask
        # y_ibi_hat = y_ibi_hat * mask.unsqueeze(1)

        # Loss
        loss_b = F.binary_cross_entropy_with_logits(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy_with_logits(y_db_hat, y_db)
        # loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss = 0
        if 'beat' in self.tasks:
            loss += loss_b
        if 'downbeat' in self.tasks:
            loss += loss_db
        # if 'tempo' in self.tasks:
        #     loss += loss_ibi

        # Logging
        logs = {
            'train_loss': loss,
            'train_loss_b': loss_b,
            'train_loss_db': loss_db,
            # 'train_loss_ibi': loss_ibi,
        }
        self.log_dict(logs, prog_bar=True)

        return {'loss': loss, 'logs': logs}
    

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, 'val')
    

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, 'test')
    
    
    def _eval_step(self, batch, split):
        # Data
        x, (y_b, y_db, y_ibi), length = batch
        x = x.float()
        y_b = y_b.float()
        y_db = y_db.float()
        # y_ibi = y_ibi.long()
        length = length.long()

        # Forward pass
        y_b_hat, y_db_hat = self(x)

        # Mask out the padding part
        for i in range(y_b_hat.shape[0]):
            y_b_hat[i, length[i]:] = 0
            y_db_hat[i, length[i]:] = 0
            # y_ibi_hat[i, :, length[i]:] = 0

        # Loss
        loss_b = F.binary_cross_entropy_with_logits(y_b_hat, y_b)
        loss_db = F.binary_cross_entropy_with_logits(y_db_hat, y_db)
        # loss_ibi = nn.NLLLoss(ignore_index=0)(y_ibi_hat, y_ibi)
        loss = loss_b + loss_db # + loss_ibi

        # Metrics
        accs_b, precs_b, recs_b, fs_b = 0, 0, 0, 0
        accs_db, precs_db, recs_db, fs_db = 0, 0, 0, 0

        for i in range(x.shape[0]):
            # get sample from batch
            y_b_hat_i = (y_b_hat[i, :length[i]].sigmoid() > 0.5).int()
            y_db_hat_i = (y_db_hat[i, :length[i]].sigmoid() > 0.5).int()
            # y_ibi_hat_i = y_ibi_hat[i, :, :length[i]].topk(1, dim=0)[1][0]
            
            y_b_i = y_b[i, :length[i]]
            y_db_i = y_db[i, :length[i]]
            # y_ibi_i = y_ibi[i, :length[i]]

            # # filter out ignore indexes
            # y_ibi_hat_i = y_ibi_hat_i[y_ibi_i != 0]
            # y_ibi_i = y_ibi_i[y_ibi_i != 0]

            # get accuracy
            acc_b, prec_b, rec_b, f_b = f_measure_framewise(y_b_i, y_b_hat_i)
            acc_db, prec_db, rec_db, f_db = f_measure_framewise(y_db_i, y_db_hat_i)
            
            accs_b += acc_b
            precs_b += prec_b
            recs_b += rec_b
            fs_b += f_b

            accs_db += acc_db
            precs_db += prec_db
            recs_db += rec_db
            fs_db += f_db

        accs_b /= x.shape[0]
        precs_b /= x.shape[0]
        recs_b /= x.shape[0]
        fs_b /= x.shape[0]

        accs_db /= x.shape[0]
        precs_db /= x.shape[0]
        recs_db /= x.shape[0]
        fs_db /= x.shape[0]

        # Logging
        logs = {
            '{}_loss'.format(split): loss,
            '{}_loss_b'.format(split): loss_b,
            '{}_loss_db'.format(split): loss_db,
            # '{}_loss_ibi'.format(split): loss_ibi,
            '{}_acc_b'.format(split): accs_b,
            '{}_prec_b'.format(split): precs_b,
            '{}_rec_b'.format(split): recs_b,
            '{}_f_b'.format(split): fs_b,
            '{}_acc_db'.format(split): accs_db,
            '{}_prec_db'.format(split): precs_db,
            '{}_rec_db'.format(split): recs_db,
            '{}_f_db'.format(split): fs_db,
            '{}_f1'.format(split): fs_b,  # this will be used as the monitor for logging and checkpointing callbacks
        }
        self.log_dict(logs, prog_bar=True)

        return {'val_loss': loss, 'logs': logs}
    

def f_measure_framewise(y, y_hat):
    acc = (y_hat == y).float().mean()
    TP = torch.logical_and(y_hat==1, y==1).float().sum()
    FP = torch.logical_and(y_hat==1, y==0).float().sum()
    FN = torch.logical_and(y_hat==0, y==1).float().sum()

    p = TP / (TP + FP + np.finfo(float).eps)
    r = TP / (TP + FN + np.finfo(float).eps)
    f = 2 * p * r / (p + r + np.finfo(float).eps)
    return acc, p, r, f

