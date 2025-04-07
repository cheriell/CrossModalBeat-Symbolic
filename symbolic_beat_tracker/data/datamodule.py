import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../'))

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningDataModule

from data.io import read_note_sequence
from data.constants import tolerance, resolution
    

class ASAPDataModule(LightningDataModule):


    def __init__(self, 
        dataset_path: str,   # Path to the ASAP dataset
        max_length: int,
        batch_size_train: int, 
        batch_size_eval: int,
        num_workers: int, 
        input_features: list, 
    ):

        super().__init__()

        self.dataset_path = dataset_path
        self.max_length = max_length
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers

        assert all(feature in ['pitch', 'onset', 'duration', 'velocity'] for feature in input_features)
        self.input_features = input_features


    def setup(self, stage = None):
        # Dataset metadata
        metadata = pd.read_csv('../ASAP_dataset_split.csv')
        
        # Get absolute paths to the midi and annotation files in (midi, annot) pairs
        self.fns = {'train': [], 'val': [], 'test': []}
        
        for i, row in metadata.iterrows():
            split = row['split']
            midi_fn = os.path.join(self.dataset_path, row['midi_performance'])
            annot_fn = os.path.join(self.dataset_path, row['performance_annotations'])
            self.fns[split].append((midi_fn, annot_fn))


    def train_dataloader(self):
        dataset = ASAPDataset(fns=self.fns['train'], input_features=self.input_features, max_length=self.max_length, eval=False)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size = self.batch_size_train,
            sampler = sampler,
            num_workers = self.num_workers,
            drop_last = True,
        )
        return dataloader


    def val_dataloader(self):
        dataset = ASAPDataset(fns=self.fns['val'], input_features=self.input_features, max_length=self.max_length, eval=True)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size = self.batch_size_eval,
            sampler = sampler,
            num_workers = self.num_workers,
            drop_last = False,
        )
        return dataloader


    def test_dataloader(self):
        dataset = ASAPDataset(fns=self.fns['test'], input_features=self.input_features, max_length=self.max_length, eval=True)
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size = self.batch_size_eval,
            sampler = sampler,
            num_workers = self.num_workers,
            drop_last = False,
        )
        return dataloader


class ASAPDataset(torch.utils.data.Dataset):

    
    def __init__(self, 
        fns: list,  # List of (midi, annot) pairs
        input_features: list, # Input features to use (for the input note sequence)
        max_length: int, # Maximum length of the input note sequence (number of notes)
        eval: bool  # Whether to use the dataset for evaluation or training
    ) -> None:

        self.fns = fns
        self.input_features = input_features
        self.max_length = max_length
        self.eval = eval
        self.len = len(fns) * 2  # On average, we sample each performance twice per epoch

        # Prepare the dataset. 
        self.note_seq_list = []
        self.outputs_list = []
        
        for fn_idx, (midi_fn, annot_fn) in enumerate(self.fns):
            print('Preparing sample {}/{}'.format(fn_idx+1, len(self.fns)), end='\r')

            # Note sequence (pitch, onset, duration, velocity)
            note_seq = read_note_sequence(midi_fn)  # (length, 4)

            # Beats and downbeats
            annot = pd.read_csv(annot_fn, header=None, sep='\t')
            beats = annot[0].to_numpy()
            downbeats = np.array([annot[0][i] for i in range(len(annot)) if annot[2][i] == 'db'])
            
            probs_beat = np.array([np.min(np.abs(beats - note_seq[i,1])) < tolerance for i in range(len(note_seq))]).astype(int)
            probs_downbeat = np.array([np.min(np.abs(downbeats - note_seq[i,1])) < tolerance for i in range(len(note_seq))]).astype(int)
            ibis = []
            for i in range(len(note_seq)):
                l = len((beats - note_seq[i,1]) < 0) - 1
                if l == -1: l += 1
                if l+1 == len(beats): l -= 1
                ibis.append(beats[l+1] - beats[l])
            
            outputs = np.array([probs_beat, probs_downbeat, ibis])  # (3, length)
            
            # Append to the list
            self.note_seq_list.append(note_seq)
            self.outputs_list.append(outputs)
        
        print()
        

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Load data
        idx = idx % len(self.fns)
        x = self.note_seq_list[idx]
        y = self.outputs_list[idx]

        if not self.eval:
            # Data augmentation
            x, y = self.data_augmentation(x, y)

            # Sample segment (self.max_length notes) & input features
            length = len(x)
            start_idx = np.random.choice(range(max(1, len(x) - self.max_length)))   # All performances are longer than self.max_length
            x = x[start_idx: start_idx + self.max_length, :]
            y = y[:, start_idx: start_idx + self.max_length]

            # Padding
            if length < self.max_length:
                x = np.concatenate([x, np.zeros((self.max_length - length, 4))], axis=0)
                y = np.concatenate([y, np.zeros((3, self.max_length - length))], axis=1)

        else:
            length = len(x)
            # Pad all sequences into the maximum length in the dataset (247, 2046)
            x = np.concatenate([x, np.zeros((2500 - length, 4))], axis=0)
            y = np.concatenate([y, np.zeros((3, 2500 - length))], axis=1)
        
        # Use the specified input features
        if 'velocity' not in self.input_features: x[:, 3] = 60
        if 'duration' not in self.input_features: x[:, 2] = 0.1
        

        # Separate y
        probs_beat = y[0,:].astype(float)
        probs_downbeat = y[1,:].astype(float)
        ibis = np.round(np.clip(y[2,:], 0, 4) / resolution).astype(int)  # Convert ibi into categorical

        return x, (probs_beat, probs_downbeat, ibis), length
    
    @staticmethod
    def data_augmentation(x, y):
        # tempo change
        tempo_change_ratio = random.uniform(0.8, 1.2)
        x[:, 1:3] *= 1 / tempo_change_ratio  # onset and duration
        y[2,:] *= 1 / tempo_change_ratio   # ibi

        # pitch shift
        shift = round(random.uniform(-12, 12))
        x[:, 0] += shift

        # extra notes
        x_new = np.zeros((len(x) * 2, 4))  # duplicate
        y_new = np.zeros((3, len(x) * 2))
        x_new[::2,:] = np.copy(x)   # original notes
        y_new[:,::2] = np.copy(y)
        x_new[1::2,:] = np.copy(x)   # extra notes
        y_new[:, 1::2] = np.copy(y)
        # octave shift (+-12) for extra notes only
        octave_shift = ((np.round(np.random.random(len(x_new))) - 0.5) * 24). astype(int)
        octave_shift[::2] = 0
        x_new[:,0] += octave_shift
        x_new[:,0][x_new[:,0] < 0] += 12
        x_new[:,0][x_new[:,0] > 127] -= 12
        # random ratio of extra notes
        ratio = random.random() * 0.3
        probs = np.random.random(len(x_new))
        probs[::2] = 0
        remaining = probs < ratio
        x_new = x_new[remaining,:]
        y_new = y_new[:, remaining]

        # missing notes
        concurrent_notes = np.diff(x[:, 1]) < tolerance
        ratio = random.random()
        p = concurrent_notes * np.random.random(len(concurrent_notes))
        remaining = np.concatenate([np.array([True]), p < (1 - ratio)])
        x = x[remaining,:]
        y = y[:, remaining]

        return x, y