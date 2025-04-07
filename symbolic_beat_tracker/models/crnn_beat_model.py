# Remove tempo estimation.
# Change to feed beat into downbeat GRU.

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.encoding import get_in_features, encode_note_sequence


class CRNNBeatModel(nn.Module):

    def __init__(self,
            hidden_size: int = 64,
            num_layers_convs: int = 5,
            num_layers_gru: int = 2,
            kernel_size: int = 3,
            dropout: float = 0.15,
        ):
        """CRNN beat tracking model, modified from the PM2S implementation.
        Joint learning of beats, downbeats, and inter-beat-intervals.
        """

        super().__init__()

        in_features = get_in_features()

        # Convolutional block
        self.convs_beat = conv_layers(
            in_features = in_features,
            hidden_size = hidden_size,
            num_layers_convs = num_layers_convs,
            kernel_size = kernel_size,
            dropout = dropout,
        )
        self.convs_downbeat = conv_layers(
            in_features = in_features,
            hidden_size = hidden_size,
            num_layers_convs = num_layers_convs,
            kernel_size = kernel_size,
            dropout = dropout,
        )

        # GRUs
        self.gru_beat = gru_layers(
            input_size = hidden_size,
            hidden_size = hidden_size,
            num_layers = num_layers_gru,
            dropout = dropout,
        )
        self.gru_downbeat = gru_layers(
            input_size = hidden_size * 2,
            hidden_size = hidden_size,
            num_layers = num_layers_gru,
            dropout = dropout,
        )

        # Linear output layers
        self.out_beat = nn.Linear(hidden_size, 1)
        self.out_downbeat = nn.Linear(hidden_size, 1)
        
        
    def forward(self, x):
        # x: (batch_size, sequence_length, 4)

        x = encode_note_sequence(x)  # (batch_size, sequence_length, in_features)

        # Convolutional block
        x = x.unsqueeze(1)   # (batch_size, 1, sequence_length, in_features)

        # convolutional layers
        x_convs_beat = self.convs_beat(x)  # (batch_size, sequence_length, hidden_size)
        x_convs_downbeat = self.convs_downbeat(x)  # (batch_size, sequence_length, hidden_size)

        # GRUs
        x_gru_beat = self.gru_beat(x_convs_beat)  # (batch_size, sequence_length, hidden_size)
        # Concatenate convolutional and GRU outputs for beat tracking
        x_concat = torch.cat([x_convs_downbeat, x_gru_beat], dim=2)  # (batch_size, sequence_length, hidden_size*2)
        x_gru_downbeat = self.gru_downbeat(x_concat)  # (batch_size, sequence_length, hidden_size)

        # Linear output layers
        y_beat = self.out_beat(x_gru_beat).squeeze(2)  # (batch_size, sequence_length)
        y_downbeat = self.out_downbeat(x_gru_downbeat).squeeze(2)  # (batch_size, sequence_length)

        return y_beat, y_downbeat
    

class conv_layers(nn.Module):

    def __init__(self,
        in_features: int,
        hidden_size: int = 512,
        num_layers_convs: int = 3,
        kernel_size: int = 9,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        for i in range(num_layers_convs):

            self.convs.append(nn.Sequential(
                nn.Conv2d(
                    in_channels = 1 if i == 0 else hidden_size,
                    out_channels = hidden_size,
                    kernel_size = (kernel_size, in_features if i == 0 else 1),
                    padding = (kernel_size // 2, 0),
                ),
                nn.BatchNorm2d(hidden_size),
                nn.ELU(),
                nn.Dropout(p=dropout),
            ))

    def forward(self, x):
        # x: (batch_size, 1, sequence_length, in_features)
        for conv in self.convs:
            x = conv(x)  # (batch_size, hidden_size, sequence_length, 1)
        x = x.squeeze(3).transpose(1, 2)  # (batch_size, sequence_length, hidden_size)

        return x
        

class gru_layers(nn.Module):

    def __init__(self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = True,
        )
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        x, _ = self.gru(x)  # (batch_size, sequence_length, hidden_size*2)
        x = self.linear(x)  # (batch_size, sequence_length, hidden_size)
        return x
    

##################################################
# Sanity check
##################################################

def _check_model():

    batch_size = 2
    sequence_length = 200
    in_features = get_in_features()

    model = CRNNBeatModel()

    x = torch.randn(batch_size, sequence_length, in_features)
    y_beat, y_downbeat = model(x)


def _check_double_sigmoid():

    import numpy as np
    x = np.linspace(-5, 5, 100)
    y = 1 / (1 + np.exp(-x))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(x, y)
    axs[0].set_title('Single sigmoid')
    y = 1 / (1 + np.exp(-y))
    axs[1].plot(x, y)
    axs[1].set_title('Double sigmoid')
    plt.tight_layout()
    plt.savefig('figures/double_sigmoid.png')


if __name__ == '__main__':
    # _check_model()
    _check_double_sigmoid()