import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../'))

from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI(save_config_kwargs = {"overwrite": True})


if __name__ == '__main__':
    cli_main()