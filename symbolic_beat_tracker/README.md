# Symbolic Beat Tracker

This folder provides code for the symbolic beat tracker. To work on the code, please make sure that you are under the current folder:

    cd ./symbolic_beat_tracker


## Model training

The symbolic beat tracking model is modified from the PM2S model, which is based on a CRNN architecture. We removed the tempo tracking part, to let the model jointly learn beat and downbeat tracking.

Before running the training script, please change the `trainer.logger.init_args.save_dir` and `data.init_args.dataset_path` in the `config.yaml` file to your where you want to save your trianing logs and where you saved the ASAP dataset. To train the model, please run:

    python3 main.py fit --config config.yaml
