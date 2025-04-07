# Symbolic Beat Tracker

This folder provides code for the symbolic beat tracker. To work on the code, please make sure that you are under the current folder:

    cd ./symbolic_beat_tracker


## Model training

The symbolic beat tracking model is modified from the PM2S model, which is based on a CRNN architecture. We removed the tempo tracking part, to let the model jointly learn beat and downbeat tracking.

To train the model, please run:

    python3 main.py fit --config config.yaml
