# Cross-Modal Beat Tracking (Symbolic Beat Tracker)

| [Paper](https://doi.org/10.5334/tismir.238) | [Parent_Code_Repo](https://github.com/SunnyCYC/CrossModalBeat) |

In this repo, we provide the code to train the symbolic beat tracking model in the TISMIR paper, _Cross-Modal Approaches to Beat Tracking: A Case Study on Chopin Mazurkas_.

## Part 1. Dataset

Please refer to `ASAP_dataset_split.ipynb` for the dataset and dataset split details. This notebook include information on the dataset version, the subset and the dataset split that is used in this demo.

## Part 2. Python Environment

This repo is developed with python 3.13 and pytorch 2.6. Please refer to `requirements.txt` for the required python packages.

    pip install -r requirements.txt
    

## Part 3. Symbolic Beat Tracker (Training and Inference)

Please refer to code in folder `./symbolic_beat_tracker`.

For model training and inference, please follow the instructions in `./symbolic_beat_tracker/README.md`
