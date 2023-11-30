# PAUNet-Weather4cast-2023
This code was used for the submission to the Weather4cast 2023 challenge - NeurIPS2023, by the team "CHG"

This repo is the official implementation of "PAUNet: Precipitation Attention-based U-Net for rain prediction from satellite radiance data" 

conda env create -f environment.yml creates an environment with the required packages to execute the code in this repo

## Description of files
This repository consists of the following files, and an explanation is provided
- Weather4cast-2023.yml: A conda yml file used to set up the installations
- PAUNet.py: A Python file containing the PAUNet model architecture along with usage
- timestamps_and_splits_stage2.csv: A CSV file containing the information about the train, validation, and test data splits based on timestamps
- PAUNet_core.h5: Trained model corresponding to the core challenge
- PAUNet_nowcasting.h5: Trained model corresponding to the nowcasting challenge
- PAUNet_transfer_learning.h5: Trained model corresponding to the transfer_learning challenge
- prediction.py: A Python script designed for model predictions

## Installation
To use this code, you need the following
- Clone the repository
- Setup the conda environment 'Weather4cast-2023.yml'. This environment is solely for prediction purposes. Thus, GPU usage is not implemented.
- Fetch the data you want from the competition website. Follow the instructions at https://github.com/agruca-polsl/weather4cast-2023.git. The data should be in the data directory following the structure specified in the instructions.

## Predictions
The prediction script 
Arguments:
- model_path: Path to the saved model
- challenge: name of the challenge, {core, nowcasting, transfer_learning}
- len_seq_predict: number of prediction sequence time instance, either 32 or 16, based on the challenge
- year: year
- region: region name
- inputs_root: the root directory of input files, possibly /data in the starter kit. Based on the year and region information, the script will look for specific files within this directory
- target_folder: target folder to save model outputs
  
- Run: python prediction.py "PAUNet_core.h5" "core" "32" "2019" "roxi_0004" "data" "predictions"

