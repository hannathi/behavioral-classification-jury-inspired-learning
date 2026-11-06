# Jury-inspired learning for automated social behavioral classification in mice

# Scripts
This repository does not include the recordings and parquet files. 
Scripts for different stages of the pipeline can be found in different folders

## FormatDLC
After processing the recordings in DLC, there is a script for creating pairwise parquetfiles (create_pairwise_parquet.py) and extracting features (based on SimBA)

## Preprocessing
This folder has a script for extracting all the features used during training based on the SimBA features (features.py) and then building the final dataset.

## Training
Both the majority voting script (HCScript.py) and jury-based script (JuryScript.py) can be found here.


