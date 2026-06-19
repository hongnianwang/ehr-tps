#!/usr/bin/env python3
# This script runs time series shapelet mining for each laboratory variable.
# It reads Train and Test files for each feature from a given directory,
# and runs the s3m_eval script for each feature.

import os

path_data = "../data/mimiciv/ts_lab/"
dirlist = os.listdir(path_data)

# Extract feature names from the filenames
features = set()
for file in dirlist:
    if file.startswith("Train_") or file.startswith("Test_"):
        feature_name = file.split('_')[1]
        features.add(feature_name)

# Iterate over each feature to read corresponding Train and Test files
for feature in features:
    try:
        path_train = os.path.join(path_data, f"Train_{feature}")
        path_test = os.path.join(path_data, f"Test_{feature}")

        os.system(
            f"bash s3m_eval_lab.sh -i {path_train} -e {path_test} -d {feature[:-4]}"
        )
        print(f"done {feature}")
    except Exception as e:
        print(f"Error processing feature '{feature}': {e}")
