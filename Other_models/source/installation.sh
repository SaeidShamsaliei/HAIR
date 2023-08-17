#!/bin/bash
# Installing the required packages for docker image

echo "Is Conda Env activated? press no to stop"
pip install --no-input opencv-python==4.5.1.48
conda install -y scikit-image
conda install -y -c conda-forge einops
pip install --no-input keras-swa
pip install --no-input git+https://github.com/mjkvaak/ImageDataAugmentor
pip install --no-input "opencv-python-headless<4.3"
pip install --no-input -U segmentation-models
conda install -y -c conda-forge tqdm
pip install --no-input keras-tuner --upgrade
