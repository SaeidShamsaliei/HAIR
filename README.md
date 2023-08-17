# HAIR: A Dataset of Historic Aerial Images of Riverscapes for Semantic Segmentation


# Training

## MagNet
Training the [**MagNet**](https://github.com/VinAIResearch/MagNet) has two steps:

### Training backbone networks

1. Make the virtual environment with requirements.
2. Activate the environment. For example:
```bash
cd <path to the directory>/MagNet-main/

source <env name>/bin/activate
```
3. Modify the config file at `<path to the directory/MagNet-main/backbone/experiments/deepglobe_river>`
4. 
```bash
cd <path to the directory>/MagNet-main/

source <env name>/bin/activate
```
### Training refinement modules

modify the parameters of the `<path to the directory/MagNet-main/scripts/deepglobe_river/train_magnet.sh `

### Inference

Modify the script `MagNet-main/prediction_from_dir.py`. The parameters are smiliar to the original [**MagNet**](https://github.com/VinAIResearch/MagNet).

```bash
python prediction_from_dir.py --progressive_flag 1 \
--dataset deepglobe_river \
--scales 612-612,1224-1224,2448-2448 \
--crop_size 612 612 \
--input_size 508 508 \
--model fpn \
--pretrained <your path> \
--pretrained_refinemen <your path> \
--num_classes 6 \
--n_points 0.75 \
--n_patches -1 \
--smooth_kernel 11 \
--sub_batch_size 1 \

```


## All the models except MagNet

### Training the networks

1.Make the environment by making a docker container or setting up a conda environment using the requirements. In case of using docker:
```bash
# move to the directory
cd <path to the directory>/Other_models/

# To build the docker container
docker build -t saeids/segmentation segmentation/

# To start the container
docker run --rm -it -v $-v $<path to workscape data>:/workspace/data -v $<path to source code>:/workspace/code -v <path to data>:/data -p <your port of choice>:8888 --gpus device=<your device of choice> saeids/tf_gpu_v1 /bin/bash
```
2. Activate the environment. For example:
```bash
conda activate tf_gpu_env
```

3. Install all the required libreries
```bash
# If using the docker:
sh ./workspace/code/Other_models/source/installation.sh
```

4. Modify the training script `Other_models/source/run_from_dir_pretrain.py` for pre-trained on grayscaled DeepGlobe, and `Other_models/source/run_from_dir_with_args.py` for training from scratch.

4. Train the model
```bash
cd <path to the directory>/Other_models/source
# train
python run_from_dir_pretrain.py
# or
python run_from_dir_pretrain.py
```

### Inference
1. Modify the config file at `Other_models/source/configs/prediction_config.yaml`

2. run the script
```bash
# make sure the directory is correct
cd <path to the directory>/Other_models/source

python run_predictions_costum.py
```

# Evalattion

## MagNet

1. Make sure to follow the inference and have the predictions for all the images
2. Modify the paths at the script `Other_models/source/Nips_test_miou_magnet.py`
3. Run the script:
```bash
# make sure the directory is correct
cd <path to the directory>/Other_models/source

python Nips_test_miou_magnet.py
```


## All the models except MagNet

1. Modify the paths at the script `Other_models/source/Nips_test_miou.py`
2. Run the script:
```bash
# make sure the directory is correct
cd <path to the directory>/Other_models/source

python Nips_test_miou.py
```


# Grayscale DeepGlobe Pre-trained weightes

The weights of pre-trained models on grayscaled [**DeepGlobe**](http://deepglobe.org/) can be found in [**Pretrained weights**](https://drive.google.com/drive/folders/1i_2zE-kjR37h6yGVWRaq554bF1-ykzMQ).