
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

