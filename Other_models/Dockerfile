FROM continuumio/miniconda3

COPY tf_gpu_environment.yml ./

RUN conda env create -f tf_gpu_environment.yml

RUN apt-get --allow-releaseinfo-change update
RUN apt-get -y install gcc
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python==4.5.1.48
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "tf_gpu_env", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure tensorflow is installed:"
RUN python -c "import tensorflow as tf"