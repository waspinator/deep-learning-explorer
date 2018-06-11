# Keras Docker Environment

This is a docker environment for Keras.

The environment includes the following packages:

- Keras from master
- CUDA 9.0
- CUDNN 7.0.4.31-1
- Tensforflow 1.5
- CNTK 2.1
- OpenCV 3.2
- Python 3.6
- Jupyter Notebook
- COCO API

## Logging in

To "log into" a bash shell in a docker container:

 1. find the id of the container using `docker ps`
 2. log in using `docker exec -it <first two+ letters of the docker container id> bash`
 3. resize the windows size using `reset -w`
