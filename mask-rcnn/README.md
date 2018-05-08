# Mask R-CNN

Mask R-CNN is a framework for object instance detection and segmentation.

This repository has a Mask R-CNN implementation written in Keras.

It contains a Jupyter Notebook for training a model, and a REST API
for detecting objects.

## Jupyter Notebook

Start the notebook environment to train your model by `cd`ing into the
`docker` directory and running

    docker-compose up

You can access the Jupyter notebook server on http://localhost:8889

## REST API

After you complete training, you can start the REST API by running

    docker-compose -f docker-compose.yml -f docker-compose.api.yml up
