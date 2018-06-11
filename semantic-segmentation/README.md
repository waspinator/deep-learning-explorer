# Semantic Segmentation

    - FCN
        - https://arxiv.org/abs/1605.06211


This repository has FCN implementation written in Keras.

It contains Jupyter Notebooks for training the models and detecting classes on custom COCO style data.

## Jupyter Notebook

Start the notebook environment to train your model by `cd`ing into the
`docker` directory and running

    docker-compose up

You can access the Jupyter notebook server on http://localhost:8889

## REST API

After you complete training, you can start the REST API by running

    docker-compose -f docker-compose.yml -f docker-compose.api.yml up
