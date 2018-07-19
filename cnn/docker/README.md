# Docker

Use the nvidia-docker runtime to create a container with everything needed to train and infer using the model

## Files

    - environment_name.dockerfile
        - build a container with everything needed to run the model
        - run jupyter notebooks by default
    - docker-compose.yml
        - use nvidia-docker runtime
        - include volumes to the model and data directory
        - open ports for jupyter notebook
    - docker-compose.api.yml
        - open ports for the api
        - run the api