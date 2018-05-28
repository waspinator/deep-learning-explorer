# Libraries

    - model library (named after the model name)
        - class based
            - init method build the model using `config`
            - public methods
                - load_weights
                - train
                - detect
            - use coco-style data as input and output
            - includes any custom layers

    - data generator
        - keras data generators compatible with the model
        - data manipulation and augmentation
        - coco-style

    - utils
        - functions for testing
        - any other functions needed