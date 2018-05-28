ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
      sklearn_pandas \
      tensorflow-gpu && \
    pip install https://cntk.ai/PythonWheel/GPU/cntk-2.1-cp36-cp36m-linux_x86_64.whl && \
    conda install \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pygpu \
      pyyaml \
      scikit-learn \
      six \
      theano && \
    git clone git://github.com/keras-team/keras.git /src && \ 
    cd /src && git checkout tags/2.1.6 && \
    pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git@2.1.6 && \
    conda clean -yt

# Install COCO API
USER root
RUN conda install cython && \
    mkdir -p /cocoapi && \
    chown $NB_USER /cocoapi

RUN apt-get update && \
    apt-get install libgtk2.0-0 -y && \
    rm -rf /var/lib/apt/lists/* && \
    conda install --channel https://conda.anaconda.org/menpo opencv3 && \
    pip install git+git://github.com/aleju/imgaug.git && \ 
    git clone git://github.com/waleedka/coco.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Install Additional python packages
RUN conda install scikit-image

# Downgrade CuDNN for compatibility with Tensforflow 1.5
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
    libcudnn7=7.0.4.31-1+cuda9.0 \
    libcudnn7-dev=7.0.4.31-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*

# Install API library and pycococreator
RUN pip install flask-restplus && \
    pip install git+git://github.com/waspinator/pycococreator.git@0.2.0

RUN pip install sacred && \
    pip install git+git://github.com/keras-team/keras-contrib.git

# switch to "keras" user
USER $NB_USER

# ADD theanorc /home/keras/.theanorc

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /

EXPOSE 8889

CMD jupyter notebook --port=8889 --ip=0.0.0.0 --no-browser --allow-root
