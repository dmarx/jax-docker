# via https://github.com/google/jax/issues/6340
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    # JAXLIB_VERSION=0.1.62
    JAXLIB_VERSION=0.1.73

# install python3-pip
RUN apt update && apt install python3-pip -y

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
ln -sf /usr/bin/pip3 /usr/bin/pip

# install dependencies via pip
RUN pip install --no-cache-dir --upgrade numpy scipy six wheel matplotlib pandas
RUN pip install --no-cache-dir jaxlib==${JAXLIB_VERSION}+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install --no-cache-dir --upgrade tensorflow 
RUN pip install --no-cache-dir --upgrade jax trax flax dm-haiku optax chex jraph
