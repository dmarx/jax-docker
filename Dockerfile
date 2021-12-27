# via https://github.com/google/jax/issues/6340
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.62

# install python3-pip
RUN apt update && apt install python3-pip -y

# install dependencies via pip
RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html
