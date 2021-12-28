# via https://github.com/google/jax/issues/6340
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    # https://storage.googleapis.com/jax-releases/jax_releases.html
    # JAXLIB_VERSION=0.1.62
    JAXLIB_VERSION=0.1.65 \
    JUPYTER_TOKEN=foobar
    

# install python3-pip
RUN apt update && apt install python3-pip -y

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
ln -sf /usr/bin/pip3 /usr/bin/pip

# install dependencies via pip
RUN pip install --no-cache-dir --upgrade numpy scipy six wheel matplotlib pandas
RUN pip install --no-cache-dir jaxlib==${JAXLIB_VERSION}+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install --no-cache-dir jax==0.2.17
RUN pip install --no-cache-dir --upgrade trax flax dm-haiku optax chex jraph
RUN pip install --no-cache-dir --upgrade tensorflow 

# install assorted utilities
RUN apt install -y git

# Install and configure jupyter entrypoint
EXPOSE 8889
RUN pip install --no-cache-dir jupyterlab
ENTRYPOINT jupyter lab \
    --allow-root \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8889 \
    --ServerApp.token=$JUPYTER_TOKEN