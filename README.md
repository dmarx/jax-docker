# jax-docker
containerized jax environment

# Usage

```
git clone https://github.com/dmarx/jax-docker.git
cd jax-docker
docker build -t jax-base .
docker run --gpus=all -it jax-base
```
