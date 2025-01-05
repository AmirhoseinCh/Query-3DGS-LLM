#!/bin/bash

# Allow the Docker container to display GUI applications on the host's X server
xhost +

# Run the Docker container
docker run \
    --gpus all \
    -it --rm \
    --name Legaussian-container \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/zrb/LEGaussians:/workspace/LEGaussians \
    legaussians

# export PYTHONPATH=$PYTHONPATH:/workspace/LEGaussians/submodules/diff-gaussian-rasterization
# export PYTHONPATH=$PYTHONPATH:/workspace/LEGaussians/submodules/simple-knn
