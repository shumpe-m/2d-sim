docker run --gpus all --rm -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v $HOME/2D-sim:/root/2D-sim \
    2dsim:ubuntu20.04 /bin/bash
