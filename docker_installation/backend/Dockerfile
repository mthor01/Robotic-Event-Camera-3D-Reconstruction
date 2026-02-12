
# ### Build with:
# 
#     docker build -t deoxys docker_deoxys/
#
# ### Run with:
#
#    docker run -it --rm -v /cshome:/cshome -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged deoxys
#
# ### To test if everything works correctly, one can run the following (careful, that reaching home state is possible from current configuration)
# 
#    python examples/reset_robot_joints.py
#
## Mini tmux guide:
#    prefix key: ctrl+b (+ release), used to insert tmux commands; all commands below assume first pressing and releasing prefix key
#    %: split pane vertically
#    ": to split pane horizontally
#    x: to close current pane
#    left/right/up/down: switch the active panel
#    0/1/2/...: switch the active window
#    ctrl+left/right/up/down: resize panel
#    : followed by command: run commands


# Naming the Docker with `deoxys_autostart`:
# ### Build with:
# 
#     docker build -t deoxys_autostart docker_deoxys/
#
# ### Run with:
#
#    docker run -it --rm -v /cshome:/cshome -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged deoxys_autostart



# FROM ros:noetic
# FROM ros:humble
FROM ubuntu:20.04

WORKDIR /root/

RUN apt update && apt upgrade -y
RUN apt install git nano wget tmux sudo -y 

ENV DEBIAN_FRONTEND=noninteractive
# ENV LANG=en_US.UTF-8
# ENV LC_ALL=en_US.UTF-8
# RUN echo "tzdata tzdata/Areas select Europe" | sudo debconf-set-selections
# RUN echo "tzdata tzdata/Zones/Europe select Berlin" | sudo debconf-set-selections

RUN apt install tzdata -y 




# # add to /etc/security/limits.conf
# nano /etc/security/limits.conf

# @realtime soft rtprio 99
# @realtime soft priority 99
# @realtime soft memlock 102400
# @realtime hard rtprio 99
# @realtime hard priority 99
# @realtime hard memlock 102400

RUN git clone https://github.com/UT-Austin-RPL/deoxys_control && \
    cd deoxys_control/deoxys && \
    # Only needed, if libfranka 0.8.0 is needed due to old robot system version
    # sed -i 's/read version/export version=0.8.0/g' ./InstallPackage && \
    sed -i 's/read version/export version=0.9.2/g' ./InstallPackage && \
    bash ./InstallPackage

RUN cd deoxys_control/deoxys && \
    make -j build_franka=1 && \
    make -j build_deoxys=1


# For some reason lib files dont get created in /lib directory, so move them
RUN cd deoxys_control/deoxys && \
    mv libfranka.so* lib/ && \
    mv libyaml-cpp* lib/ && \
    mv libzmqpp.so lib/


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN bash Miniconda3-latest-Linux-x86_64.sh -b && ./miniconda3/bin/conda init

RUN . ~/.bashrc && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


RUN . ~/.bashrc && \
    conda create -n "deoxys" python=3.8 ipython ipykernel pylint jupyter pip pyzmq PyYAML easydict google-api-python-client pybullet numpy hidapi glfw numba termcolor simplejson mujoco==3.1.0 protobuf==3.20.0 matplotlib h5py scipy -c conda-forge -y

RUN . ~/.bashrc && \
    conda activate deoxys


# Change IP of NUC, as we run the NUC and deoxys code on the same device
RUN sed -i 's/IP: 172.16.0.3/IP: 172.16.0.1/g' deoxys_control/deoxys/config/charmander.yml

# Automatically activate the environment
RUN echo "conda activate deoxys" >> ~/.bashrc


WORKDIR /root/deoxys_control/deoxys


# Add deoxys to pythonpath to enable local imports
RUN echo "export PYTHONPATH=/root/deoxys_control/deoxys:$PYTHONPATH" >> ~/.bashrc



### TODO: with RT Priority 40 the controller (auto_arm) crashes quite often, resulting in lags/pauses until restart. With no RT Priority set it works fine, so leave it like that for now 
# ENTRYPOINT tmux new-session -d -s deoxys_autostart \; \
#            split-window -h 'chrt 40 ./auto_scripts/auto_arm.sh config/charmander.yml' \; \
#            split-window -v 'chrt 40 ./auto_scripts/auto_gripper.sh config/charmander.yml' \; \
#            attach


ENTRYPOINT tmux new-session -d -s deoxys_autostart \; \
    split-window -h './auto_scripts/auto_arm.sh config/charmander.yml' \; \
    split-window -v './auto_scripts/auto_gripper.sh config/charmander.yml' \; \
    attach








