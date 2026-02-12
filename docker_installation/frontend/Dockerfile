
# syntax=docker/dockerfile:1.4

# BUILD COMMAND:
# 
#      DOCKER_BUILDKIT=1 docker build --ssh default -t openvla ./docker_openvla
#
#
# RUN COMMANDS:
#
#     docker run -it --rm --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/ott/datasets/:/datasets --net=host --privileged openvla

# OR with /cshome/share/ott/datasets:

#     docker run -it --rm --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /cshome/share/ott/datasets/:/datasets --net=host --privileged openvla


FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/opt/miniconda3/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    nano \
    wget \
    tmux \
    sudo \
    unzip











# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh && \
    /opt/miniconda3/bin/conda clean -ya

# Initialize conda
RUN /opt/miniconda3/bin/conda init bash

RUN . ~/.bashrc && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r



# ################################################################################
# ###  OpenVLA Envrionment Installation
# ################################################################################
# # Create conda environment with Python 3.10
# RUN /opt/miniconda3/bin/conda create -n openvla python=3.10 -y

# # Make RUN commands use the new environment
# SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "openvla", "/bin/bash", "-c"]

# # Install PyTorch with CUDA 12.4 support
# RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
# RUN conda install ipykernel -y

# # Set working directory
# WORKDIR /workspace

# # Clone OpenVLA repository
# RUN git clone https://github.com/openvla/openvla.git

# # Change to openvla directory and install
# WORKDIR /workspace/openvla
# RUN pip install -e .

# # Install dependencies for Flash Attention
# RUN pip install packaging ninja

# # Verify Ninja installation
# RUN ninja --version; echo $?

# # # Install Flash Attention 2
# # RUN conda install -c nvidia/label/cuda-12.1.105 cuda-toolkit
# # RUN conda install -c nvidia/label/cuda-12.9.1 cuda-toolkit -y 
# # RUN pip install "flash-attn==2.5.5" --no-build-isolation


# # ### TODO: alternatively, minimal dependencies could also be installed using:
# # Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# # > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt



WORKDIR /workspace

SHELL ["/bin/bash", "-c"]

################################################################################
###  Deoxys INstallation
################################################################################
# For some reason, this  deoxys installation's server does not work
# Therefore, we use the deoxys_autostart docker container
# In theory, we would not need the full deoxys install here, but currently I don't want to remove the unrequired parts
# TODO in the future this could be reomoved. 
RUN git clone https://github.com/UT-Austin-RPL/deoxys_control && \
    cd deoxys_control/deoxys && \
    # Only needed, if libfranka 0.8.0 is needed due to old robot system version
    # sed -i 's/read version/export version=0.8.0/g' ./InstallPackage && \
    sed -i 's/read version/export version=0.9.2/g' ./InstallPackage && \
    bash ./InstallPackage

RUN cd deoxys_control/deoxys && \
    # make -j build_franka=1 && \ # Since they do not work in this container anyways, we don't need to build them 
    make -j build_deoxys=1


# # For some reason lib files dont get created in /lib directory, so move them
# RUN cd deoxys_control/deoxys && \
#     mv libfranka.so* lib/ && \
#     mv libyaml-cpp* lib/ && \
#     mv libzmqpp.so lib/

RUN . ~/.bashrc && \
    conda create -n "deoxys" python=3.10 ipython ipykernel pylint jupyter pip pyzmq PyYAML easydict google-api-python-client pybullet numpy hidapi glfw numba termcolor simplejson mujoco==3.1.0 protobuf==3.20.0 matplotlib h5py scipy -c conda-forge -y
    # conda create -n "deoxys" python=3.8 ipython ipykernel pylint jupyter pip pyzmq PyYAML easydict google-api-python-client pybullet numpy hidapi glfw numba termcolor simplejson mujoco==3.1.0 protobuf==3.20.0 matplotlib h5py scipy -c conda-forge -y

# Change IP of NUC, as we run the NUC and deoxys code on the same device
RUN sed -i 's/IP: 172.16.0.3/IP: 172.16.0.1/g' deoxys_control/deoxys/config/charmander.yml

# Add deoxys to pythonpath to enable local imports
RUN echo "export PYTHONPATH=/workspace/deoxys_control/deoxys:$PYTHONPATH" >> ~/.bashrc

# ### END INSTALL DEOXYS



################################################################################
###  Librealsense instal
################################################################################
## Librealsense installation from precompiled packages (required to detect multiple connected cameras correctly)
## the Beta debian packages are the easiest to install and support ubuntu 24.04 and ROS2 Jazzy
# RUN mkdir librealsense2 && \
#     cd librealsense2 && \
#     wget https://github.com/realsenseai/librealsense/releases/download/v2.57.5/librealsense2_noble_x86_debians_2_57_5_beta.zip && \
#     unzip librealsense2_noble_x86_debians_2_57_5_beta.zip && \
#     sudo apt install ./librealsense2_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-dbg_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-dev_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-gl_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-gl-dbg_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-gl-dev_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-udev-rules_2.57.5-0~realsense.17779_amd64.deb -y && \
#     sudo apt install ./librealsense2-utils_2.57.5-0~realsense.17779_amd64.deb -y

#### END Librealsense install



# Set the default command to activate the environment and start bash
# CMD ["/opt/miniconda3/bin/conda", "run", "-n", "openvla", "/bin/bash"]

# RUN echo "conda activate openvla" >> ~/.bashrc


# COPY openvla_test.py /workspace/openvla_test.py

# # Optional Cleanup
# RUN rm -rf /var/lib/apt/lists/*


# RUNNING: 
# ENTRYPOINT tmux new-session -d -s deoxys_autostart \; \
#     split-window -h './auto_scripts/auto_arm.sh config/charmander.yml' \; \
#     split-window -v './auto_scripts/auto_gripper.sh config/charmander.yml' \; \
#     attach



# Use this one for playstation controller:
# https://github.com/ArturSpirin/pyPS4Controller

# pip install pyPS4Controller




# ################################################################################
# ### Convert Dataset to RLDS
# ### using 
# ### https://github.com/kpertsch/rlds_dataset_builder
# ### https://github.com/google-research/rlds
# ################################################################################


# RUN git clone https://github.com/kpertsch/rlds_dataset_builder

# RUN cd rlds_dataset_builder && \
#     conda env create -f environment_ubuntu.yml


# # cd example_dataset
# # python3 create_example_data.py
# # tfds build



################################################################################
###  Install My Working Package
################################################################################
RUN mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts


RUN echo "redoooyo"

# RUN --mount=type=ssh \
    # git clone --branch dev2 git@github.com:David0tt/vla_pipeline.git



SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "deoxys", "/bin/bash", "-c"]

# RUN pip install -e ./vla_pipeline

# ENV HYDRA_FULL_ERROR=1



# Alternative OpenVLA setup:
# RUN apt-get update && apt-get install -y cuda-toolkit-12-9
# RUN conda create -n py313 python=3.13 -y && \
#     conda activate py313 && \
#     pip install torch
#     pip install transformers==4.40.1 tokenizers==0.19.1 pillow torchvision 'timm<=1.0.0' flash_attn pyzmq scipy && \
#     pip install pyzmq scipy


# RUN conda create -n py312 python=3.12 -y 

# SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "py312", "/bin/bash", "-c"]

# RUN pip install torch && \
#     pip install transformers==4.40.1 tokenizers==0.19.1 pillow torchvision 'timm<=1.0.0' flash_attn pyzmq scipy accelerate

# Download OpenVLA
# RUN python openvla_test.py 

RUN echo "echo 'ATTENTION: The Deoxys Server installation in this container does not work, use the one from the deoxys container! This one only exists to use the client library. Cleanup should be done in the future.' " >> ~/.bashrc




# Install additional dependencies for franka pipeline into the deoxys environment: 
SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "deoxys", "/bin/bash", "-c"]

RUN pip install typer robosuite

RUN pip install lerobot

################################################################################
###  FIXES
################################################################################

# For some reason, the installed pybullet is built against numpy<2; Fix this by upgrading -> it is built against the correct numpy
RUN pip install --upgrade pybullet

RUN echo "conda activate deoxys" >> ~/.bashrc



# Pull build repository
RUN echo "rebuild3"
RUN --mount=type=ssh \
    git clone git@github.com:David0tt/franka_pipeline.git


RUN sudo apt-get update
RUN yes | unminimize


RUN sudo apt install -y libgtk2.0-dev libgtk-3-dev pkg-config


RUN sudo apt install usbutils -y 




RUN sudo apt install libxkbcommon-x11-0 -y 

RUN pip uninstall -y opencv-python-headless
RUN pip uninstall -y opencv-python
RUN pip install --no-cache-dir opencv-python==4.12.0.88

RUN --mount=type=ssh \
    cd franka_pipeline && \
    git pull


# Already install the franka description package for visualization in rerun
# TODO It is required in a specific version, since afterwards, some changes were made that break the visualization.
RUN cd /workspace/franka_pipeline/franka_pipeline/visualization && \
    git clone https://github.com/frankarobotics/franka_description && \
    cd franka_description && \
    git checkout 68a3288ac16e5b2149777b3c241e5743ac8e06ce


# Installation (for rerun 3D Vis)
RUN pip install trimesh urdf-parser-py Pillow --quiet && \
    pip install pycollada --quiet && \
    pip install xacro && \
    pip install urdf-parser-py trimesh scipy



RUN sudo apt-get update && sudo apt-get install -y lsb-release curl gnupg    
RUN sudo mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | gpg --dearmor | sudo tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | sudo tee /etc/apt/sources.list.d/librealsense.list && \
    sudo apt-get update


   
RUN sudo apt install -y librealsense2-utils librealsense2-dev
    # Optionally librealsense2-dbg


RUN pip install pyrealsense2
RUN pip install msgpack

