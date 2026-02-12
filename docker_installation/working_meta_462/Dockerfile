FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /work

# Basic tools + deps mentioned in the 4.6.2 Linux install doc
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
    libcanberra-gtk-module mesa-utils ffmpeg \
    python3-pip python3-dev \
 && rm -rf /var/lib/apt/lists/*

# 1) Add Prophesee APT repo list (this is your downloaded file)
COPY metavision.list /etc/apt/sources.list.d/metavision.list

# (Optional) Pin to a specific SDK version if needed (doc shows pinning example)
# Replace 4.6.2 with the exact version string if the repo uses 4.6.2.0 / 4.6.2-1 etc.
RUN printf "Package: metavision-*\nPin: version 4.6.2*\nPin-Priority: 1001\n" \
    > /etc/apt/preferences.d/prophesee

# 2) Install Metavision SDK packages
RUN apt-get update \
 && apt-get install -y metavision-sdk \
 && rm -rf /var/lib/apt/lists/*

# Optional: if you need python 3.9 bindings on Ubuntu 22.04 (doc mentions this extra package)
# RUN apt-get update && apt-get install -y metavision-sdk-python3.9 && rm -rf /var/lib/apt/lists/*

# Copy IDS uEye EVS plugin (must match SDK 4.6.2)
COPY ueye-evs_4.6.2.2_amd64.deb /tmp/

RUN apt-get update \
 && apt-get install -y /tmp/ueye-evs_4.6.2.2_amd64.deb \
 && rm /tmp/ueye-evs_4.6.2.2_amd64.deb \
 && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install \
    numpy \
    h5py \
    opencv-python \
    pyrealsense2 \
    pyzmq \
    msgpack


# Optional: HDF5 plugin path for Ubuntu 22.04 (doc gives this exact path)
ENV HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH:-}:/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins

