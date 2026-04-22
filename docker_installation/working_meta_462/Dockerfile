FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /work

# Basic tools + deps mentioned in the 4.6.2 Linux install doc
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release \
    usbutils libusb-1.0-0 \
    libcanberra-gtk-module mesa-utils ffmpeg \
    python3-pip python3-dev \
 && rm -rf /var/lib/apt/lists/*

# RealSense repository + packages
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | gpg --dearmor > /etc/apt/keyrings/librealsenseai.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo $(lsb_release -cs) main" > /etc/apt/sources.list.d/librealsense.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    librealsense2-utils \
    librealsense2-dev \
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
    msgpack \
    torch \
    torchvision \
    pillow \
    tensorboard \
    tqdm \
    matplotlib \
    scipy\
    open3d


# Optional: HDF5 plugin path for Ubuntu 22.04 (doc gives this exact path)
ENV HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH:-}:/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins

ARG UID=101403
ARG GID=5018

RUN groupadd -g $GID cs_stud || true && \
    useradd -m -u $UID -g $GID mthormann || true

RUN chown root:root /usr/share/metavision/apps/metavision_studio/internal/client/chrome-sandbox \
 && chmod 4755 /usr/share/metavision/apps/metavision_studio/internal/client/chrome-sandbox

USER mthormann

RUN chown mthormann:cs_stud /home/mthormann/.bashrc
