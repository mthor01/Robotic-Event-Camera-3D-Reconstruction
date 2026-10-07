FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /work

# Basic tools + deps mentioned in the Metavision Linux install doc
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release \
    usbutils libusb-1.0-0 \
    libcanberra-gtk-module mesa-utils ffmpeg \
    python3-pip python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Install OpenEB, the open-source part of the Metavision SDK, from Prophesee's
# public APT repository (no login needed). Preprocessing only uses
# metavision_core.event_io to read the raw event files, so neither the full
# SDK nor the IDS uEye EVS camera plugin is needed. The old repository
# apt.prophesee.ai is offline and the public one only offers OpenEB 5.x.
# Pinning the meta package pins all of its components, which require the
# exact same version.
RUN curl -fsSL https://propheseeai.jfrog.io/artifactory/api/security/keypair/prophesee-gpg/public \
      -o /etc/apt/keyrings/prophesee.asc \
 && echo "deb [signed-by=/etc/apt/keyrings/prophesee.asc] https://propheseeai.jfrog.io/artifactory/openeb-debian/ jammy main" \
      > /etc/apt/sources.list.d/openeb.list \
 && apt-get update \
 && apt-get install -y metavision-openeb=5.3.0 \
 && rm -rf /var/lib/apt/lists/*


# open3d is pinned because 0.20.0 extracts empty meshes from the legacy
# ScalableTSDFVolume used by reconstruction.py.
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install \
    numpy \
    h5py \
    opencv-python \
    torch \
    torchvision \
    pillow \
    tensorboard \
    tqdm \
    matplotlib \
    seaborn \
    scipy \
    open3d==0.19.0 \
    huggingface_hub


# Optional: HDF5 plugin path for Ubuntu 22.04 (doc gives this exact path)
ENV HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH:-}:/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins

ARG UID=101403
ARG GID=5018

RUN groupadd -g $GID cs_stud || true && \
    useradd -m -u $UID -g $GID mthormann || true

USER mthormann

RUN chown mthormann:cs_stud /home/mthormann/.bashrc
