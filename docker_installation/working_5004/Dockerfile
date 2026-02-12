FROM prophesee/openeb:ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /work

# Pick an OpenEB version/tag/branch.
# If you REALLY need "4.6.2": OpenEB open-source tags around that era may be "v4.6.0" etc.
# Default here is a known public branch from the OpenEB repo docs.
ARG OPENEB_REF=5.2.0

# --- System deps (safe even if base already has them) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils build-essential software-properties-common \
    wget unzip curl git cmake \
    libopencv-dev libboost-all-dev libusb-1.0-0-dev libprotobuf-dev protobuf-compiler \
    libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg \
    python3-pip python3-venv python3-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Get OpenEB sources ---
RUN git clone --recursive https://github.com/prophesee-ai/openeb.git /work/openeb \
 && cd /work/openeb \
 && git checkout "${OPENEB_REF}" \
 && git submodule update --init --recursive

# --- Python deps (use OpenEB's own pinned requirements) ---
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r /work/openeb/utils/python/requirements_openeb.txt \
 && python3 -m pip install -r /work/openeb/utils/python/requirements_pytorch_cpu.txt \
 && python3 -m pip install pyrealsense2

# --- pybind11 (needed for Python bindings) ---
# Current OpenEB docs recommend pybind11 2.11.0. :contentReference[oaicite:1]{index=1}
RUN wget -q https://github.com/pybind/pybind11/archive/v2.11.0.zip \
 && unzip -q v2.11.0.zip \
 && cd pybind11-2.11.0 \
 && mkdir build && cd build \
 && cmake .. -DPYBIND11_TEST=OFF \
 && cmake --build . \
 && cmake --build . --target install \
 && cd /work \
 && rm -rf pybind11-2.11.0 v2.11.0.zip

# --- Build + install OpenEB (includes Python bindings by default) ---
# You can explicitly disable bindings with -DCOMPILE_PYTHON3_BINDINGS=OFF (we want ON). :contentReference[oaicite:2]{index=2}
RUN cd /work/openeb \
 && mkdir -p build && cd build \
 && cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DPython3_EXECUTABLE=$(command -v python3) \
      -DCOMPILE_PYTHON3_BINDINGS=ON \
 && cmake --build . -- -j"$(nproc)" \
 && cmake --build . --target install \
 && ldconfig

# Runtime env so libs/plugins are found (matches docs guidance). :contentReference[oaicite:3]{index=3}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin:${HDF5_PLUGIN_PATH}

# Copy uEye EVS driver
COPY ueye-evs_5.0.0.4_amd64.deb /tmp/

# Install it properly via apt (not dpkg)
RUN apt-get update \
 && apt-get install -y /tmp/ueye-evs_5.0.0.4_amd64.deb \
 && rm /tmp/ueye-evs_5.0.0.4_amd64.deb \
 && rm -rf /var/lib/apt/lists/*


# Quick sanity check at build time (fails build if bindings aren't there)
RUN python3 -c "import metavision_hal, metavision_core; print('OK:', metavision_hal.__name__, metavision_core.__name__)"

