FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Build args
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV WORKER_MODEL_DIR=/app/model
ENV WORKER_USE_CUDA=True
ENV WORKER_MODEL_NAME=SG161222/RealVisXL_V4.0
ENV WORKER_ID_LENGTH=4
ENV WORKER_TOTAL_LENGTH=5
ENV WORKER_SCHEDULER_TYPE=euler


SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV WORKER_DIR=/app
RUN mkdir ${WORKER_DIR}
WORKDIR ${WORKER_DIR}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client cmake g++ ninja-build && \
    apt-get install -y libaio-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
                && chown -R user:user ${WORKER_DIR}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV SHELL=/bin/bash

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt ${WORKER_DIR}/requirements.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements.txt && \
    rm ${WORKER_DIR}/requirements.txt

# Fetch the model
COPY builder/build_model.py ${WORKER_DIR}/build_model.py
RUN python3 -u ${WORKER_DIR}/build_model.py --model-name="${WORKER_MODEL_NAME}" --model-dir="${WORKER_MODEL_DIR}" --use-cuda
RUN rm ${WORKER_DIR}/build_model.py

# Add src files (Worker Template)
ADD src ${WORKER_DIR}

ENV RUNPOD_DEBUG_LEVEL=INFO

CMD python3 -u ${WORKER_DIR}/rp_handler.py --model-dir="${WORKER_MODEL_DIR}"
