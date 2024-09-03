# Stage 1: Build stage
FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Install dependencies, Python, and Git
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install PyTorch separately to avoid reinstallation of all dependencies
RUN python${PYTHON_VERSION} -m pip install --upgrade pip && \
    python${PYTHON_VERSION} -m pip install torch --no-cache-dir

# Upgrade pip and install Python dependencies
COPY requirements ./requirements
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir -r requirements/requirements.txt

# Stage 2: Final image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ARG PYTHON_VERSION=3.11

# Install Git directly in the final stage for ClearML to track the repository changes
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python and necessary binaries from the build stage
COPY --from=build /usr/local /usr/local
COPY --from=build /usr/lib /usr/lib
COPY --from=build /usr/bin/python${PYTHON_VERSION} /usr/bin/python${PYTHON_VERSION}

# Create a symlink for `python`
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy the ClearML configuration file
RUN --mount=type=secret,id=CLEAR_ML_CONF_PATH,dst=/tmp/clearml.conf \
    cp /tmp/clearml.conf /root/clearml.conf

# Set working directory
WORKDIR /app

# Copy the application code from the build context
COPY . .

# Define an environment variable for script arguments
ARG CMD_ARGS=""
ENV CMD_ARGS=$CMD_ARGS

# Run the script with the provided arguments
CMD ["sh", "-c", "PYTHONPATH=. python simple_classifier/train.py $CMD_ARGS"]