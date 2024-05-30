# Start from the official PyTorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages if needed
RUN pip install --no-cache-dir opencv-python-headless
WORKDIR /train

COPY requirements ./requirements
RUN pip install -r requirements/requirements.txt --extra-index-url=https://repo.sberned.ru/repository/pypi-public/simple
RUN pip install -r requirements/requirements-debug.txt --index-url=https://pypi.python.org/simple

COPY . .

ENTRYPOINT ["bash", "mleco/entrypoint.sh"]