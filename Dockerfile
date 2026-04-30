# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy project files
COPY . .

# Install Python dependencies using uv
# Note: we use --system to install into the container's python environment
RUN uv pip install --system -r pyproject.toml

# Ensure models and data directories exist
RUN mkdir -p checkpoints logs figs

# Set environment variables for CUDA
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# The entrypoint will run our training script
# Vertex AI will pass additional arguments to this command
ENTRYPOINT ["python", "train.py"]
