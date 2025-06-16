# Use a Python base image. Python 3.9 is a good option for Detectron2 compatibility.
# 'slim-buster' is a lightweight Debian-based image, suitable for CPU-only deployments.
FROM python:3.9-slim-buster

# Set the working directory inside the container.
# Your 'app' folder (containing main.py) will be copied into '/app/app'.
WORKDIR /app

# Install system-level dependencies.
# These are crucial for OpenCV and other underlying libraries that PyTorch/Detectron2 might need.
# 'build-essential' is for compiling, 'git' for cloning Detectron2,
# 'libgl1-mesa-glx', 'libsm6', 'libxext6', 'libglib2.0-0' are common graphical/display libraries
# that prevent headless environments from crashing when using image processing (like OpenCV).
# '--no-install-recommends' helps keep the image size smaller.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libglib2.0-0 && \
    # Clean up APT cache to reduce the final image size.
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy 'requirements.txt' first.
# This leverages Docker's caching, so if only your code changes (not dependencies),
# this step won't rerun unnecessarily, speeding up builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch and TorchVision.
# --- IMPORTANT: These lines are specifically for a CPU-only environment (like Render's free tier). ---
# Ensure these versions are compatible with your Detectron2 training setup if possible.
# The '--index-url' points to the CPU-specific PyTorch wheels.
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# ... (código anterior) ...

# Install Detectron2. Its installation requires 'git'.
# --> ESTA ES LA LÍNEA CLAVE: Asegúrate de que quede así. <--
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# ... (código posterior) ...
# Copy your 'app' folder (which contains 'main.py') into the container.
# It will reside at '/app/app' inside the container.
COPY app /app/app

# Expose the port on which your FastAPI application will listen.
# Uvicorn, by default, often uses port 8000. Render will map its external port to this internal one.
EXPOSE 8000

# Command to start the FastAPI application using Uvicorn.
# 'app.main:app' tells Uvicorn to look for the 'app' object inside 'main.py', which is located in the 'app' directory.
# '--host 0.0.0.0' makes the app listen on all available network interfaces.
# '--port 8000' specifies the internal port for the application.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]