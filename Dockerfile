# Usa imagen base oficial de Python
FROM python:3.10-slim

# Instala dependencias del sistema necesarias para OpenCV y compilación de detectron2
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia todos los archivos al contenedor
COPY . /app

# Instala PyTorch (CPU), torchvision y otras dependencias
RUN pip install --upgrade pip \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install opencv-python pillow fastapi uvicorn requests numpy

# Instala detectron2 desde GitHub para CPU
RUN pip install git+https://github.com/facebookresearch/detectron2.git

# Expone el puerto
EXPOSE 10000

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
