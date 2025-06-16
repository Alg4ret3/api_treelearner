# Usa una imagen base de Python. Python 3.9 es una buena opción para Detectron2.
# python:3.9-slim-buster es adecuada para despliegues solo con CPU.
FROM python:3.9-slim-buster

# Establecer el directorio de trabajo dentro del contenedor
# La carpeta 'app' de tu proyecto se copiará a '/app/app'
WORKDIR /app

# Instalar dependencias del sistema necesarias
# Esto es crucial para OpenCV y otras librerías que PyTorch/Detectron2 podrían necesitar.
# --no-install-recommends ayuda a mantener el tamaño de la imagen reducido.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libglib2.0-0 \ # A menudo necesaria para sistemas de visualización incluso sin GUI
    # Limpieza para reducir el tamaño de la imagen
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt e instalar las dependencias de Python
# Esto se hace primero para aprovechar el caché de Docker si los requisitos no cambian.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch y TorchVision.
# --- IMPORTANTE: Estas líneas son para CPU-only (gratis en Render). ---
# Asegúrate de que las versiones coincidan con lo que usaste en tu entrenamiento si es posible.
# PyTorch 2.0.1 y Torchvision 0.15.2 son comunes con Detectron2.
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Instalar Detectron2. Su instalación requiere git.
# Se usa un commit específico para estabilidad.
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@e05206979603348123c72b21703666b6c00d463b'

# Copiar la carpeta 'app' (que contiene main.py) al contenedor
# Esto significa que main.py estará en /app/app/main.py
COPY app /app/app

# Exponer el puerto en el que correrá la aplicación FastAPI
EXPOSE 8000 # Puerto común para Uvicorn/FastAPI

# Comando para iniciar la aplicación FastAPI con Uvicorn
# 'app.main:app' significa: desde el directorio '/app/app', encuentra 'main.py', y en 'main.py', encuentra el objeto 'app'.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]