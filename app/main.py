import os
import io
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from contextlib import asynccontextmanager # Necesario para la nueva forma de lifespan events

# --- Configuración de la Aplicación FastAPI ---
app = FastAPI(
    title="Detectron2 Object Detection API",
    description="API para detección de objetos con modelo Detectron2 (.pth) y descarga automática."
)

# --- Variables Globales y Constantes ---
# Define las clases de tu conjunto de datos (Asegúrate de que coincidan con tu entrenamiento)
CLASS_NAMES = ["Ciprés", "Palo Santo", "Pino"]

# Ruta donde se guardará el modelo dentro del contenedor Docker
MODEL_FILENAME = "model_final.pth"
MODEL_PTH_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Variable global para el predictor (se cargará una sola vez)
predictor = None

# --- Funciones de Utilidad ---

def descargar_modelo():
    """
    Descarga el modelo .pth desde Google Drive si no existe.
    Incluye una verificación para detectar si la descarga es una página HTML de error.
    """
    if not os.path.exists(MODEL_PTH_PATH):
        print(f"Descargando el modelo desde Google Drive a: {MODEL_PTH_PATH}...")
        # URL de descarga directa de Google Drive
        url = "https://drive.google.com/uc?export=download&id=1kO8C4-YSRAp7Y9Yx-xcZWYW3erwa-TWi"

        try:
            # Usar stream=True y allow_redirects=True para manejar descargas grandes y redirecciones
            with requests.get(url, stream=True, allow_redirects=True) as r:
                r.raise_for_status() # Lanza un HTTPError para códigos de estado 4xx/5xx

                # *** Verificación de contenido HTML ***
                # Leer los primeros bytes para verificar si es una página HTML
                # Si es HTML, es probable que sea una página de error/advertencia de Google Drive.
                first_chunk = r.iter_content(chunk_size=4096).__next__() # Lee el primer chunk
                
                # Intenta decodificar el chunk para buscar HTML
                try:
                    decoded_chunk = first_chunk.decode('utf-8', errors='ignore').lower()
                    if '<!doctype html>' in decoded_chunk or '<html' in decoded_chunk:
                        # Si parece HTML, no es el archivo del modelo
                        error_preview = decoded_chunk[:500].replace('\n', ' ') # Previsualizar el error HTML
                        raise RuntimeError(
                            f"La URL de Google Drive devolvió una página HTML en lugar del archivo "
                            f".pth. Posiblemente un error, advertencia o límite de descarga. "
                            f"Contenido inicial: {error_preview}..."
                        )
                except UnicodeDecodeError:
                    # Si no se puede decodificar como UTF-8, probablemente es binario, lo cual es bueno.
                    pass

                # Si no es HTML, escribe los bytes leídos y el resto del contenido al archivo
                with open(MODEL_PTH_PATH, "wb") as f:
                    f.write(first_chunk) # Escribe el primer chunk que ya leímos
                    for chunk in r.iter_content(chunk_size=8192): # Continúa escribiendo el resto
                        f.write(chunk)
            print("Modelo descargado exitosamente.")

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Fallo en la descarga HTTP del modelo: {e}")
            raise RuntimeError(f"Fallo al descargar el modelo: {e}. Verifica la URL y la conexión a internet.")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            raise # Relanza el RuntimeError para que FastAPI lo capture
    else:
        print("El modelo ya existe en el contenedor. No es necesaria la descarga.")

def load_predictor_instance():
    """
    Carga el predictor de Detectron2. Se ejecuta una sola vez.
    """
    global predictor
    if predictor is None:
        try:
            print("Configurando Detectron2...")
            cfg = get_cfg()
            # Usamos un modelo base de COCO, luego sobreescribimos las cabezas y pesos.
            # Asegúrate de que esta configuración base sea compatible con tu modelo .pth.
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
            cfg.MODEL.WEIGHTS = MODEL_PTH_PATH
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cpu" # IMPORTE: Usamos CPU para Render (planes gratuitos)

            print("Cargando predictor Detectron2...")
            predictor = DefaultPredictor(cfg)

            # Registra los metadatos de las clases si tu modelo fue entrenado con clases específicas
            # Esto es útil si usaras Visualizer, aunque no directamente para el JSON de la API.
            MetadataCatalog.get("my_dataset_for_api").set(thing_classes=CLASS_NAMES)
            
            print("Predictor Detectron2 cargado exitosamente.")
        except Exception as e:
            print(f"ERROR: Fallo al cargar el modelo Detectron2: {e}")
            # Captura cualquier excepción y relánzala como RuntimeError para manejo en lifespan
            raise RuntimeError(f"Fallo al cargar el modelo: {e}. Asegúrate de que el .pth es válido y las dependencias están instaladas correctamente.")
    return predictor

# --- Lifespan Events de FastAPI (Reemplaza @app.on_event) ---
# Esta es la forma recomendada y más moderna de manejar eventos de inicio/apagado en FastAPI.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Función que maneja los eventos de inicio y apagado de la aplicación.
    Descarga y carga el modelo al inicio.
    """
    print("Iniciando la aplicación...")
    try:
        descargar_modelo() # Primero descarga el modelo
        load_predictor_instance() # Luego carga el predictor
        yield # La aplicación está lista para recibir solicitudes
    except RuntimeError as e:
        print(f"CRÍTICO: La aplicación no pudo iniciarse debido a un error de carga del modelo: {e}")
        # En un entorno de producción real, aquí podrías querer un mejor manejo,
        # pero para Render, el error durante el inicio causará un fallo de despliegue.
        # Re-lanzar la excepción para que el proceso de Uvicorn falle.
        raise
    finally:
        print("Apagando la aplicación...")
        # Aquí puedes añadir lógica para limpiar recursos si es necesario
        pass # Por ahora no hay nada que limpiar explícitamente

app.add_event_handler("startup", lifespan) # Registra el manejador de lifespan con la aplicación

# --- Endpoints de la API ---

# Endpoint de salud para Render
@app.get("/health")
async def health_check():
    """Verifica si la API está en línea y el modelo ha sido cargado."""
    if predictor is not None:
        return {"status": "ok", "message": "API está en línea y el modelo cargado."}
    else:
        # Esto podría ocurrir si el modelo aún no ha terminado de cargar o hubo un error crítico
        raise HTTPException(status_code=503, detail="Modelo no cargado o falló al cargar.")

# Endpoint de predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Realiza la detección de objetos en una imagen subida.
    """
    # Asegurarse de que el predictor esté cargado (ya debería estarlo por el evento startup)
    # Volver a llamarlo aquí es redundante si se usa lifespan, pero no dañino.
    # Mejor quitar la llamada y solo usar 'predictor' directamente si confías en lifespan.
    # No obstante, por seguridad defensiva, si el predictor es None, lo relanza.
    if predictor is None:
        raise HTTPException(status_code=503, detail="El modelo no está listo. Intente de nuevo en un momento.")

    # Validar el tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Por favor, sube una imagen.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Detectron2 espera imágenes en formato BGR de OpenCV
        image_np = np.array(image)[:, :, ::-1]

        # Realizar la inferencia
        outputs = predictor(image_np)

        # Extraer resultados
        instances = outputs["instances"].to("cpu") # Mover a CPU para procesar resultados
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        classes = instances.pred_classes.numpy().tolist()

        return JSONResponse(content={
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "class_names": [CLASS_NAMES[cls_id] for cls_id in classes] # Añadir nombres de clase para la conveniencia
        })
    except Exception as e:
        print(f"ERROR durante la inferencia o procesamiento de imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante la predicción: {e}")