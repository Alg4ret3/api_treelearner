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
from detectron2.data import MetadataCatalog # Necesario si usas MetadataCatalog


app = FastAPI(
    title="Detectron2 Object Detection API",
    description="API para detección de objetos con modelo Detectron2 (.pth) y descarga automática."
)

# Define las clases de tu conjunto de datos (Asegúrate de que coincidan con tu entrenamiento)
CLASS_NAMES = ["Ciprés", "Palo Santo", "Pino"] 

# Ruta donde se guardará el modelo dentro del contenedor Docker
# os.path.dirname(__file__) será '/app/app' si copias la carpeta 'app' a '/app/app'
MODEL_PTH_PATH = os.path.join(os.path.dirname(__file__), "model_final.pth")

# Variable global para el predictor (se cargará una sola vez)
predictor = None

# Función para descargar el modelo si no existe
def descargar_modelo():
    if not os.path.exists(MODEL_PTH_PATH):
        print(f"Descargando el modelo desde Google Drive a: {MODEL_PTH_PATH}...")
        # URL de descarga directa de Google Drive
        url = "https://drive.google.com/uc?export=download&id=1kO8C4-YSRAp7Y9Yx-xcZWYW3erwa-TWi"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Lanza un error para códigos de estado HTTP malos

            # Manejar redirecciones de Google Drive para archivos grandes
            # A veces Google Drive redirige a una URL diferente para la descarga real
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    with open(MODEL_PTH_PATH, "ab") as f: # Usar "ab" para añadir chunks
                        f.write(chunk)
            print("Modelo descargado exitosamente.")
        except requests.exceptions.RequestException as e:
            print(f"Error al descargar el modelo: {e}")
            raise RuntimeError(f"Fallo al descargar el modelo: {e}. Verifica la URL y la conexión a internet.")
    else:
        print("El modelo ya existe en el contenedor.")

# Función para cargar el predictor de Detectron2
def load_predictor():
    global predictor
    if predictor is None:
        descargar_modelo() # Asegurarse de que el modelo esté descargado
        
        # Configuración de Detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # Usar la longitud de tus clases
        cfg.MODEL.WEIGHTS = MODEL_PTH_PATH # Ruta al modelo descargado
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu" # Usaremos CPU en el nivel gratuito de Render

        try:
            print("Cargando predictor Detectron2...")
            predictor = DefaultPredictor(cfg)
            # Opcional: Registrar metadatos si necesitas Visualizer en otro lugar (no para la API JSON)
            MetadataCatalog.get("my_dataset_for_api").set(thing_classes=CLASS_NAMES)
            print("Predictor Detectron2 cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo Detectron2: {e}")
            raise RuntimeError(f"Fallo al cargar el modelo: {e}. Asegúrate de que el .pth es válido y las dependencias están instaladas.")
    return predictor

# --- Evento de inicio de la aplicación FastAPI ---
# El modelo se descarga y carga cuando la aplicación se inicia por primera vez
@app.on_event("startup")
async def startup_event():
    load_predictor()

# --- Endpoint de salud para Render ---
@app.get("/health")
async def health_check():
    if predictor is not None:
        return {"status": "ok", "message": "API is online and model loaded."}
    else:
        # Esto podría ocurrir si el modelo aún no ha terminado de cargar o hubo un error
        raise HTTPException(status_code=503, detail="Model not yet loaded or failed to load.")

# --- Endpoint de predicción ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Asegurarse de que el predictor esté cargado (ya debería estarlo por el evento startup)
    current_predictor = load_predictor()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)[:, :, ::-1] # Convertir de RGB a BGR para OpenCV (Detectron2 usa BGR por defecto)

    try:
        outputs = current_predictor(image_np)
    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {e}")

    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy().tolist()
    scores = instances.scores.numpy().tolist()
    classes = instances.pred_classes.numpy().tolist()

    return JSONResponse(content={
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "class_names": [CLASS_NAMES[cls_id] for cls_id in classes] # Añadir nombres de clase para la conveniencia del frontend
    })