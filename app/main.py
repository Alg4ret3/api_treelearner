import os
import io
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

app = FastAPI()

# Descargar modelo si no existe
def descargar_modelo():
    model_path = os.path.join(os.path.dirname(__file__), "model_final.pth")
    if not os.path.exists(model_path):
        print("Descargando el modelo...")
        url = "https://drive.google.com/uc?export=download&id=1kO8C4-YSRAp7Y9Yx-xcZWYW3erwa-TWi"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Modelo descargado exitosamente")
    else:
        print("El modelo ya existe")

descargar_modelo()

# Configuración de Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)[:, :, ::-1]  # Convertir de RGB a BGR para OpenCV

    outputs = predictor(image_np)
    instances = outputs["instances"]

    boxes = instances.pred_boxes.tensor.cpu().numpy().tolist()
    scores = instances.scores.cpu().numpy().tolist()
    classes = instances.pred_classes.cpu().numpy().tolist()

    return JSONResponse(content={
        "boxes": boxes,
        "scores": scores,
        "classes": classes
    })
