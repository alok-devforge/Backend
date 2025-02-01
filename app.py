from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from uuid import uuid4 
from ultralytics import YOLO
import logging
import glob
import ultralytics  # Added import to check version
from PIL import Image  # Import Pillow for image handling

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider specifying trusted origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
try:
    logger.info("Loading YOLOv8 model...")
    logger.info(f"Ultralytics version: {ultralytics.__version__}")  # Log the version
    model = YOLO('models/model.pt')  # Ensure 'model.pt' is a YOLOv8 model
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        logger.info("Received a request for detection.")
        
        # Optional: Validate the uploaded file
        # await validate_image(file)

        img = await file.read()
        img_filename = f"{uuid4()}.jpg"
        upload_path = os.path.join("images", "original", img_filename)
        annotated_dir = os.path.join(os.path.dirname(__file__), "images", "annotated")

        # Ensure upload and annotation directories exist
        os.makedirs(os.path.join("images", "original"), exist_ok=True)
        os.makedirs(annotated_dir, exist_ok=True)
        logger.info(f"Saving uploaded image to {upload_path}.")

        with open(upload_path, "wb") as f:
            f.write(img)

        logger.info("Performing detection...")
        results = model(upload_path)[0]

        if not results:
            raise HTTPException(status_code=500, detail="No results from model.")

        # Check if there are any detections
        if (len(results.boxes) == 0):
            logger.info("No detections found in the image.")
            return FileResponse(
                upload_path,
                media_type='image/jpeg',
                headers={"X-Detection": "No detections found."}
            )

        # Generate annotated image using results.plot()
        annotated_image = results.plot()
        annotated_image = Image.fromarray(annotated_image)
        annotated_path = os.path.join(annotated_dir, img_filename)
        annotated_image.save(annotated_path)
        logger.info(f"Annotated image saved to {annotated_path}.")

        # Return FileResponse with correct media type
        return FileResponse(annotated_path, media_type='image/jpeg')
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
