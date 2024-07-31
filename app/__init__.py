from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime
import sys
from loguru import logger
from app import routers
from app.detection import DetectionController
# from app.config import settings
# import logging
# from app.logging_config import LOGGING_CONFIG

def create_app():
    # ============================= Logger =============================
    # logging.config.dictConfig(LOGGING_CONFIG)
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level=10,
    )
    # logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

    # ============================= FastAPI Setup =============================
    description = """
    API YOLOv8 menggunakan Docker. 🚀   
    """
    app = FastAPI(
            title="FASTAPI YOLOv8 (by Zaklabs)",
            description=description,
            version="0.0.1",
            terms_of_service=None,
            contact={'name':'Zaki Fuadi','email':'zfuadi81@gmail.com'},
            license_info=None,
            swagger_ui_parameters={"defaultModelsExpandDepth": -1}
        )
    

    # This function is needed if you want to allow client requests 
    # from specific domains (specified in the origins argument) 
    # to access resources from the FastAPI server, 
    # and the client and server are hosted on different domains.
    origins = [
        "http://localhost",
        "http://localhost:8008",
        "*"
    ]
    
    # Add the middleware of App
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Root endpoint
    @app.get('/', tags=["API Docs"])
    def root():
        return {
        "server" : "Zaklabs - FASTAPI YOLOv8",
        "title" : "FASTAPI YOLOv8 (by Zaklabs)",
        "description":"Modul Engine FastApi YOLOv8",
        "version":"0.0.1",
        "contact": {'name':'Zaki Fuadi','email':'zfuadi81@gmail.com'},
        "date" : datetime.datetime.now(),
        "sys.version": sys.version,
        "Tensorflow.__version__": "n/a",
        "nvidia-smi": "no cuda"
    }
    
    # Initialize and load the models
    # DetectionController.load_model("./model/best.pt")
    
    # Include the route
    app.include_router(routers.router, prefix="/api/v1", tags=["API Docs"])  
    
    return app