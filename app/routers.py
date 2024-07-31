from fastapi import APIRouter, status, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from app.detection import DetectionController

router = APIRouter()

@router.get("/", include_in_schema=False, tags=[""], status_code=status.HTTP_200_OK)
def docs():
    return RedirectResponse("/docs")

@router.get("/healthcheck", tags=["API Docs"], status_code=status.HTTP_200_OK)
async def perform_healthcheck():
    return await DetectionController.get_status_ultralytics()
    
@router.post("/detection", tags=["API Docs"], status_code=status.HTTP_200_OK)
async def upload_image(file: UploadFile = File(...)):
    result = await DetectionController.process_image(file)
    return result

# @router.post("/detection2", tags=["API Docs"], status_code=status.HTTP_200_OK)
# async def upload_image(file: UploadFile = File(...)):
#     result = await DetectionController.runner(file)
#     return result