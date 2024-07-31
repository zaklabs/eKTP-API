import ultralytics
from ultralytics import YOLO
from fastapi import UploadFile
from PIL import Image
import io
from app import utils
from skimage.transform import resize
import numpy as np
import cv2
from fastapi.responses import StreamingResponse
import time
import base64
import json
from fastapi.responses import JSONResponse
from starlette import status
import threading
from loguru import logger
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
# import logging
# logger = logging.getLogger(__name__)

class DetectionController:
    
    detection_model = None
    model_loading_error = None
    # executor = ThreadPoolExecutor(max_workers=10)

    @classmethod
    def load_model(cls, model_path):
        try:
            cls.detection_model = YOLO(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            cls.model_loading_error = str(e)
            logger.error(f"Error loading model: {cls.model_loading_error}")
   
    @staticmethod
    async def get_status_ultralytics():
        # model = str(DetectionController.detection_model) if DetectionController.detection_model else "Model not loaded"
        response_json = {'healthcheck': 'Everything OK!', 
                "Ultralytics Version": ultralytics.__version__,
                "Model": "YOLOv8"}
        return JSONResponse(content=response_json, status_code=status.HTTP_200_OK)
    
    # Define the scale_image function
    def scale_image(image, orig_shape):
        return resize(image, orig_shape, mode='reflect', anti_aliasing=True)
    
    # Define the finding poin function
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    # Define the transform point to warped image function
    def four_point_transform(image, pts):
        rect = DetectionController.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    # url_list = ['http://10.81.84.211:8003/healthcheck']
 
    # def download_file(url, file_name):
    #     try:
    #         html = requests.get(url, stream=True)
    #         # open(f'{file_name}.json', 'wb').write(html.content)
    #         return html.status_code
    #     except requests.exceptions.RequestException as e:
    #         return e
    
    # @staticmethod
    # def process_image_async_test(file: UploadFile):
    #     current_thread = threading.current_thread().name
    #     logger.debug(f"Current thread: {current_thread}")
    #     if DetectionController.model_loading_error:
    #         logger.error(f"Model loading error: {DetectionController.model_loading_error}")
    #         response_json = {
    #             'error': 'Model not loaded',
    #             'details': DetectionController.model_loading_error
    #         }
    #         return JSONResponse(content=response_json, status_code=status.HTTP_400_BAD_REQUEST)

    #     try:
    #         start_time = time.time()
    #         logger.debug("Started processing image.")
            
    #         image_data = file
    #         input_image = utils.get_image_from_bytes(image_data)
    #         end_time = time.time()
    #         execution_time = end_time - start_time

    #         response_json = {
    #             "execution_time": execution_time,
    #             "status": "success",
    #         }
            
    #         logger.info("Processing completed successfully.")
    #         return response_json
    #     except Exception as e:
    #         logger.error(f"Error processing image: {e}")
    #         response_json = {
    #             "error": str(e)
    #         }
    #         return response_json
   
    # @staticmethod
    # async def runner(file: UploadFile):
    #     threads= []
    #     with ThreadPoolExecutor(max_workers=20) as executor:
    #         image_data = await file.read()
    #         # for url in DetectionController.url_list:
    #         #     file_name = uuid.uuid1()
    #             # threads.append(executor.submit(DetectionController.download_file, url, file_name))
    #         # threads.append(executor.submit(DetectionController.process_image_sync, image_data))
    #         threads.append(executor.submit(DetectionController.process_image_async_test, image_data))
                
    #         for task in as_completed(threads):
    #             # print(task.result())
    #             return JSONResponse(content=task.result())
            
    @staticmethod
    async def process_image(file: UploadFile):
        threads= []
        with ThreadPoolExecutor(max_workers=20) as executor:
            image_data = await file.read()
            threads.append(executor.submit(DetectionController.process_image_async, image_data))
               
            for task in as_completed(threads):
                # print(task.result())
                return JSONResponse(content=task.result())      
                  
    # @staticmethod
    # async def process_image(file: UploadFile):
    #     loop = asyncio.get_event_loop()
    #     response = await loop.run_in_executor(DetectionController.executor, DetectionController.process_image_sync, file)
    #     if 'error' in response:
    #         return JSONResponse(content=response, status_code=status.HTTP_400_BAD_REQUEST)
    #     return JSONResponse(content=response)
    
    @staticmethod
    def process_image_async(file: bytes):
        current_thread = threading.current_thread().name
        logger.debug(f"Current thread: {current_thread}")
        if DetectionController.model_loading_error:
            logger.error(f"Model loading error: {DetectionController.model_loading_error}")
            response_json = {
                'error': 'Model not loaded',
                'details': DetectionController.model_loading_error
            }
            return JSONResponse(content=response_json, status_code=status.HTTP_400_BAD_REQUEST)

        try:
            start_time = time.time()
            logger.debug("Started processing image.")
            
            image_data = file
            input_image = utils.get_image_from_bytes(image_data)
            
            result = DetectionController.detection_model.predict(input_image, conf=0.55, save=False, verbose=True)[0]
            logger.debug("Prediction completed.")
            
            if result.masks != None:
                masks_nhw = result.masks.data.cpu().numpy()
                masks_hwn = np.moveaxis(masks_nhw, 0, -1)
                masks = DetectionController.scale_image(masks_hwn, result.masks.orig_shape)
                mask_squeezed = np.squeeze(masks, axis=2)
                binary_mask = (mask_squeezed > 0.8).astype(np.uint8)
                
                contours,_ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cmax = max(contours, key = cv2.contourArea)
                width, height = binary_mask.shape

                new_mask = np.zeros([width, height, 3], dtype=np.uint8)
                cv2.fillPoly(new_mask, pts=[cmax], color=(255,255,255))
                                
                mask = np.zeros(new_mask.shape, dtype=np.uint8)
                gray = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
                blur = cv2.bilateralFilter(gray, 9, 75, 75)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                rect = cv2.minAreaRect(cnts[0])
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.fillPoly(mask, [box], (255, 255, 255))

                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.5, minDistance=150)
                corners = corners.reshape(-1, 2)
                
                input_image_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
                warped_image = DetectionController.four_point_transform(input_image_cv, corners)
                
                _, buffer = cv2.imencode('.jpg', warped_image)
                base64_str = base64.b64encode(buffer).decode('utf-8')
                
                end_time = time.time()
                execution_time = end_time - start_time

                response_json = {
                    "execution_time": execution_time,
                    "status": "success",
                    "warped_image": base64_str
                }
                
                logger.info("Processing completed successfully.")
                return response_json
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            response_json = {
                "error": str(e)
            }
            return response_json
    
    # === NON THREADING ===        
    # @staticmethod
    # async def process_image(file: UploadFile):
    #     current_thread = threading.current_thread().name
    #     logger.debug(f"Current thread: {current_thread}")
    #     if DetectionController.model_loading_error:
    #         logger.error(f"Model loading error: {DetectionController.model_loading_error}")
    #         response_json = {
    #             'error': 'Model not loaded',
    #             'details': DetectionController.model_loading_error
    #         }
    #         return JSONResponse(content=response_json, status_code=status.HTTP_400_BAD_REQUEST)

    #     try:
    #         # Measure start time
    #         start_time = time.time()
    #         logger.debug("Started processing image.")
            
    #         # get image from bytes
    #         image_data = await file.read()
    #         input_image = utils.get_image_from_bytes(image_data)
            
    #         # predict the source image
    #         result = DetectionController.detection_model.predict(input_image, conf=0.55, save=False, verbose=True)[0]
    #         logger.debug("Prediction completed.")
            
    #         if result.masks != None:
    #             # detection
    #             # result.boxes.xyxy   # box with xyxy format, (N, 4)
    #             # cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    #             # probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    #             # boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)
    #             # segmentation
    #             masks_nhw = result.masks.data.cpu().numpy()     # masks, (N, H, W)
    #             masks_hwn = np.moveaxis(masks_nhw, 0, -1) # masks, (H, W, N)
    #             # rescale masks to original image
    #             masks = DetectionController.scale_image(masks_hwn, result.masks.orig_shape)
    #             # masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    #             # Convert mask to single channel image binary
    #             mask_squeezed = np.squeeze(masks, axis=2)
    #             binary_mask = (mask_squeezed > 0.8).astype(np.uint8)
                
            
    #             # find maximum contour
    #             contours,_ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #             cmax = max(contours, key = cv2.contourArea)
    #             width, height = binary_mask.shape

    #             #fill maximum contour and draw   
    #             new_mask = np.zeros( [width, height, 3],dtype=np.uint8 )
    #             cv2.fillPoly(new_mask, pts =[cmax], color=(255,255,255))
                                
    #             mask = np.zeros(new_mask.shape, dtype=np.uint8)
    #             gray = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
    #             blur = cv2.bilateralFilter(gray,9,75,75)
    #             thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #             # Find distorted rectangle contour and draw onto a mask
    #             cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #             rect = cv2.minAreaRect(cnts[0])
    #             box = cv2.boxPoints(rect)
    #             box = np.intp(box)
    #             cv2.fillPoly(mask, [box], (255,255,255))

    #             # Find corners on the mask
    #             mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #             corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.5, minDistance=150)
                
    #             # reshape coorners values
    #             corners = corners.reshape(-1, 2)
                
    #             # Change source image to cvtColor
    #             input_image_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
                
    #             # Warp the image
    #             warped_image = DetectionController.four_point_transform(input_image_cv, corners)
                
    #             # # Convert binary mask to image
    #             # output_img = Image.fromarray(warped_image)
            
    #         # # Convert warped image to byte array
    #         # _, buffer = cv2.imencode('.jpg', warped_image)
    #         # byte_arr = io.BytesIO(buffer)
    #         # return StreamingResponse(content=byte_arr, media_type="image/jpeg")
        
    #         # Convert warped image to byte array and then to base64 string
    #         _, buffer = cv2.imencode('.jpg', warped_image)
    #         base64_str = base64.b64encode(buffer).decode('utf-8')
            
    #         # Measure end time
    #         end_time = time.time()
    #         execution_time = end_time - start_time

    #         # Create response JSON with execution time and base64 image
    #         response_json = {
    #             "execution_time": execution_time,
    #             "status": "success",
    #             "warped_image": base64_str
    #         }
            
    #         logger.info("Processing completed successfully.")
    #         return JSONResponse(content=response_json)
    #     except Exception as e:
    #         logger.error(f"Error processing image: {e}")
    #         # Create response JSON with error msg
    #         response_json = {
    #             "error": str(e)
    #         }
    #         # Return json with status code 400
    #         return JSONResponse(content=response_json, status_code=status.HTTP_400_BAD_REQUEST)
    #         # return {'error': str(e)}

# Initialize and load the models
DetectionController.load_model("./model/best.pt")