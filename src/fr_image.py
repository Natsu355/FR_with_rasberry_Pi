from picamera2 import Picamera2
import libcamera
import cv2
import time

imp_s_time = time.time()
from deepface import DeepFace
imp_e_time = time.time()
print("----- Import Time ------ ", imp_e_time-imp_s_time)

from utils.constants import *
from utils.db_utils import VectorDB


vec_db = VectorDB()


def capture_img():
    """
    captures image using Pi camera and stores it in Result directory
    :return : None
    """
    pica = Picamera2()
    config = pica.create_still_configuration(
                                           main={"size":(920,1380), 'format': 'RGB888'},
                                           raw = None,
                                           transform = libcamera.Transform(hflip=True,vflip=True))
    pica.configure(config)
    print("------- Capturing Image -------- ")
    time.sleep(2)
    pica.start()
    pica.capture_file(tool_directories['RESULT_DIR']+"Testing.jpg")
    pica.stop()
    
    

def run_inference():
    """
    sends the captured image to the deepface module for inference process and saves the resulting image with detection boxes if present any.
    :return: None
    """
    capture_img()
    match_name=""
    match_score = 0
        
    inp_image = cv2.imread(tool_directories['RESULT_DIR']+"Testing.jpg")
    resized_inp_image = cv2.resize(inp_image, (340,480))
    infer_s_time = time.time()
    inp_embeddings = DeepFace.represent(img_path=resized_inp_image, enforce_detection=False,
                                        model_name=tool_constants["MODEL"],
                                        detector_backend=tool_constants["DETECTOR"])
    infer_e_time = time.time()
    print("--------- Inference Time --------- ", infer_e_time-infer_s_time)
                                        
    
    for embed in inp_embeddings:
        bbox = embed['facial_area']
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = bbox['w'] + x1
        y2 = bbox['h'] + y1
        print(bbox)
        
        if x1==0 or y1==0:
            continue
            
        query_results = vec_db.query_data(embed['embedding'])
        if query_results!=None:
            match_name = query_results['name']
            match_score = query_results['score']
        print("------- match name and score ----- ", match_name, match_score)
        
        resized_inp_image = cv2.rectangle(resized_inp_image,(x1,y1), (x2,y2), (255, 0, 0), 3)
        resized_inp_image = cv2.putText(resized_inp_image,match_name,(x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
    cv2.imwrite(tool_directories['RESULT_DIR']+"Testing_result.jpg", resized_inp_image)
    
    
run_inference()
    