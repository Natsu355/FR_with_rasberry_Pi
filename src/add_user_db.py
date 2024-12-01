import os
import cv2
import time
import libcamera
from picamera2 import Picamera2
from deepface import DeepFace

from utils.db_utils import VectorDB
from utils.constants import tool_constants


vecDb = VectorDB()

user_name = input("---- Enter the name of User to be registered ---> ")


def check_dir_exists():
    """
    Checks if "DB_data" directory is present or not if not it creates the Directory
    :return: None
    """
    dir_name = "DB_data"
    curr_dir = os.getcwd()
    target_dir = os.path.join(curr_dir,dir_name)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"----- Directory '{dir_name}' created at {curr_dir} -----")
    else:
        print(f"------ Directory '{dir_name}' already exists in {curr_dir} ------")


def register_user():
    """
    Runs the deepface module on the captured image and stores the resulting facial features in the Pinecone DB
    :return: None
    """

    check_dir_exists()
    capture_img(user_name)
    
    infer_s_time = time.time()
    inp_embeddings = DeepFace.represent(img_path="DB_data/"+user_name+".jpg",
                                        enforce_detection=False,
                                        model_name=tool_constants['MODEL'],
                                        detector_backend=tool_constants['DETECTOR'])
    x_pos = inp_embeddings[0]['facial_area']['x']
    y_pos = inp_embeddings[0]['facial_area']['y']
    infer_e_time = time.time()
    print("--------- Inference Time ---------- ", infer_e_time-infer_s_time)
    
    if x_pos==0 and y_pos==0:
        print("Error: Face not detected...Try again!!")
        
    else:
        print(inp_embeddings[0]['facial_area'])
        
        vecDb.upsert_data(inp_embeddings, user_name)


def capture_img(user_name):
    """
    captures the image using Pi camera and stores it in the DB_data 
    :return : None
    """
    pica = Picamera2()
    config = pica.create_still_configuration(
                                           main={"size":(920,1380), 'format': 'RGB888'},
                                           raw = None,
                                           transform = libcamera.Transform(hflip=True,vflip=True))
    pica.configure(config)
    
    pica.start()
    print("------ Capturing Image -------- ")
    time.sleep(2)
    pica.capture_file("DB_data/"+ user_name +".jpg")
    pica.stop()
    
    
register_user()
