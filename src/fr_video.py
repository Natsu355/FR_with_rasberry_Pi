from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time
import libcamera
import cv2

imp_s_time = time.time()
from deepface import DeepFace
imp_e_time = time.time()
print("----- Import Time ------ ", imp_e_time-imp_s_time)

from utils.constants import *
from utils.db_utils import VectorDB
from utils.vid_utils import *
from utils.mail_utils import *

vec_db = VectorDB()

first_mail = 0


def run_inference_on_frames(first_mail):
    """
    
    :param: first_mail: flag to determine it is first time sending a detection mail
    :return: None
    """
    cap = cv2.VideoCapture(tool_directories['RESULT_DIR']+'test.mp4')
    c=0
    
    while cap.isOpened():
        match_name = ""
        match_score = 0
        c=c+1
        
        ret,inp_frame = cap.read()
        # Check if the frame was read correctly
        if not ret:
            print("Error: Could not read frame or end of video reached.")
            break

        # Ensure the frame is not empty
        if inp_frame is None:
            print("Warning: Received an empty frame. Skipping...")
            continue
        
        print("--------- Reading Frame {0} -------".format(c))
        resized_inp_image = cv2.resize(inp_frame, (360,480))
        
        if c%2==0:
            ## Skiping alternate frames to decrease computational time
            cv2.imwrite(tool_directories['TEMPORARY_DIR']+"seq-"+str(c)+".jpg", resized_inp_image)
            continue
        
        
        infer_s_time = time.time()
        ## feeding the resized image to the model
        inp_embeddings = DeepFace.represent(img_path=resized_inp_image, enforce_detection=False,
                                        model_name=tool_constants["MODEL"],
                                        detector_backend=tool_constants["DETECTOR"])
        infer_e_time = time.time()
        print("--------- Inference Time for Frame {0}--------- ".format(c), infer_e_time-infer_s_time)
                                        
        
        ## Looping through multiple face detections in a frame
        for embed in inp_embeddings:
            bbox = embed['facial_area']
            print(bbox)
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = bbox['w'] + x1
            y2 = bbox['h'] + y1
            
            ## skiping to next detection if a face is not detected in a frame
            if x1==0 and y1==0:
                continue
            
            ## If a face is detected and no mail is sent prior to these frames
            elif first_mail==0:
                first_mail = 1
            
            ## Querying the DB for similarity matches
            query_results = vec_db.query_data(embed['embedding'])
            if query_results!=None:
                match_name = query_results['name']
                match_score = query_results['score']
                
            print("------- match name ----- ", match_name, match_score)
            ## Draws the bounding box and matched name on the image and saves it to "tmp" directory
            resized_inp_image = cv2.rectangle(resized_inp_image,(x1,y1), (x2,y2), (255, 0, 0), 3)
            resized_inp_image = cv2.putText(resized_inp_image,match_name,(x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imwrite(tool_directories['TEMPORARY_DIR']+"seq-"+str(c)+".jpg", resized_inp_image)
        
        ## checks if it is first time sending a mail, if so it sends the mail with detected facial image
        if first_mail==1:
            print("---------- Trying to send Mail ------------")
            send_email_with_image(tool_directories['TEMPORARY_DIR']+"seq-"+str(c)+".jpg")
            first_mail = 999
        
    cap.release()
    
    print("------- Generating MP4 from the Detected Images ------- ")
    ## converts the all detected frames stored in "tmp" dir to .mp4 format using ffmpeg tool.
    print(images_to_mp4(tool_directories['TEMPORARY_DIR'], tool_directories['RESULT_DIR']+'test_results.mp4'))



def capture_video():
    """
    records video for 10 seconds and saves it to 'results' directory
    :return: None
    """
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (820,1280), 'format': 'RGB888'},
                                                     raw = None,
                                                     transform = libcamera.Transform(hflip=True,vflip=True))
    picam2.configure(video_config)

    encoder = H264Encoder(10000000)
    output = FfmpegOutput(tool_directories['RESULT_DIR']+'test.mp4')
    print("------- Starting Camera -------")
    picam2.start_recording(encoder, output)
    time.sleep(10)
    picam2.stop_recording()
    


#tot_s_time = time.time()

capture_video()
run_inference_on_frames(first_mail)

#tot_e_time = time.time()
#print("############## TOTAL FUNCTIONALITY TIME ################ ", tot_e_time-tot_s_time)

