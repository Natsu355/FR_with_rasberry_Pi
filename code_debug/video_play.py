import time

import cv2
from deepface import DeepFace
from pinecone import Pinecone, ServerlessSpec
#from constants import *

#pc = Pinecone(api_key = tool_constants["PINECONE_API"])
#index_name = pc.Index("facerecognition")

index_name = ""
def query_db(embed):
    st = time.time()
    query_response = index_name.query(
        namespace="experimentation",
        vector=embed,
        top_k=1,
        include_values=True,
    )
    et = time.time()
    print("****** IN quering *******", et-st)
    matched_name = query_response['matches'][0]['id']
    matched_score = query_response['matches'][0]['score']
    if matched_score>=0.0:
        print("Results after quering the DB ------>> ", matched_name, matched_score, query_response['usage'])
        return matched_name

cap = cv2.VideoCapture('my.mp4')
c=0
while cap.isOpened():
    c=c+1
    ret,frame = cap.read()
    # Check if the frame was read correctly
    if not ret:
        print("Error: Could not read frame or end of video reached.")
        break

    # Ensure the frame is not empty
    if frame is None:
        print("Warning: Received an empty frame. Skipping...")
        continue
    if c>50:
        continue
    start_t = time.time()
    frame=cv2.resize(frame,(320,460))
    results = DeepFace.represent(img_path=frame, enforce_detection=False,
                             model_name='Facenet512',
                             detector_backend='ssd')
    end_t = time.time()
    print("********** Processing each frame *********** ", end_t-start_t)
    match_name='*'
    for res in results:
        print("-------------> Facial area --> ", res['facial_area'])
        bbox = res['facial_area']
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = bbox['w'] + x1
        y2 = bbox['h'] + y1
        if x1==0 or y1==0:
            continue
        #if c%10!=0:
        #match_name = query_db(res['embedding'])
        frame = cv2.rectangle(frame,(x1,y1), (x2,y2), (255, 0, 0), 4)
        frame = cv2.putText(frame,match_name,(x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imwrite("tmp/seq-"+str(c)+".jpg",frame)
    #cv2.imshow("video capture", frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

cap.release()
#cv2.destroyAllWindows()

