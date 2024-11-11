import time

from deepface import DeepFace
from pinecone import Pinecone, ServerlessSpec

from constants import *

pc = Pinecone(api_key = tool_constants["PINECONE_API"])
index_name = pc.Index("facerecognition")


# suri_img = DeepFace.represent(img_path='suri.jpg', model_name='Facenet512', detector_backend='retinaface')
# harish_img = DeepFace.represent(img_path='harish.jpg', model_name='Facenet512', detector_backend='retinaface')
# gowtham_img = DeepFace.represent(img_path='gowtham.jpg', model_name='Facenet512', detector_backend='retinaface')
#
# # print("suri embeddings : ********** ", suri_img[0]['embedding'])
#
#
# upsert_response = index_name.upsert(
#     vectors=[
#         {
#             "id": "suri",
#             "values": suri_img[0]['embedding']
#         },
#         {
#             "id": "gowtham",
#             "values": gowtham_img[0]['embedding']
#         },
#         {
#             "id": "harish",
#             "values": harish_img[0]['embedding']
#         }
#     ],
#     namespace="experimentation"
# )

# print("Response ----> ", upsert_response)

def get_embeddings():
    inp_embeddings = DeepFace.represent(img_path='suri.jpg',
                                        model_name=tool_constants["MODEL"],
                                        detector_backend=tool_constants["DETECTOR"])
    return inp_embeddings[0]['embedding']

def query_db():

    query_response = index_name.query(
        namespace="experimentation",
        vector=get_embeddings(),
        top_k=1,
        include_values=True,
    )

    matched_name = query_response['matches'][0]['id']
    matched_score = query_response['matches'][0]['score']

    print("Results after quering the DB ------>> ", matched_name, matched_score)

start_time = time.time()
query_db()
end_time = time.time()

print("total_time ---------------------> ", end_time-start_time)