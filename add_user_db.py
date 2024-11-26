import os
import cv2
from deepface import DeepFace
from utils.db_utils import VectorDB
from utils.constants import tool_constants


#vecDb = VectorDB()




def check_dir_exists():
    """

    :return:
    """
    print(tool_constants)
    dir_name = "DB_Data"
    curr_dir = os.getcwd()
    target_dir = os.path.join(curr_dir,dir_name)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"----- Directory '{dir_name}' created at {curr_dir} -----")
    else:
        print(f"------ Directory '{dir_name}' already exists in {curr_dir} ------")


def register_user():
    """

    :return:
    """
    check_dir_exists()
    inp_embeddings = DeepFace.represent(img_path='',
                                        enforce_detection=False,
                                        model_name=tool_constants['MODEL'],
                                        detector_backend=tool_constants['DETECTOR'])
    #vecDb.upsert_data(inp_embeddings, "")



check_dir_exists()