import time
from .constants import *
from pinecone import Pinecone

class VectorDB:

    def __init__(self):
        self.index_name = None
        self.pc = None
        self.initialize()

    def initialize(self):
        self.pc = Pinecone(api_key=tool_constants["PINECONE_API"])
        self.index_name = self.pc.Index("facerecognition")


    def upsert_data(self,inp_embeddings, id_name):
        """

        :param inp_embeddings:
        :param id_name:
        :return:
        """
        upsert_response = self.index_name.upsert(
            vectors=[
                {
                    "id": id_name,
                    "values": inp_embeddings[0]['embedding']
                }
            ],
            namespace=tool_constants['NAMESPACE']
        )
        print(upsert_response)


    def delete_data(self,del_id):
        """

        :param del_id:
        :return:
        """
        del_response = self.index_name.delete(ids=del_id, namespace=tool_constants['NAMESPACE'])
        print(del_response)


    def query_data(self,embedding):
        """

        :param embedding:
        :return:
        """
        st = time.time()
        query_response = self.index_name.query(
            namespace="experimentation",
            vector=embedding,
            top_k=1,
            include_values=False,
        )
        et = time.time()
        print("Time taken for Querying the DB -------> ",et-st)
        matched_name = query_response['matches'][0]['id']
        matched_score = query_response['matches'][0]['score']
        if matched_score>=0.2:
            return {'name':matched_name,
                    'score':matched_score
                    }
        else:
            return None


    def fetch_data(self):
        """

        :return:
        """
        all_ids = self.list_all_ids()
        fetch_response = self.index_name.fetch(ids=all_ids, namespace=tool_constants['NAMESPACE'])
        print(fetch_response)


    def list_all_ids(self):
        """

        :return:
        """
        all_ids=[]
        for ids in self.index_name.list(namespace=tool_constants['NAMESPACE']):
            all_ids.append(ids)
        return all_ids[0]


# vecDb = VectorDB()
# vecDb.initialize()
