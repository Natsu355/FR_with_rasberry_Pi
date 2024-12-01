import time
from .constants import *
from pinecone import Pinecone

class VectorDB:

    def __init__(self):
        self.index_name = None
        self.pc = None
        self.initialize()

    def initialize(self):
        """
        Initializes the Pinecone Database instance using API key
        """
        self.pc = Pinecone(api_key=tool_constants["PINECONE_API"])
        self.index_name = self.pc.Index(tool_constants["PINECONE_INDEX"])


    def upsert_data(self,inp_embeddings, id_name):
        """
        Inserts facial features of a person along with the name into Pinecone database
        :param inp_embeddings: facial features
        :param id_name: name of person
        :return: None
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
        deletes the entry in Pinecone DB using name of the person
        :param del_id: name of person stored in DB
        :return: None
        """
        del_response = self.index_name.delete(ids=del_id, namespace=tool_constants['NAMESPACE'])
        print(del_response)



    def query_data(self,embedding):
        """
        Searches the DB using facial features and returns the name,score if there is a match otherwise None 
        :param embedding:
        :return: {name, score}
        """
        st = time.time()
        query_response = self.index_name.query(
            namespace=tool_constants['NAMESPACE'],
            vector=embedding,
            top_k=1,
            include_values=False,
        )
        et = time.time()
        print("Time taken for Querying the DB -------> ",et-st)
        matched_name = query_response['matches'][0]['id']
        matched_score = query_response['matches'][0]['score']
        if matched_score>=0.3:
            return {'name':matched_name,
                    'score':matched_score
                    }
        else:
            return None



    def fetch_data(self):
        """
        Returns the name and facial features of a person queryed using the id
        :return: None
        """
        all_ids = self.list_all_ids()
        fetch_response = self.index_name.fetch(ids=all_ids, namespace=tool_constants['NAMESPACE'])
        print(fetch_response)



    def list_all_ids(self):
        """
        Returns all the Id's of entries stored in a Namespace in DB.
        :return: Id's
        """
        all_ids=[]
        for ids in self.index_name.list(namespace=tool_constants['NAMESPACE']):
            all_ids.append(ids)
        return all_ids[0]


#vecDb = VectorDB()
#print(vecDb.list_all_ids())

