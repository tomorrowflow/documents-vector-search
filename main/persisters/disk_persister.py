import os
import shutil
import pickle
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

class DiskPersister:
    def __init__(self, base_path, qdrant_host="localhost", qdrant_port=6333):
        self.base_path = base_path
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def save_text_file(self, data, file_path):
        path = os.path.join(self.base_path, file_path)

        self.__make_sure_path_exists(path)

        with open(path, 'w', encoding="utf-8") as file:
            file.write(data)

    def read_text_file(self, file_path):
        path = os.path.join(self.base_path, file_path)

        with open(path, 'r', encoding="utf-8") as file:
            return file.read()

    def save_bin_file(self, data, file_path):
        path = os.path.join(self.base_path, file_path)

        self.__make_sure_path_exists(path)

        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def read_bin_file(self, file_path):
        path = os.path.join(self.base_path, file_path)

        with open(path, 'rb') as file:
            return pickle.load(file)

    def create_folder(self, folder_name):
        directory_path = os.path.join(self.base_path, folder_name)
        os.makedirs(directory_path)

    def remove_folder(self, folder_name):
        directory_path = os.path.join(self.base_path, folder_name)

        if os.path.exists(directory_path):
            shutil.rmtree(directory_path, ignore_errors=True)

    def remove_file(self, file_path):
        path = os.path.join(self.base_path, file_path)

        if os.path.exists(path):
            os.remove(path)

    def is_path_exists(self, relative_path):
        path = os.path.join(self.base_path, relative_path)
        return os.path.exists(path)

    def read_folder_files(self, relative_path):
        path = os.path.join(self.base_path, relative_path)
        files = []
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                files.append(os.path.relpath(os.path.join(root, filename), path))
        return files

    def __make_sure_path_exists(self, path):
        directory_path = os.path.dirname(path)

        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def save_collection_to_qdrant(self, collection_name, vectors, ids):
        """
        Save a collection of vectors to Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection.
            vectors (list): List of vectors to be saved.
            ids (list): List of IDs corresponding to the vectors.
        """
        points = [
            PointStruct(id=id_, vector=vector.tolist())
            for id_, vector in zip(ids, vectors)
        ]
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

    def load_collection_from_qdrant(self, collection_name):
        """
        Load a collection from Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection to load.

        Returns:
            tuple: (vectors, ids) where vectors are the loaded vectors
                   and ids are the corresponding IDs.
        """
        collection_info = self.qdrant_client.get_collection(collection_name)
        if collection_info.vectors_count == 0:
            return [], []

        # Get all points from the collection
        scroll_result = self.qdrant_client.scroll(
            collection_name=collection_name,
            limit=500  # Adjust based on expected collection size
        )

        vectors = []
        ids = []

        for point in scroll_result[0]:
            vectors.append(point.vector)
            ids.append(point.id)

        return vectors, ids
