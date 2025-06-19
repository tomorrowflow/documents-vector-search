import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

class QdrantIndexer:
    """
    QdrantIndexer class for vector storage using Qdrant database.

    This class implements the same interface as the FAISS indexer, allowing for
    easy swapping of the vector storage solution. It provides methods for
    adding vectors, searching vectors, and deleting vectors.
    """

    def __init__(self, name, embedder, host="localhost", port=6333, collection_name=None):
        """
        Initialize the QdrantIndexer.

        Args:
            name (str): Name of the indexer.
            embedder: Embedding model to generate vector representations.
            host (str): Qdrant host address.
            port (int): Qdrant port number.
            collection_name (str, optional): Name of the Qdrant collection.
                                           If not provided, defaults to the indexer name.
        """
        self.name = name
        self.embedder = embedder
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name or name

        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        if self.collection_name not in collection_names:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vector_size=embedder.get_number_of_dimensions()
            )

    def get_name(self):
        """Get the name of the indexer."""
        return self.name

    def index_texts(self, ids, texts):
        """
        Add vectors to the Qdrant collection.

        Args:
            ids (list): List of IDs for the vectors.
            texts (list): List of texts to be embedded and indexed.
        """
        vectors = self.embedder.embed(texts)
        points = [
            PointStruct(id=id_, vector=vector.tolist())
            for id_, vector in zip(ids, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def remove_ids(self, ids):
        """
        Remove vectors by their IDs from the Qdrant collection.

        Args:
            ids (list): List of IDs to be removed.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"points": ids}
        )

    def serialize(self):
        """
        Serialize the indexer state.

        For Qdrant, this returns the collection name as a simple identifier.
        """
        return self.collection_name

    def search(self, text, number_of_results=10):
        """
        Search for similar vectors in the Qdrant collection.

        Args:
            text (str): Text to be embedded and used for search.
            number_of_results (int): Number of top results to return.

        Returns:
            tuple: (distances, ids) where distances are the similarity scores
                   and ids are the IDs of the matching vectors.
        """
        vector = self.embedder.embed([text])[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=number_of_results
        )

        distances = [result.score for result in search_result]
        ids = [result.id for result in search_result]

        return np.array(distances), np.array(ids)

    def get_size(self):
        """
        Get the number of vectors in the Qdrant collection.

        Returns:
            int: Number of vectors in the collection.
        """
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.vectors_count