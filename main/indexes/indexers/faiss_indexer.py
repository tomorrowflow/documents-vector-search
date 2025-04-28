import faiss
import numpy as np


class FaissIndexer:
    def __init__(self, name, embedder, serialized_index=None):
        self.name = name
        self.embedder = embedder
        if serialized_index is not None:
            self.faiss_index = faiss.deserialize_index(serialized_index)
        else:
            self.faiss_index = faiss.IndexFlatL2(embedder.get_number_of_dimensions())

    def get_name(self):
        return self.name
    
    def embed_texts(self, text):
        return self.embedder.embed(text)

    def index_embeddings(self, embeddings):
        self.faiss_index.add(embeddings)

    def index_texts(self, texts):
        self.faiss_index.add(self.embed_texts(texts))

    def serialize(self):
        return faiss.serialize_index(self.faiss_index)

    def search(self, text, number_of_results=10):
        return self.faiss_index.search(np.expand_dims(self.embedder.embed(text), axis=0), number_of_results)