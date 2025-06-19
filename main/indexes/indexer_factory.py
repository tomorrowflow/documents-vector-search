from .indexers.faiss_indexer import FaissIndexer
from .indexers.qdrant_indexer import QdrantIndexer
from .embeddings.sentence_embeder import SentenceEmbedder

def create_indexer(indexer_name):
    if indexer_name == "indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2":
        return FaissIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    if indexer_name == "indexer_FAISS_IndexFlatL2__embeddings_all-mpnet-base-v2":
        return FaissIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/all-mpnet-base-v2"))

    if indexer_name == "indexer_FAISS_IndexFlatL2__embeddings_multi-qa-distilbert-cos-v1":
        return FaissIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/multi-qa-distilbert-cos-v1"))

    # Qdrant indexer options
    if indexer_name == "indexer_Qdrant__embeddings_all-MiniLM-L6-v2":
        return QdrantIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    if indexer_name == "indexer_Qdrant__embeddings_all-mpnet-base-v2":
        return QdrantIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/all-mpnet-base-v2"))

    if indexer_name == "indexer_Qdrant__embeddings_multi-qa-distilbert-cos-v1":
        return QdrantIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/multi-qa-distilbert-cos-v1"))

    raise ValueError(f"Unknown indexer name: {indexer_name}")

def load_indexer(indexer_name, collection_name, persister):
    # Check if it's a Qdrant indexer
    if indexer_name.startswith("indexer_Qdrant__"):
        # For Qdrant, we need to check if it's a disk-based or Qdrant-based persistence
        if hasattr(persister, 'qdrant_client'):
            # If persister has Qdrant client, load directly from Qdrant
            vectors, ids = persister.load_collection_from_qdrant(collection_name)
            embedder_model = None

            # Determine the embedder model based on indexer name
            if "all-MiniLM-L6-v2" in indexer_name:
                embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
            elif "all-mpnet-base-v2" in indexer_name:
                embedder_model = "sentence-transformers/all-mpnet-base-v2"
            elif "multi-qa-distilbert-cos-v1" in indexer_name:
                embedder_model = "sentence-transformers/multi-qa-distilbert-cos-v1"

            if embedder_model:
                indexer = QdrantIndexer(indexer_name, SentenceEmbedder(model_name=embedder_model))
                # Manually set the vectors in the indexer (simulated loading)
                # Note: In a real implementation, you would need to handle vector loading appropriately
                return indexer

        # If not using Qdrant persistence or Qdrant client not available, fall back to disk
        # This is just a placeholder - in a real implementation, you would need to handle
        # loading from disk appropriately for Qdrant indexers

    # Handle FAISS indexers
    if indexer_name == "indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2":
        serialized_index = persister.read_bin_file(f"{collection_name}/indexes/{indexer_name}/indexer")
        return FaissIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2"), serialized_index)

    if indexer_name == "indexer_FAISS_IndexFlatL2__embeddings_all-mpnet-base-v2":
        serialized_index = persister.read_bin_file(f"{collection_name}/indexes/{indexer_name}/indexer")
        return FaissIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/all-mpnet-base-v2"), serialized_index)

    if indexer_name == "indexer_FAISS_IndexFlatL2__embeddings_multi-qa-distilbert-cos-v1":
        serialized_index = persister.read_bin_file(f"{collection_name}/indexes/{indexer_name}/indexer")
        return FaissIndexer(indexer_name, SentenceEmbedder(model_name="sentence-transformers/multi-qa-distilbert-cos-v1"), serialized_index)

    raise ValueError(f"Unknown indexer name: {indexer_name}")