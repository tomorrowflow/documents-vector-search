import json
import numpy as np
from datetime import datetime, timezone

from ..utils.progress_bar import wrap_generator_with_progress_bar
from ..utils.progress_bar import wrap_iterator_with_progress_bar
from ..utils.performance import log_execution_duration

class DocumentCollectionCreator:
    def __init__(self,
                 collection_name,
                 document_reader,
                 document_converter,
                 document_indexers,
                 persister,
                 indexing_batch_size=500_000,
                 use_embedding_cache=True):
        self.collection_name = collection_name
        self.document_reader = document_reader
        self.document_convertor = document_converter
        self.document_indexers = document_indexers
        self.persister = persister
        self.indexing_batch_size = indexing_batch_size
        self.use_embedding_cache = use_embedding_cache

    def create_collection(self):
        self.persister.remove_folder(self.collection_name)
        self.persister.create_folder(self.collection_name)

        self.__prcoess_collection()

    def update_collection(self):
        if not self.persister.is_path_exists(self.collection_name):
            raise Exception(f"Collection {self.collection_name} does not exist. Please create it first.")

        existing_manifest = json.loads(self.persister.read_text_file(self.__build_manifest_path()))

        self.__prcoess_collection(existing_manifest)

    def __prcoess_collection(self, existing_manifest=None):
        update_time = datetime.now(timezone.utc)
        log_execution_duration(lambda: self.__read_documents(),
                               identifier=f"Reading documents for collection: {self.collection_name}")
    
        last_modified_document_time, number_of_documents, number_of_chunks = log_execution_duration(lambda: self.__index_documents(),
                                                                                                    identifier=f"Indexing documents for collection: {self.collection_name}")
        
        self.__create_manifest_file(existing_manifest, 
                                    update_time, 
                                    last_modified_document_time, 
                                    number_of_documents, 
                                    number_of_chunks)


    def __read_documents(self):
        for document in wrap_generator_with_progress_bar(self.document_reader.read_all_documents(), self.document_reader.get_number_of_documents()):
            for converted_document in self.document_convertor.convert(document):
                document_path = f"{self.collection_name}/documents/{converted_document['id']}.json"
                self.persister.save_text_file(json.dumps(converted_document, indent=4), document_path)

    def __index_documents(self):
        index_mapping = {}
        index_item_id = 0

        last_modified_document_time = None
        number_of_documents = 0

        for document_file_names in wrap_iterator_with_progress_bar(self.__get_file_name_batches()):
            items_to_index = []
            for document_file_name in document_file_names:
                document_path = f"{self.collection_name}/documents/{document_file_name}"
                converted_document = json.loads(self.persister.read_text_file(document_path))

                modified_document_time = datetime.fromisoformat(converted_document["modifiedTime"])
                if last_modified_document_time is None or last_modified_document_time < modified_document_time:
                    last_modified_document_time = modified_document_time
                number_of_documents += 1

                for chunk_number in range(0, len(converted_document["chunks"])):
                    items_to_index.append(converted_document["chunks"][chunk_number]["indexedData"])

                    index_mapping[index_item_id] = {
                        "documentId": converted_document["id"],
                        "documentUrl": converted_document["url"],
                        "documentPath": document_path,
                        "chunkNumber": chunk_number
                    }

                    index_item_id += 1

            for indexer in self.document_indexers:
                indexer.index_embeddings(self.embed_texts(indexer, items_to_index))

        for indexer in self.document_indexers:
            index_base_path = self.__build_index_base_path(indexer)

            self.persister.save_bin_file(indexer.serialize(), f"{index_base_path}/indexer")
            self.persister.save_text_file(json.dumps(index_mapping, indent=4), f"{index_base_path}/index_document_mapping.json")
        
        return last_modified_document_time, number_of_documents, index_item_id + 1


    def embed_texts(self, indexer, texts):
        if not self.use_embedding_cache:
            return indexer.embed_texts(texts)

        index_base_path = self.__build_index_base_path(indexer)
        cache = self.__read_embeddings_cache(index_base_path)

        text_embeddings = []
        new_text_values = []
        new_text_indexes = []
        for text_index in range(0, len(texts)):
            text = texts[text_index]
            if text in cache:
                text_embeddings.append(cache[text])
            else:
                text_embeddings.append(None)
                new_text_values.append(text)
                new_text_indexes.append(text_index)

        if len(new_text_values) != 0:
            print(f"Embedding {len(new_text_values)} new texts for indexer {indexer.get_name()}...")

            new_embeddings = indexer.embed_texts(new_text_values)

            for index in range(0, len(new_text_values)):
                text = new_text_values[index]
                text_embeddings[new_text_indexes[index]] = new_embeddings[index]
                cache[text] = new_embeddings[index]
        
            self.__save_embeddings_cache(index_base_path, cache)

        return np.array(text_embeddings)

    def __save_embeddings_cache(self, index_base_path, cache):
        serializable_cache = {}
        for key, embedding in cache.items():
            if hasattr(embedding, 'tolist'):
                serializable_cache[key] = embedding.tolist()
            else:
                serializable_cache[key] = embedding

        self.persister.save_text_file(json.dumps(serializable_cache), f"{index_base_path}/embeddings_cache.json")

    def __read_embeddings_cache(self, index_base_path):
        cache_path = f"{index_base_path}/embeddings_cache.json"
        if self.persister.is_path_exists(cache_path):
            return json.loads(self.persister.read_text_file(cache_path))
        
        print(f"Embeddings cache not present for indexer {index_base_path}. Creating new one.")
        return {}

    def __build_index_base_path(self, indexer):
        return f"{self.collection_name}/indexes/{indexer.get_name()}"

    def __get_file_name_batches(self):
        file_names = self.persister.read_folder_files(f"{self.collection_name}/documents")

        return [file_names[i:i + self.indexing_batch_size] for i in range(0, len(file_names), self.indexing_batch_size)]

    def __create_manifest_file(self, 
                               existing_manifest, 
                               update_time, 
                               last_modified_document_time, 
                               number_of_documents, 
                               number_of_chunks):
        manifest_content = self.__create_manifest_content(existing_manifest, 
                                                          update_time, 
                                                          last_modified_document_time, 
                                                          number_of_documents, 
                                                          number_of_chunks)

        self.persister.save_text_file(json.dumps(manifest_content, indent=4), self.__build_manifest_path())

    def __build_manifest_path(self):
        return f"{self.collection_name}/manifest.json"

    def __create_manifest_content(self, existing_manifest, update_time, last_modified_document_time, number_of_documents, number_of_chunks):
        if existing_manifest:
            return { **existing_manifest,
                "updatedTime": update_time.isoformat(),
                "lastModifiedDocumentTime": last_modified_document_time.isoformat(),
                "numberOfDocuments": number_of_documents,
                "numberOfChunks": number_of_chunks,
            }

        return {
            "collectionName": self.collection_name,
            "updatedTime": update_time.isoformat(),
            "lastModifiedDocumentTime": last_modified_document_time.isoformat(),
            "numberOfDocuments": number_of_documents,
            "numberOfChunks": number_of_chunks,
            "reader": self.document_reader.get_reader_details(),
            "indexers": [{ "name": indexer.get_name() } for indexer in self.document_indexers],
        }
