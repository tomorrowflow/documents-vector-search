import json
import argparse

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import load_indexer
from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.utils.performance import log_execution_duration

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=True, help="collaction name (will be used as root folder name)")
ap.add_argument("-index", "--index", required=False, default="indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2", help="index that will be used for search")
ap.add_argument("-query", "--query", required=True, help="text query for search")
args = vars(ap.parse_args())

disk_persister = DiskPersister(base_path="./data/collections")

indexer = load_indexer(args['index'], args['collection'], disk_persister)

searcher = DocumentCollectionSearcher(collection_name=args['collection'], 
                           indexer=indexer, 
                           persister=disk_persister)

search_result = log_execution_duration(lambda: searcher.search(args['query']),
                                       identifier=f"Searching collection: {args['collection']} by query: {args['query']}")

print(json.dumps(search_result, indent=4))