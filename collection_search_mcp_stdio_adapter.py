import json
import argparse

from mcp.server.fastmcp import FastMCP

from main.persisters.disk_persister import DiskPersister
from main.indexes.indexer_factory import load_indexer
from main.core.documents_collection_searcher import DocumentCollectionSearcher
from main.utils.performance import log_execution_duration

mcp = FastMCP("documents-search")

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=True, help="collaction name (will be used as root folder name)")
ap.add_argument("-index", "--index", required=False, default="indexer_FAISS_IndexFlatL2__embeddings_all-MiniLM-L6-v2", help="index that will be used for search")
args = vars(ap.parse_args())

disk_persister = DiskPersister(base_path="./data/collections")

indexer = load_indexer(args['index'], args['collection'], disk_persister)

searcher = DocumentCollectionSearcher(collection_name=args['collection'], 
                           indexer=indexer, 
                           persister=disk_persister)

@mcp.tool(name=f"search_{args['collection']}", description="Search documents in the collection")
def search_documents(query: str) -> str:
    search_results = searcher.search(query, include_all_chunks_content=True)

    return json.dumps(search_results, indent=4)
    

if __name__ == "__main__":
    mcp.run(transport='stdio')