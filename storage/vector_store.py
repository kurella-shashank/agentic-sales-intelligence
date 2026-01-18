from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from pathlib import Path
import json


class WeaviateVectorStore:
    """
    Vector store using Weaviate for persistent storage
    Supports dynamic file uploads and incremental indexing
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 weaviate_url: str = "http://localhost:8080"):
        """
        Initialize Weaviate vector store

        Args:
            model_name: HuggingFace model for embeddings
            weaviate_url: Weaviate server URL
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # Connect to Embedded Weaviate (no Docker needed!)
        print("Starting embedded Weaviate...")
        try:
            self.client = weaviate.connect_to_embedded(
                version="1.27.0",
                persistence_data_path="./weaviate_data",
                headers={
                    "X-OpenAI-Api-Key": "none"  # Not using OpenAI
                }
            )
            print("✓ Connected to embedded Weaviate")

            # Create collection if it doesn't exist
            self._create_collection()

        except Exception as e:
            print(f"⚠️  Could not start embedded Weaviate: {e}")
            print("   Make sure weaviate-client>=4.9.3 is installed")
            raise

    def _create_collection(self):
        """Create SalesMetadata collection in Weaviate"""

        try:
            # Check if collection exists
            if self.client.collections.exists("SalesMetadata"):
                print("✓ SalesMetadata collection already exists")
                self.collection = self.client.collections.get("SalesMetadata")
                return

            # Create new collection
            self.collection = self.client.collections.create(
                name="SalesMetadata",
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We provide our own vectors
                properties=[
                    wvc.config.Property(
                        name="content",
                        data_type=wvc.config.DataType.TEXT,
                        description="Rich metadata description of the table"
                    ),
                    wvc.config.Property(
                        name="table_name",
                        data_type=wvc.config.DataType.TEXT,
                        description="Cleaned table name"
                    ),
                    wvc.config.Property(
                        name="original_filename",
                        data_type=wvc.config.DataType.TEXT,
                        description="Original CSV filename"
                    ),
                    wvc.config.Property(
                        name="columns",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        description="List of column names"
                    ),
                    wvc.config.Property(
                        name="row_count",
                        data_type=wvc.config.DataType.INT,
                        description="Number of rows in table"
                    ),
                ]
            )
            print("✓ Created SalesMetadata collection")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents with metadata to Weaviate

        Args:
            documents: List of dicts with 'content' and 'metadata' keys
        """
        print(f"Adding {len(documents)} documents to Weaviate...")

        # Generate embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Insert into Weaviate with proper error handling
        try:
            with self.collection.batch.dynamic() as batch:
                for i, doc in enumerate(documents):
                    properties = {
                        "content": str(doc['content']),
                        "table_name": str(doc['metadata']['table_name']),
                        "original_filename": str(doc['metadata']['original_filename']),
                        "columns": [str(col) for col in doc['metadata']['columns']],  # Ensure all are strings
                        "row_count": int(doc['metadata']['row_count'])  # Ensure it's an int
                    }

                    batch.add_object(
                        properties=properties,
                        vector=embeddings[i].tolist()
                    )

            # Check for failed objects
            if len(self.collection.batch.failed_objects) > 0:
                print(f"⚠️  {len(self.collection.batch.failed_objects)} objects failed to insert")
                for failed in self.collection.batch.failed_objects[:3]:  # Show first 3 errors
                    print(f"   Error: {failed}")
            else:
                print(f"✓ Successfully added {len(documents)} documents to Weaviate")

        except Exception as e:
            print(f"⚠️  Batch insertion error: {e}")
            print(f"   Attempting individual insertion...")

            # Fallback: Insert one by one
            success_count = 0
            for i, doc in enumerate(documents):
                try:
                    properties = {
                        "content": str(doc['content']),
                        "table_name": str(doc['metadata']['table_name']),
                        "original_filename": str(doc['metadata']['original_filename']),
                        "columns": [str(col) for col in doc['metadata']['columns']],
                        "row_count": int(doc['metadata']['row_count'])
                    }

                    self.collection.data.insert(
                        properties=properties,
                        vector=embeddings[i].tolist()
                    )
                    success_count += 1
                except Exception as insert_error:
                    print(f"   Failed to insert document {i}: {insert_error}")

            print(f"✓ Successfully added {success_count}/{len(documents)} documents individually")

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of documents with similarity scores
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Search in Weaviate
        response = self.collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=k,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )

        # Format results
        results = []
        for obj in response.objects:
            # Convert distance to similarity score (1 - distance for cosine)
            similarity_score = 1 - obj.metadata.distance if obj.metadata.distance else 0

            results.append({
                'content': obj.properties['content'],
                'metadata': {
                    'table_name': obj.properties['table_name'],
                    'original_filename': obj.properties['original_filename'],
                    'columns': obj.properties['columns'],
                    'row_count': obj.properties['row_count']
                },
                'similarity_score': similarity_score
            })

        return results

    def delete_by_table_name(self, table_name: str):
        """
        Delete documents for a specific table
        Useful when a file is removed or re-uploaded

        Args:
            table_name: Table name to delete
        """
        self.collection.data.delete_many(
            where=wvc.query.Filter.by_property("table_name").equal(table_name)
        )
        print(f"✓ Deleted documents for table: {table_name}")

    def clear_all(self):
        """Clear all documents from Weaviate"""
        # Delete collection and recreate
        if self.client.collections.exists("SalesMetadata"):
            self.client.collections.delete("SalesMetadata")
            print("✓ Deleted SalesMetadata collection")

        self._create_collection()
        print("✓ Recreated empty SalesMetadata collection")

    def get_all_tables(self) -> List[str]:
        """Get list of all indexed table names"""
        response = self.collection.query.fetch_objects(
            limit=100,
            return_properties=["table_name"]
        )

        return list(set([obj.properties['table_name'] for obj in response.objects]))

    def close(self):
        """Close Weaviate connection"""
        self.client.close()
        print("✓ Closed Weaviate connection")


class MetadataIndexer:
    """
    Creates rich metadata descriptions for tables and indexes them
    Works with both in-memory and Weaviate vector stores
    """

    def __init__(self, db_manager, use_weaviate: bool = True):
        """
        Initialize with DuckDB manager

        Args:
            db_manager: DuckDBManager instance
            use_weaviate: Use Weaviate (True) or in-memory (False)
        """
        self.db_manager = db_manager
        self.use_weaviate = use_weaviate

        if use_weaviate:
            try:
                self.vector_store = WeaviateVectorStore()
            except Exception as e:
                print(f"⚠️  Falling back to in-memory vector store")
                from storage.vector_store_simple import SimpleVectorStore
                self.vector_store = SimpleVectorStore()
                self.use_weaviate = False
        else:
            from storage.vector_store_simple import SimpleVectorStore
            self.vector_store = SimpleVectorStore()

    def index_all_tables(self, force_reindex: bool = False):
        """
        Create metadata documents for all tables and index them

        Args:
            force_reindex: If True, clear existing index and reindex all
        """
        if force_reindex and self.use_weaviate:
            self.vector_store.clear_all()

        documents = []

        for table_name in self.db_manager.get_all_tables():
            info = self.db_manager.get_table_info(table_name)

            # Create rich description
            description = self._create_table_description(info)

            documents.append({
                'content': description,
                'metadata': {
                    'table_name': table_name,
                    'original_filename': info['original_filename'],
                    'columns': info['columns'],
                    'row_count': info['row_count']
                }
            })

        # Index documents
        self.vector_store.add_documents(documents)

        return len(documents)

    def index_single_table(self, table_name: str):
        """
        Index a single table (useful for dynamic uploads)

        Args:
            table_name: Name of table to index
        """
        # Delete existing entries for this table
        if self.use_weaviate:
            self.vector_store.delete_by_table_name(table_name)

        # Get table info
        info = self.db_manager.get_table_info(table_name)

        # Create description
        description = self._create_table_description(info)

        # Index
        self.vector_store.add_documents([{
            'content': description,
            'metadata': {
                'table_name': table_name,
                'original_filename': info['original_filename'],
                'columns': info['columns'],
                'row_count': info['row_count']
            }
        }])

        print(f"✓ Indexed table: {table_name}")

    def _create_table_description(self, table_info: Dict) -> str:
        """Create searchable description of table"""
        description = f"Table: {table_info['table_name']}\n"
        description += f"Source: {table_info['original_filename']}\n"
        description += f"Contains {table_info['row_count']:,} records\n\n"

        # Add column information
        description += "Columns:\n"
        for col in table_info['columns']:
            col_type = table_info['types'].get(col, 'unknown')
            description += f"- {col} ({col_type})\n"

        # Add sample values
        if table_info['sample_data']:
            description += "\nSample data:\n"
            for i, row in enumerate(table_info['sample_data'][:2], 1):
                description += f"Record {i}:\n"
                for key, value in list(row.items())[:8]:
                    description += f"  {key}: {value}\n"

        # Add semantic hints
        description += "\nData characteristics:\n"

        filename_lower = table_info['original_filename'].lower()
        columns_lower = [col.lower() for col in table_info['columns']]

        if 'amazon' in filename_lower:
            description += "- Amazon marketplace sales data\n"
        if 'international' in filename_lower:
            description += "- International sales transactions\n"
        if 'expense' in filename_lower:
            description += "- Expense and financial records\n"
        if 'p_l' in table_info['table_name'] or 'mrp' in ' '.join(columns_lower):
            description += "- Pricing and profit/loss information\n"

        # Identify key column types
        if any('date' in col for col in columns_lower):
            description += "- Time-series data with dates\n"
        if any('amount' in col or 'price' in col or 'rate' in col for col in columns_lower):
            description += "- Financial/monetary data\n"
        if any('customer' in col for col in columns_lower):
            description += "- Customer-level information\n"
        if any('sku' in col or 'style' in col or 'category' in col for col in columns_lower):
            description += "- Product catalog information\n"
        if any('qty' in col or 'quantity' in col or 'pcs' in col for col in columns_lower):
            description += "- Quantity/inventory data\n"

        return description

    def search_tables(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant tables"""
        return self.vector_store.similarity_search(query, k=k)

    def close(self):
        """Close vector store connection"""
        if hasattr(self.vector_store, 'close'):
            self.vector_store.close()


# Test Weaviate vector store
if __name__ == "__main__":
    from storage.duckdb_manager import DuckDBManager

    print("=" * 80)
    print("Testing Weaviate Vector Store")
    print("=" * 80)

    # Initialize DuckDB
    db = DuckDBManager()
    db.load_csv_files("/Users/manohar.vanam/Documents/sales-analyzer/Sales Dataset")

    print("\n" + "=" * 80)
    print("Indexing Metadata in Weaviate")
    print("=" * 80)

    # Create metadata indexer with Weaviate
    indexer = MetadataIndexer(db, use_weaviate=True)
    num_indexed = indexer.index_all_tables(force_reindex=True)

    print(f"\n✅ Indexed {num_indexed} tables in Weaviate")

    # Test searches
    print("\n" + "=" * 80)
    print("Test Searches")
    print("=" * 80)

    test_queries = [
        "Amazon sales orders",
        "international customer transactions",
        "product pricing and MRP",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = indexer.search_tables(query, k=2)

        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['metadata']['table_name']} (score: {result['similarity_score']:.3f})")
            print(f"     File: {result['metadata']['original_filename']}")
            print(f"     Columns: {', '.join(result['metadata']['columns'][:5])}...")

    # Test dynamic indexing
    print("\n" + "=" * 80)
    print("Test Dynamic Re-indexing")
    print("=" * 80)

    print("\nRe-indexing single table: amazon_sale_report")
    indexer.index_single_table('amazon_sale_report')

    # List all indexed tables
    print("\n" + "=" * 80)
    print("All Indexed Tables")
    print("=" * 80)
    tables = indexer.vector_store.get_all_tables()
    for table in tables:
        print(f"  • {table}")

    indexer.close()
    db.close()

    print("\n✅ Weaviate test completed!")