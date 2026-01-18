import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, List
from storage.vector_store import MetadataIndexer
from storage.duckdb_manager import DuckDBManager


class MetadataAgent:
    """
    Discovers which tables/files are relevant for a query
    Uses vector search on table metadata
    """

    def __init__(self, db_manager: DuckDBManager, indexer: MetadataIndexer):
        self.name = "Metadata Discovery"
        self.db_manager = db_manager
        self.indexer = indexer

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find relevant tables for the query

        Args:
            query: User query
            context: Additional context

        Returns:
            Dictionary with relevant tables and metadata
        """
        # Search for relevant tables
        results = self.indexer.search_tables(query, k=3)

        relevant_tables = []

        for result in results:
            table_name = result['metadata']['table_name']

            # Get detailed info
            table_info = self.db_manager.get_table_info(table_name)

            relevant_tables.append({
                'table_name': table_name,
                'original_filename': result['metadata']['original_filename'],
                'relevance_score': result['similarity_score'],
                'columns': table_info['columns'],
                'row_count': table_info['row_count'],
                'sample_data': table_info['sample_data'][:2]  # Only 2 samples
            })

        return {
            'agent': self.name,
            'relevant_tables': relevant_tables,
            'recommendation': self._generate_recommendation(relevant_tables, query)
        }

    def _generate_recommendation(self, tables: List[Dict], query: str) -> str:
        """Generate natural language recommendation"""

        if not tables:
            return "No relevant tables found for this query."

        recommendation = f"Found {len(tables)} relevant table(s):\n\n"

        for i, table in enumerate(tables, 1):
            recommendation += f"{i}. **{table['original_filename']}** ({table['row_count']:,} rows)\n"
            recommendation += f"   Relevance: {table['relevance_score']:.2%}\n"
            recommendation += f"   Key columns: {', '.join(table['columns'][:5])}\n\n"

        return recommendation


# Test the metadata agent
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Metadata Agent")
    print("=" * 80)

    # Setup
    db = DuckDBManager()
    db.load_csv_files("/Users/shashank/Desktop/AI/Blend360Assignment/Sales Dataset")

    indexer = MetadataIndexer(db)
    indexer.index_all_tables()

    agent = MetadataAgent(db, indexer)

    # Test queries
    test_queries = [
        "Show me Amazon sales for April 2022",
        "What are the international customer orders?",
        "Find product pricing and MRP information",
        "Show expense records",
        "Which products are in stock?"
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print('=' * 80)

        result = agent.process(query)
        print(result['recommendation'])

    db.close()