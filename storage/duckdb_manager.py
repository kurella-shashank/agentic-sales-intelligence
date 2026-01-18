import duckdb
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class DuckDBManager:
    """Manages DuckDB connections and CSV data loading"""

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize DuckDB connection

        Args:
            db_path: Path to database file or :memory: for in-memory
        """
        self.conn = duckdb.connect(db_path)
        self.tables = {}

        # Configure DuckDB for better CSV handling
        self.conn.execute("SET memory_limit='4GB'")
        self.conn.execute("SET threads TO 4")

    def load_csv_files(self, csv_directory: str):
        """
        Load all CSV files from directory as external tables

        Args:
            csv_directory: Path to directory containing CSV files
        """
        csv_path = Path(csv_directory)

        if not csv_path.exists():
            raise FileNotFoundError(f"Directory not found: {csv_directory}")

        csv_files = list(csv_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {csv_directory}")

        for csv_file in csv_files:
            # Clean table name: remove spaces, special chars
            table_name = self._clean_table_name(csv_file.stem)

            # Create view directly on CSV (no loading into memory)
            abs_path = csv_file.absolute()

            # DuckDB can handle files with spaces if properly quoted
            query = f"""
            CREATE OR REPLACE VIEW {table_name} AS 
            SELECT * FROM read_csv_auto('{abs_path}', 
                                        ignore_errors=true,
                                        header=true,
                                        sample_size=100000)
            """

            try:
                self.conn.execute(query)
                self.tables[table_name] = {
                    'path': str(csv_file),
                    'original_name': csv_file.name
                }

                # Get row count
                count_result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = count_result[0] if count_result else 0

                print(f"✓ Loaded: {table_name:30s} | Rows: {row_count:,}")

            except Exception as e:
                print(f"✗ Failed to load {csv_file.name}: {str(e)}")

    def _clean_table_name(self, name: str) -> str:
        """
        Clean table name for SQL compatibility
        Remove spaces, special characters
        """
        # Replace spaces and special chars with underscore
        clean = name.replace(' ', '_')
        clean = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean)
        # Remove consecutive underscores
        while '__' in clean:
            clean = clean.replace('__', '_')
        return clean.lower().strip('_')

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        Args:
            sql: SQL query string

        Returns:
            Query results as pandas DataFrame
        """
        try:
            result = self.conn.execute(sql).fetchdf()
            return result
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}\n\nSQL:\n{sql}")

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a table

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with schema information
        """
        query = f"DESCRIBE {table_name}"
        schema_df = self.conn.execute(query).fetchdf()

        return {
            "table_name": table_name,
            "columns": schema_df['column_name'].tolist(),
            "types": dict(zip(schema_df['column_name'], schema_df['column_type']))
        }

    def get_all_tables(self) -> List[str]:
        """Get list of all available tables"""
        return list(self.tables.keys())

    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get sample rows from a table

        Args:
            table_name: Name of the table
            limit: Number of rows to return

        Returns:
            Sample data as DataFrame
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a table

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        schema = self.get_table_schema(table_name)
        sample = self.get_sample_data(table_name, limit=3)

        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        row_count = self.execute_query(count_query)['row_count'].iloc[0]

        return {
            "table_name": table_name,
            "original_filename": self.tables[table_name]['original_name'],
            "row_count": row_count,
            "columns": schema['columns'],
            "types": schema['types'],
            "sample_data": sample.to_dict('records')
        }

    def get_metadata_summary(self) -> str:
        """
        Generate a text summary of all tables for LLM context
        """
        summary = "Available Sales Data Tables:\n\n"

        for table_name in self.get_all_tables():
            info = self.get_table_info(table_name)

            summary += f"Table: {table_name}\n"
            summary += f"Original File: {info['original_filename']}\n"
            summary += f"Rows: {info['row_count']:,}\n"
            summary += f"Columns: {', '.join(info['columns'])}\n"

            # Add sample values for first few columns
            if info['sample_data']:
                summary += "Sample data:\n"
                first_row = info['sample_data'][0]
                for col in list(first_row.keys())[:5]:  # First 5 columns
                    summary += f"  - {col}: {first_row[col]}\n"

            summary += "\n"

        return summary

    def close(self):
        """Close database connection"""
        self.conn.close()


# Test the DuckDB manager
if __name__ == "__main__":
    print("=" * 80)
    print("Testing DuckDB Manager with Your Sales Dataset")
    print("=" * 80)

    # Initialize DuckDB
    db = DuckDBManager()

    # Load your CSV files
    data_path = "/Users/manohar.vanam/Documents/sales-analyzer/Sales Dataset"
    print(f"\nLoading CSV files from: {data_path}\n")

    try:
        db.load_csv_files(data_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        exit(1)

    print(f"\n✅ Loaded {len(db.get_all_tables())} tables\n")

    # Show metadata summary
    print("=" * 80)
    print("METADATA SUMMARY (for LLM context)")
    print("=" * 80)
    print(db.get_metadata_summary())

    # Test a query on the largest file (Amazon Sale Report)
    print("=" * 80)
    print("Test Query: Amazon Sale Report Statistics")
    print("=" * 80)

    try:
        # Get first table to test
        first_table = db.get_all_tables()[0]
        schema = db.get_table_schema(first_table)

        print(f"\nQuerying table: {first_table}")
        print(f"Columns available: {schema['columns'][:10]}...")  # Show first 10

        # Simple count query
        result = db.execute_query(f"SELECT COUNT(*) as total_rows FROM {first_table}")
        print(f"\nTotal rows: {result['total_rows'].iloc[0]:,}")

        # Sample data
        sample = db.get_sample_data(first_table, limit=3)
        print(f"\nSample data:")
        print(sample)

    except Exception as e:
        print(f"Error in test query: {e}")

    db.close()
    print("\n✅ Test completed!")