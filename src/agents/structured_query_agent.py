import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, List, Tuple
from storage.duckdb_manager import DuckDBManager
import pandas as pd

try:
    # from langchain_ollama import ChatOllama
    from langchain_google_genai import ChatGoogleGenerativeAI

    ChatGoogleGenerativeAI = True
except ImportError:
    ChatGoogleGenerativeAI = False
    print("⚠️  Gemini not available, using rule-based SQL generation")


class StructuredQueryAgent:
    """Generates and executes SQL queries on sales data"""

    def __init__(self, db_manager: DuckDBManager, use_llm: bool = True):
        self.name = "Structured Query"
        self.db_manager = db_manager
        self.use_llm = use_llm and ChatGoogleGenerativeAI

        if self.use_llm:
            try:
                self.llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            temperature=0,  # Gemini 3.0+ defaults to 1.0
                        )
                # self.llm = ChatOllama(
                #     model="gpt-oss:latest",
                #     temperature=0,
                #     base_url="http://localhost:11434"
                # )
                test_response = self.llm.invoke("Hi")
                print("✓ Using Gemini for SQL generation")
            except Exception as e:
                print(f"⚠️  Gemini connection failed: {e}")
                print("   Falling back to rule-based SQL generation")
                self.use_llm = False

    def _quote_column(self, column_name: str) -> str:
        """Properly quote column names (handles spaces and special chars)"""
        if ' ' in column_name or '-' in column_name or any(c in column_name for c in ['(', ')', ':', '/']):
            return f'"{column_name}"'
        return column_name

    def _cast_to_numeric(self, column_name: str) -> str:
        """
        Cast a column to numeric, handling VARCHAR columns with numeric data
        TRY_CAST returns NULL for non-numeric values instead of erroring
        """
        return f"TRY_CAST({column_name} AS DOUBLE)"

    def _find_best_amount_column(self, columns: list, table_name: str) -> Tuple[str, bool]:
        """
        Find the best column for amounts/sales
        Returns: (column_name, is_varchar)
        """
        schema = self.db_manager.get_table_schema(table_name)

        priority_keywords = [
            ['gross amt', 'gross_amt'],
            ['amount'],
            ['amt'],
            ['rate']
        ]

        for keywords in priority_keywords:
            for col in columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    # Check if VARCHAR
                    col_type = schema['types'].get(col, '').upper()
                    is_varchar = 'VARCHAR' in col_type or 'TEXT' in col_type
                    return (col, is_varchar)

        return (None, False)

    def _find_column(self, columns: list, keywords: list) -> str:
        """Find column matching any keyword"""
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                return col
        return None

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate and execute SQL query"""

        if not context or 'relevant_tables' not in context:
            return {'agent': self.name, 'error': 'No table context provided', 'sql': None, 'results': None}

        relevant_tables = context['relevant_tables']
        if not relevant_tables:
            return {'agent': self.name, 'error': 'No relevant tables found', 'sql': None, 'results': None}

        primary_table = relevant_tables[0]
        table_name = primary_table['table_name']

        # Generate SQL
        if self.use_llm:
            sql = self._generate_sql_with_llm(query, primary_table)
        else:
            sql = self._generate_sql_rule_based(query, primary_table)

        # Execute SQL
        try:
            results = self.db_manager.execute_query(sql)

            return {
                'agent': self.name,
                'table_used': table_name,
                'original_filename': primary_table['original_filename'],
                'sql': sql,
                'results': results,
                'row_count': len(results),
                'summary': self._summarize_results(results, query)
            }

        except Exception as e:
            return {
                'agent': self.name,
                'table_used': table_name,
                'sql': sql,
                'error': str(e),
                'results': None
            }

    def _generate_sql_with_llm(self, query: str, table_info: Dict) -> str:
        """Generate SQL using LLM"""

        # Quote all column names
        quoted_columns = [self._quote_column(col) for col in table_info['columns']]

        # Get schema info to help LLM understand VARCHAR columns
        schema = self.db_manager.get_table_schema(table_info['table_name'])
        column_types = []
        for col in table_info['columns']:
            col_type = schema['types'].get(col, 'UNKNOWN')
            column_types.append(f"{self._quote_column(col)} ({col_type})")

        prompt = f"""You are a SQL expert. Generate a DuckDB SQL query for the following request.

Table: {table_info['table_name']}
Columns with types: {', '.join(column_types[:10])}

Sample data:
{table_info['sample_data'][0] if table_info['sample_data'] else 'No sample data'}

User Query: {query}

Requirements:
- Use DuckDB SQL syntax
- Column names with spaces MUST be wrapped in double quotes: "Column Name"
- If a column contains numbers but is VARCHAR, use TRY_CAST("column" AS DOUBLE)
- Return ONLY the SQL query, no explanations or markdown
- Use appropriate aggregations (SUM, COUNT, AVG) if needed
- Add ORDER BY for rankings
- Limit results to 20 rows if listing data

SQL Query:"""

        try:
            response = self.llm.invoke(prompt)
            sql = response.content.strip()

            # Clean up SQL (remove markdown)
            sql = sql.replace('```sql', '').replace('```', '').strip()

            return sql

        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}, falling back to rule-based")
            return self._generate_sql_rule_based(query, table_info)

    def _generate_sql_rule_based(self, query: str, table_info: Dict) -> str:
        """Generate SQL using rules with smart column detection"""

        table_name = table_info['table_name']
        columns = table_info['columns']
        query_lower = query.lower()

        # Find the best amount column (returns tuple: column_name, is_varchar)
        amount_col, amount_is_varchar = self._find_best_amount_column(columns, table_name)

        # If amount column is VARCHAR, we need to cast it
        if amount_col:
            amount_col_quoted = self._quote_column(amount_col)
            if amount_is_varchar:
                amount_col_sql = self._cast_to_numeric(amount_col_quoted)
            else:
                amount_col_sql = amount_col_quoted
        else:
            amount_col_sql = None

        # Find other columns
        customer_col = self._find_column(columns, ['customer', 'cust'])
        category_col = self._find_column(columns, ['category', 'style', 'sku'])
        date_col = self._find_column(columns, ['date', 'month'])
        status_col = self._find_column(columns, ['status'])

        # COUNT queries
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            if 'cancelled' in query_lower and status_col:
                return f"""SELECT COUNT(*) as cancelled_count 
                FROM {table_name}
                WHERE {self._quote_column(status_col)} = 'Cancelled'"""

            if customer_col and 'customer' in query_lower:
                return f"SELECT COUNT(DISTINCT {self._quote_column(customer_col)}) as unique_customers FROM {table_name}"

            return f"SELECT COUNT(*) as total_count FROM {table_name}"

        # SUM/TOTAL queries
        if any(word in query_lower for word in ['total', 'sum', 'how much']):
            if amount_col_sql:
                if 'by customer' in query_lower and customer_col:
                    return f"""SELECT {self._quote_column(customer_col)}, 
                       SUM({amount_col_sql}) as total
                FROM {table_name}
                GROUP BY {self._quote_column(customer_col)}
                ORDER BY total DESC
                LIMIT 20"""

                if ('by category' in query_lower or 'by product' in query_lower) and category_col:
                    return f"""SELECT {self._quote_column(category_col)}, 
                       SUM({amount_col_sql}) as total
                FROM {table_name}
                GROUP BY {self._quote_column(category_col)}
                ORDER BY total DESC
                LIMIT 20"""

                return f"SELECT SUM({amount_col_sql}) as total_amount FROM {table_name}"

        # TOP/BEST queries
        if any(word in query_lower for word in ['top', 'best', 'highest', 'largest']):
            if customer_col and amount_col_sql:
                limit = 10
                for word in query_lower.split():
                    if word.isdigit():
                        limit = int(word)
                        break

                return f"""SELECT {self._quote_column(customer_col)}, 
                   SUM({amount_col_sql}) as total
            FROM {table_name}
            GROUP BY {self._quote_column(customer_col)}
            ORDER BY total DESC
            LIMIT {limit}"""

        # SHOW/LIST queries
        if any(word in query_lower for word in ['show', 'list', 'display']):
            if 'by customer' in query_lower or 'per customer' in query_lower:
                if customer_col and amount_col_sql:
                    return f"""SELECT {self._quote_column(customer_col)}, 
                       SUM({amount_col_sql}) as total
                FROM {table_name}
                GROUP BY {self._quote_column(customer_col)}
                ORDER BY total DESC
                LIMIT 20"""

            if 'by category' in query_lower or 'by product' in query_lower:
                if category_col and amount_col_sql:
                    return f"""SELECT {self._quote_column(category_col)}, 
                       SUM({amount_col_sql}) as total
                FROM {table_name}
                GROUP BY {self._quote_column(category_col)}
                ORDER BY total DESC
                LIMIT 20"""

        # STATUS queries
        if status_col and 'cancelled' in query_lower:
            if amount_col_sql:
                return f"""SELECT COUNT(*) as cancelled_orders,
                       SUM({amount_col_sql}) as cancelled_amount
                FROM {table_name}
                WHERE {self._quote_column(status_col)} = 'Cancelled'"""
            return f"""SELECT COUNT(*) as cancelled_orders
            FROM {table_name}
            WHERE {self._quote_column(status_col)} = 'Cancelled'"""

        # Default
        key_columns = [c for c in [customer_col, amount_col, date_col, category_col] if c]
        if key_columns:
            cols_str = ', '.join([self._quote_column(c) for c in key_columns[:5]])
            return f"SELECT {cols_str} FROM {table_name} LIMIT 20"

        return f"SELECT * FROM {table_name} LIMIT 20"

    def _summarize_results(self, results: pd.DataFrame, query: str) -> str:
        """Generate natural language summary of results"""

        if results.empty:
            return "No results found."

        if len(results) == 1 and len(results.columns) <= 2:
            summary_parts = []
            for col in results.columns:
                value = results.iloc[0][col]
                if pd.notna(value) and isinstance(value, (int, float)):
                    if value > 1000000:
                        summary_parts.append(f"{col}: ₹{value:,.2f} ({value / 1000000:.2f}M)")
                    else:
                        summary_parts.append(f"{col}: {value:,.0f}")
                else:
                    summary_parts.append(f"{col}: {value}")
            return " | ".join(summary_parts)

        summary = f"Found {len(results)} results."

        if len(results) > 1 and len(results.columns) > 1:
            first_col = results.columns[0]
            second_col = results.columns[1]
            top_value = results.iloc[0]
            if pd.notna(top_value[second_col]):
                summary += f"\n\nTop result: {top_value[first_col]} with ₹{top_value[second_col]:,.2f}"

        return summary


# Test
if __name__ == "__main__":
    from storage.vector_store import MetadataIndexer
    from metadata_agent import MetadataAgent

    print("=" * 80)
    print("Testing FIXED Structured Query Agent (VARCHAR Casting)")
    print("=" * 80)

    db = DuckDBManager()
    db.load_csv_files("/Users/shashank/Desktop/AI/Blend360Assignment/Sales Dataset")

    indexer = MetadataIndexer(db)
    indexer.index_all_tables()

    metadata_agent = MetadataAgent(db, indexer)
    query_agent = StructuredQueryAgent(db, use_llm=True)  # Test rule-based first

    test_queries = [
        "What is the total amount in international sales?",
        "Show me sales by customer from international sales",
        "Top 5 customers by sales in international report",
        "What is the total amount in Amazon sales?",
        "How many cancelled orders in Amazon sales?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print('=' * 80)

        metadata_result = metadata_agent.process(query)
        print(f"📁 Using: {metadata_result['relevant_tables'][0]['original_filename']}")

        query_result = query_agent.process(query, context=metadata_result)

        if 'error' in query_result:
            print(f"❌ Error: {query_result['error']}")
            print(f"SQL: {query_result['sql']}")
        else:
            print(f"\n✅ SQL:\n{query_result['sql']}")
            print(f"\n📈 Results ({query_result['row_count']} rows):")
            print(query_result['results'].head(10).to_string(index=False))
            print(f"\n💡 {query_result['summary']}")

    db.close()