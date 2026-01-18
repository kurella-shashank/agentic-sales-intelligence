import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any, List
from storage.duckdb_manager import DuckDBManager
import pandas as pd

try:
    # from langchain_ollama import ChatOllama
    from langchain_google_genai import ChatGoogleGenerativeAI

    ChatGoogleGenerativeAI = True
except ImportError:
    ChatGoogleGenerativeAI = False


class SummarizationAgent:
    """
    Generates narrative summaries and insights from data
    """

    def __init__(self, use_llm: bool = True):
        self.name = "Summarization"
        self.use_llm = use_llm and ChatGoogleGenerativeAI

        if self.use_llm:
            try:
                self.llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            temperature=0,  # Gemini 3.0+ defaults to 1.0
                        )
                # self.llm = ChatOllama(
                #     model="gpt-oss:latest",
                #     temperature=0.3,  # Slightly creative for summaries
                #     base_url="http://localhost:11434"
                # )
                print("✓ Using Gemini for summarization")
            except Exception as e:
                print(f"⚠️  Gemini not available: {e}")
                self.use_llm = False

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate narrative summary of results

        Args:
            query: User query
            context: Context with results and analysis

        Returns:
            Dictionary with summary
        """
        if not context:
            return {
                'agent': self.name,
                'summary': 'No data available to summarize',
                'key_points': []
            }

        # Extract information from context
        results = context.get('results')
        analysis = context.get('insights', [])
        sql = context.get('sql', '')
        table_used = context.get('table_used', 'unknown')

        if self.use_llm:
            summary = self._generate_llm_summary(query, results, analysis, table_used)
        else:
            summary = self._generate_rule_based_summary(query, results, analysis, table_used)

        return {
            'agent': self.name,
            'summary': summary['narrative'],
            'key_points': summary['key_points']
        }

    def _generate_llm_summary(self, query: str, results: pd.DataFrame,
                              analysis: List[str], table_used: str) -> Dict[str, Any]:
        """Generate summary using LLM"""

        # Prepare data summary
        data_summary = ""
        if results is not None and not results.empty:
            data_summary = f"Data shape: {len(results)} rows\n"
            data_summary += f"Sample data:\n{results.head(5).to_string()}\n"

        insights_text = "\n".join(f"- {insight}" for insight in analysis) if analysis else "No analysis available"

        prompt = f"""You are a business analyst. Create a concise, professional summary of sales data.

User Question: {query}

Data Source: {table_used}

{data_summary}

Analysis Insights:
{insights_text}

Task: Write a brief 2-3 sentence summary highlighting the most important findings.
Focus on actionable insights and key numbers.
Use a professional, business-friendly tone.

Summary:"""

        try:
            response = self.llm.invoke(prompt)
            narrative = response.content.strip()

            # Extract key points
            key_points = analysis[:3] if analysis else []

            return {
                'narrative': narrative,
                'key_points': key_points
            }
        except Exception as e:
            print(f"⚠️  LLM summarization failed: {e}")
            return self._generate_rule_based_summary(query, results, analysis, table_used)

    def _generate_rule_based_summary(self, query: str, results: pd.DataFrame,
                                     analysis: List[str], table_used: str) -> Dict[str, Any]:
        """Generate summary using rules"""

        if results is None or results.empty:
            return {
                'narrative': f"No data found for the query: '{query}'",
                'key_points': []
            }

        # Build narrative
        narrative_parts = []

        # Opening
        narrative_parts.append(f"Analysis of {table_used} data shows:")

        # Add key findings
        if len(results) == 1:
            # Single result
            first_col = results.columns[0]
            value = results.iloc[0][first_col]
            if isinstance(value, (int, float)):
                if value > 1000000:
                    narrative_parts.append(f"The result is ₹{value:,.2f} ({value / 1000000:.2f}M).")
                else:
                    narrative_parts.append(f"The result is ₹{value:,.2f}.")
            else:
                narrative_parts.append(f"The result is {value}.")
        else:
            # Multiple results
            narrative_parts.append(f"{len(results)} records were found.")

            # If we have analysis insights, add the top one
            if analysis:
                narrative_parts.append(analysis[0])

        narrative = " ".join(narrative_parts)

        # Key points from analysis
        key_points = analysis[:3] if analysis else [f"Found {len(results)} matching records"]

        return {
            'narrative': narrative,
            'key_points': key_points
        }


# Test the summarization agent
if __name__ == "__main__":
    from storage.vector_store import MetadataIndexer
    from metadata_agent import MetadataAgent
    from structured_query_agent import StructuredQueryAgent
    from analytics_agent import AnalyticsAgent

    print("=" * 80)
    print("Testing Summarization Agent")
    print("=" * 80)

    # Setup
    db = DuckDBManager()
    db.load_csv_files("/Users/shashank/Desktop/AI/Blend360Assignment/Sales Dataset")

    indexer = MetadataIndexer(db)
    indexer.index_all_tables()

    metadata_agent = MetadataAgent(db, indexer)
    query_agent = StructuredQueryAgent(db, use_llm=True)
    analytics_agent = AnalyticsAgent(db)
    summarization_agent = SummarizationAgent(use_llm=True)  # Test rule-based

    # Test queries
    test_queries = [
        "What is the total amount in international sales?",
        "Show me top 5 customers from international sales"
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print('=' * 80)

        # Step 1: Metadata
        metadata_result = metadata_agent.process(query)
        print(f"📁 Using: {metadata_result['relevant_tables'][0]['original_filename']}")

        # Step 2: Execute query
        query_result = query_agent.process(query, context=metadata_result)

        if 'error' not in query_result:
            print(f"\n🔍 SQL:\n{query_result['sql']}")
            print(f"\n📊 Results ({query_result['row_count']} rows):")
            print(query_result['results'].head(5).to_string(index=False))

            # Step 3: Analytics
            analytics_context = {
                'results': query_result['results'],
                'sql': query_result['sql']
            }
            analytics_result = analytics_agent.process(query, context=analytics_context)

            # Step 4: Summarization
            summary_context = {
                'results': query_result['results'],
                'insights': analytics_result['insights'],
                'sql': query_result['sql'],
                'table_used': query_result['table_used']
            }
            summary_result = summarization_agent.process(query, context=summary_context)

            print(f"\n💡 Insights:")
            for insight in analytics_result['insights']:
                print(f"  • {insight}")

            print(f"\n📝 Summary:")
            print(f"  {summary_result['summary']}")

            print(f"\n🔑 Key Points:")
            for i, point in enumerate(summary_result['key_points'], 1):
                print(f"  {i}. {point}")
        else:
            print(f"\n❌ Error: {query_result['error']}")

    db.close()