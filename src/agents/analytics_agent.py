import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Any
from storage.duckdb_manager import DuckDBManager
import pandas as pd
import numpy as np


class AnalyticsAgent:
    """
    Performs analytical tasks: trends, comparisons, growth calculations
    """

    def __init__(self, db_manager: DuckDBManager):
        self.name = "Analytics"
        self.db_manager = db_manager

    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze data for trends, comparisons, and insights

        Args:
            query: User query
            context: Context including query results from structured_query_agent

        Returns:
            Dictionary with analytics results
        """
        if not context or 'results' not in context:
            return {
                'agent': self.name,
                'error': 'No query results to analyze',
                'analysis': None
            }

        results = context['results']
        query_lower = query.lower()

        # Determine analysis type
        if any(word in query_lower for word in ['trend', 'over time', 'growth', 'change']):
            analysis = self._analyze_trends(results, query)
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            analysis = self._analyze_comparison(results, query)
        elif any(word in query_lower for word in ['top', 'bottom', 'best', 'worst']):
            analysis = self._analyze_ranking(results, query)
        else:
            analysis = self._analyze_summary(results, query)

        return {
            'agent': self.name,
            'analysis_type': analysis['type'],
            'insights': analysis['insights'],
            'visualization_data': analysis.get('viz_data', None)
        }

    def _analyze_trends(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze trends over time"""

        insights = []

        # Check if we have time-series data
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower()]
        value_cols = [col for col in df.columns if col not in date_cols and pd.api.types.is_numeric_dtype(df[col])]

        if not value_cols:
            return {
                'type': 'trend',
                'insights': ['No numeric data available for trend analysis'],
                'viz_data': None
            }

        # Calculate basic statistics
        for col in value_cols[:3]:  # Analyze first 3 numeric columns
            if len(df) > 1:
                # Growth rate
                first_val = df[col].iloc[0]
                last_val = df[col].iloc[-1]

                if first_val and first_val != 0:
                    growth_rate = ((last_val - first_val) / first_val) * 100
                    insights.append(f"{col}: {growth_rate:+.1f}% change from first to last period")

                # Average
                avg_val = df[col].mean()
                insights.append(f"{col}: Average value is ₹{avg_val:,.2f}")

                # Volatility
                std_val = df[col].std()
                if avg_val != 0:
                    volatility = (std_val / avg_val) * 100
                    insights.append(f"{col}: Volatility is {volatility:.1f}%")

        return {
            'type': 'trend',
            'insights': insights,
            'viz_data': df.to_dict('records')
        }

    def _analyze_comparison(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Compare different entities or periods"""

        insights = []

        if len(df) < 2:
            return {
                'type': 'comparison',
                'insights': ['Need at least 2 items to compare'],
                'viz_data': None
            }

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {
                'type': 'comparison',
                'insights': ['No numeric data to compare'],
                'viz_data': None
            }

        # Compare top vs bottom
        for col in numeric_cols[:2]:  # First 2 numeric columns
            top_val = df[col].max()
            bottom_val = df[col].min()

            if bottom_val != 0:
                ratio = top_val / bottom_val
                insights.append(f"{col}: Highest is {ratio:.1f}x the lowest")

            # Find which rows
            top_row = df[df[col] == top_val].iloc[0]
            bottom_row = df[df[col] == bottom_val].iloc[0]

            # Get identifier (first non-numeric column)
            id_col = [c for c in df.columns if c not in numeric_cols][0] if any(
                c not in numeric_cols for c in df.columns) else df.columns[0]

            insights.append(f"Highest {col}: {top_row[id_col]} (₹{top_val:,.2f})")
            insights.append(f"Lowest {col}: {bottom_row[id_col]} (₹{bottom_val:,.2f})")

        return {
            'type': 'comparison',
            'insights': insights,
            'viz_data': df.to_dict('records')
        }

    def _analyze_ranking(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze rankings and distributions"""

        insights = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {
                'type': 'ranking',
                'insights': ['No numeric data to rank'],
                'viz_data': None
            }

        # Analyze first numeric column (usually the main metric)
        main_col = numeric_cols[0]

        # Top concentration
        total = df[main_col].sum()
        if len(df) >= 3:
            top_3_total = df[main_col].head(3).sum()
            top_3_pct = (top_3_total / total) * 100
            insights.append(f"Top 3 items account for {top_3_pct:.1f}% of total {main_col}")

        if len(df) >= 5:
            top_5_total = df[main_col].head(5).sum()
            top_5_pct = (top_5_total / total) * 100
            insights.append(f"Top 5 items account for {top_5_pct:.1f}% of total {main_col}")

        # Distribution analysis
        median_val = df[main_col].median()
        mean_val = df[main_col].mean()

        insights.append(f"{main_col}: Median = ₹{median_val:,.2f}, Mean = ₹{mean_val:,.2f}")

        if mean_val > median_val * 1.2:
            insights.append(f"Distribution is right-skewed (few high performers)")
        elif median_val > mean_val * 1.2:
            insights.append(f"Distribution is left-skewed (few low performers)")
        else:
            insights.append(f"Distribution is relatively balanced")

        return {
            'type': 'ranking',
            'insights': insights,
            'viz_data': df.to_dict('records')
        }

    def _analyze_summary(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """General summary statistics"""

        insights = []

        # Basic info
        insights.append(f"Dataset contains {len(df)} records")

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols[:3]:
            total = df[col].sum()
            avg = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()

            insights.append(
                f"{col}: Total = ₹{total:,.2f}, Avg = ₹{avg:,.2f}, Range = ₹{min_val:,.2f} to ₹{max_val:,.2f}")

        return {
            'type': 'summary',
            'insights': insights,
            'viz_data': df.to_dict('records')
        }


# Test the analytics agent
if __name__ == "__main__":
    from storage.vector_store import MetadataIndexer
    from metadata_agent import MetadataAgent
    from structured_query_agent import StructuredQueryAgent

    print("=" * 80)
    print("Testing Analytics Agent")
    print("=" * 80)

    # Setup
    db = DuckDBManager()
    db.load_csv_files("/Users/shashank/Desktop/AI/Blend360Assignment/Sales Dataset")

    indexer = MetadataIndexer(db)
    indexer.index_all_tables()

    metadata_agent = MetadataAgent(db, indexer)
    query_agent = StructuredQueryAgent(db, use_llm=True)
    analytics_agent = AnalyticsAgent(db)

    # Test queries
    test_queries = [
        "Show me top 10 customers from international sales",
        "Compare sales by category in Amazon",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print('=' * 80)

        # Step 1: Get relevant tables
        metadata_result = metadata_agent.process(query)

        # Step 2: Execute query
        query_result = query_agent.process(query, context=metadata_result)

        if 'error' not in query_result:
            # Step 3: Analyze results
            analytics_context = {
                'results': query_result['results'],
                'sql': query_result['sql']
            }

            analytics_result = analytics_agent.process(query, context=analytics_context)

            print(f"\n📊 Analysis Type: {analytics_result['analysis_type']}")
            print(f"\n💡 Insights:")
            for insight in analytics_result['insights']:
                print(f"  • {insight}")

    db.close()