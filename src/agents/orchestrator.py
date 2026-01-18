import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, TypedDict, Annotated
from storage.duckdb_manager import DuckDBManager
from storage.vector_store import MetadataIndexer
from src.agents.metadata_agent import MetadataAgent
from src.agents.structured_query_agent import StructuredQueryAgent
from src.agents.analytics_agent import AnalyticsAgent
from src.agents.summarization_agent import SummarizationAgent

try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not available, using simple orchestration")


class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    relevant_tables: list
    sql: str
    results: Any
    insights: list
    summary: str
    error: str
    current_step: str


class SalesIntelligenceOrchestrator:
    """
    Orchestrates multiple agents to answer sales queries
    Uses LangGraph for workflow management
    """

    def __init__(self, db_manager: DuckDBManager, indexer: MetadataIndexer, use_llm: bool = False):
        self.db_manager = db_manager
        self.indexer = indexer

        # Initialize agents
        self.metadata_agent = MetadataAgent(db_manager, indexer)
        self.query_agent = StructuredQueryAgent(db_manager, use_llm=use_llm)
        self.analytics_agent = AnalyticsAgent(db_manager)
        self.summarization_agent = SummarizationAgent(use_llm=use_llm)

        # Build workflow
        if LANGGRAPH_AVAILABLE:
            self.workflow = self._build_langgraph_workflow()
        else:
            self.workflow = None

    def _build_langgraph_workflow(self):
        """Build LangGraph workflow"""

        # Define the graph
        workflow = StateGraph(AgentState)

        # Add nodes (agents)
        workflow.add_node("metadata", self._metadata_node)
        workflow.add_node("query", self._query_node)
        workflow.add_node("analytics", self._analytics_node)
        workflow.add_node("summarization", self._summarization_node)

        # Define edges (flow)
        workflow.set_entry_point("metadata")
        workflow.add_edge("metadata", "query")
        workflow.add_conditional_edges(
            "query",
            self._should_analyze,
            {
                "analyze": "analytics",
                "summarize": "summarization",
                "end": END
            }
        )
        workflow.add_edge("analytics", "summarization")
        workflow.add_edge("summarization", END)

        return workflow.compile()

    def _metadata_node(self, state: AgentState) -> AgentState:
        """Metadata discovery node"""
        result = self.metadata_agent.process(state['query'])
        state['relevant_tables'] = result['relevant_tables']
        state['current_step'] = 'metadata_complete'
        return state

    def _query_node(self, state: AgentState) -> AgentState:
        """Query execution node"""
        context = {'relevant_tables': state['relevant_tables']}
        result = self.query_agent.process(state['query'], context=context)

        if 'error' in result:
            state['error'] = result['error']
            state['current_step'] = 'query_error'
        else:
            state['sql'] = result['sql']
            state['results'] = result['results']
            state['current_step'] = 'query_complete'

        return state

    def _analytics_node(self, state: AgentState) -> AgentState:
        """Analytics node"""
        context = {
            'results': state['results'],
            'sql': state['sql']
        }
        result = self.analytics_agent.process(state['query'], context=context)
        state['insights'] = result['insights']
        state['current_step'] = 'analytics_complete'
        return state

    def _summarization_node(self, state: AgentState) -> AgentState:
        """Summarization node"""
        context = {
            'results': state['results'],
            'insights': state.get('insights', []),
            'sql': state['sql'],
            'table_used': state['relevant_tables'][0]['table_name'] if state.get('relevant_tables') else 'unknown'
        }
        result = self.summarization_agent.process(state['query'], context=context)
        state['summary'] = result['summary']
        state['current_step'] = 'complete'
        return state

    def _should_analyze(self, state: AgentState) -> str:
        """Decide whether to analyze results"""

        # If error, skip to end
        if state.get('error'):
            return "end"

        # If no results, skip analysis
        if state.get('results') is None or len(state['results']) == 0:
            return "summarize"

        # Check if query needs analysis
        query_lower = state['query'].lower()

        # Simple queries don't need analysis
        if any(word in query_lower for word in ['what is', 'how much', 'how many']):
            if len(state['results']) == 1:
                return "summarize"

        # Complex queries need analysis
        if any(word in query_lower for word in ['compare', 'trend', 'top', 'best', 'analyze']):
            return "analyze"

        # Default: analyze if multiple results
        if len(state['results']) > 1:
            return "analyze"

        return "summarize"

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the multi-agent workflow

        Args:
            query: User query

        Returns:
            Complete results from all agents
        """
        if LANGGRAPH_AVAILABLE and self.workflow:
            # Use LangGraph workflow
            initial_state = AgentState(
                query=query,
                relevant_tables=[],
                sql="",
                results=None,
                insights=[],
                summary="",
                error="",
                current_step="start"
            )

            final_state = self.workflow.invoke(initial_state)

            return {
                'query': query,
                'tables_used': [t['original_filename'] for t in final_state.get('relevant_tables', [])],
                'sql': final_state.get('sql', ''),
                'results': final_state.get('results'),
                'insights': final_state.get('insights', []),
                'summary': final_state.get('summary', ''),
                'error': final_state.get('error'),
                'workflow': 'langgraph'
            }
        else:
            # Fallback: Simple sequential execution
            return self._simple_orchestration(query)

    def _simple_orchestration(self, query: str) -> Dict[str, Any]:
        """Simple sequential orchestration without LangGraph"""

        print(f"\n🔄 Processing: {query}")

        # Step 1: Metadata
        print("  1️⃣  Finding relevant tables...")
        metadata_result = self.metadata_agent.process(query)

        if not metadata_result['relevant_tables']:
            return {
                'query': query,
                'error': 'No relevant tables found',
                'workflow': 'simple'
            }

        # Step 2: Query execution
        print("  2️⃣ Executing query...")
        query_result = self.query_agent.process(query, context=metadata_result)

        if 'error' in query_result:
            return {
                'query': query,
                'tables_used': [metadata_result['relevant_tables'][0]['original_filename']],
                'error': query_result['error'],
                'sql': query_result.get('sql', ''),
                'workflow': 'simple'
            }

        # Step 3: Analytics (if needed)
        insights = []
        if query_result['results'] is not None and len(query_result['results']) > 1:
            print("  3️⃣  Analyzing results...")
            analytics_context = {
                'results': query_result['results'],
                'sql': query_result['sql']
            }
            analytics_result = self.analytics_agent.process(query, context=analytics_context)
            insights = analytics_result['insights']

        # Step 4: Summarization
        print("  4️⃣  Generating summary...")
        summary_context = {
            'results': query_result['results'],
            'insights': insights,
            'sql': query_result['sql'],
            'table_used': query_result['table_used']
        }
        summary_result = self.summarization_agent.process(query, context=summary_context)

        return {
            'query': query,
            'tables_used': [query_result['original_filename']],
            'sql': query_result['sql'],
            'results': query_result['results'],
            'insights': insights,
            'summary': summary_result['summary'],
            'key_points': summary_result['key_points'],
            'workflow': 'simple'
        }


# Test the orchestrator
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Sales Intelligence Orchestrator")
    print("=" * 80)

    # Setup
    db = DuckDBManager()
    db.load_csv_files("/Users/shashank/Desktop/AI/Blend360Assignment/Sales Dataset")

    indexer = MetadataIndexer(db)
    indexer.index_all_tables()

    # Create orchestrator
    orchestrator = SalesIntelligenceOrchestrator(db, indexer, use_llm=True)

    # Test queries
    test_queries = [
        "What is the total amount in international sales?",
        "Show me top 5 customers from international sales",
        "Compare sales by category in Amazon",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"🎯 QUERY: {query}")
        print('=' * 80)

        result = orchestrator.process_query(query)

        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
        else:
            print(f"\n📁 Tables Used: {', '.join(result['tables_used'])}")
            print(f"\n🔍 SQL:\n{result['sql']}")

            if result.get('results') is not None:
                print(f"\n📊 Results ({len(result['results'])} rows):")
                print(result['results'].head(10).to_string(index=False))

            if result.get('insights'):
                print(f"\n💡 Insights:")
                for insight in result['insights']:
                    print(f"  • {insight}")

            print(f"\n📝 Summary:")
            print(f"  {result['summary']}")

    db.close()

    print(f"\n{'=' * 80}")
    print("✅ Orchestrator test complete!")
    print('=' * 80)