import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from storage.duckdb_manager import DuckDBManager
from storage.vector_store import MetadataIndexer
from src.agents.orchestrator import SalesIntelligenceOrchestrator
import pandas as pd
import plotly.express as px
import time

# Page config
st.set_page_config(
    page_title="Sales Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .agent-step {
        padding: 1rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
        background-color: #f0f8ff;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_system():
    """Initialize the system with existing data"""
    with st.spinner("🚀 Initializing Sales Intelligence System..."):
        try:
            # Initialize DuckDB
            st.session_state.db_manager = DuckDBManager()
            st.session_state.db_manager.load_csv_files("Sales Dataset")

            # Initialize Vector Store
            indexer = MetadataIndexer(st.session_state.db_manager, use_weaviate=True)
            indexer.index_all_tables()

            # Initialize Orchestrator
            st.session_state.orchestrator = SalesIntelligenceOrchestrator(
                st.session_state.db_manager,
                indexer,
                use_llm=True  # Use rule-based for faster demo
            )

            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            return False


def process_query(query: str):
    """Process user query through orchestrator"""

    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    # Create placeholder for assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your query..."):

            # Process query
            result = st.session_state.orchestrator.process_query(query)

            # Display response
            if result.get('error'):
                st.error(f"❌ {result['error']}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {result['error']}"
                })
            else:
                # Show summary
                st.markdown(f"**📝 Summary:**")
                st.write(result.get('summary', 'Query executed successfully'))

                # Show SQL (expandable)
                with st.expander("🔍 SQL Query"):
                    st.code(result['sql'], language='sql')

                # Show results
                if result.get('results') is not None:
                    st.markdown(f"**📊 Results ({len(result['results'])} rows):**")
                    st.dataframe(result['results'], width='stretch')

                    # Show insights if available
                    if result.get('insights'):
                        st.markdown("**💡 Insights:**")
                        for insight in result['insights']:
                            st.markdown(f"- {insight}")

                    # Add visualization if numeric data
                    numeric_cols = result['results'].select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0 and len(result['results']) > 1:
                        st.markdown("**📈 Visualization:**")

                        # Choose appropriate chart
                        if len(result['results']) <= 20:
                            fig = px.bar(
                                result['results'].head(10),
                                x=result['results'].columns[0],
                                y=numeric_cols[0],
                                title=f"{numeric_cols[0]} by {result['results'].columns[0]}"
                            )
                            st.plotly_chart(fig, width='stretch')

                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result.get('summary', ''),
                    "data": result
                })


# Main UI
def main():
    # Header
    st.markdown('<div class="main-header">📊 Sales Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent AI System for Sales Data Analysis</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Controls")

        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", type="primary", width='stretch'):
                if initialize_system():
                    st.success("✅ System initialized!")
                    st.rerun()
        else:
            st.success("✅ System Ready")

            # System info
            st.divider()
            st.subheader("📁 Loaded Data")
            if st.session_state.db_manager:
                tables = st.session_state.db_manager.get_all_tables()
                for table in tables:
                    info = st.session_state.db_manager.get_table_info(table)
                    st.markdown(f"**{info['original_filename']}**")
                    st.caption(f"{info['row_count']:,} rows")

            st.divider()

            # Query examples
            st.subheader("💡 Example Queries")
            example_queries = [
                "What is the total amount in international sales?",
                "Show me top 5 customers from international sales",
                "How many cancelled orders in Amazon sales?",
                "Compare sales by category in Amazon",
                "What is the total sales for Amazon?"
            ]

            for query in example_queries:
                if st.button(query, key=query, width='stretch'):
                    st.session_state.query_input = query
                    st.rerun()

            st.divider()

            # Clear chat
            if st.button("🗑️ Clear Chat", width='stretch'):
                st.session_state.chat_history = []
                st.rerun()

    # Main content
    if not st.session_state.initialized:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.info("👈 Click **Initialize System** in the sidebar to get started!")

            st.markdown("---")

            st.markdown("### 🎯 What This System Can Do")
            st.markdown("""
            - **🔍 Semantic Search**: Automatically finds relevant data tables
            - **💬 Natural Language Queries**: Ask questions in plain English
            - **📊 Smart Analytics**: Provides insights and trends
            - **🤖 Multi-Agent Architecture**: Coordinated AI agents working together
            - **📈 Visualizations**: Automatic charts and graphs
            """)

            st.markdown("### 🏗️ Architecture")
            st.markdown("""
            1. **Metadata Agent** → Finds relevant tables using Weaviate
            2. **Query Agent** → Generates and executes SQL
            3. **Analytics Agent** → Analyzes results for patterns
            4. **Summarization Agent** → Creates natural language summaries
            5. **Orchestrator** → Coordinates all agents with LangGraph
            """)

    else:
        # Chat interface
        st.subheader("💬 Chat Interface")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                # Show data if available
                if message["role"] == "assistant" and "data" in message:
                    data = message["data"]
                    if data.get('results') is not None:
                        with st.expander("View Results"):
                            st.dataframe(data['results'])

        # Chat input
        if prompt := st.chat_input("Ask a question about your sales data..."):
            process_query(prompt)
            st.rerun()

if __name__ == "__main__":
    main()