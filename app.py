"""
Agteria Literature Scout - Streamlit Web Interface

A web-based interface for the intelligent literature discovery system.
"""

import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.agents.literature_scout import LiteratureScout
from src.utils.report_generator import ReportGenerator

# Configure page
st.set_page_config(
    page_title="Agteria Literature Scout",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #333333;
    }
    .hypothesis-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        color: #333333;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #333333;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scout_initialized' not in st.session_state:
    st.session_state.scout_initialized = False
if 'scout' not in st.session_state:
    st.session_state.scout = None
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'last_scan_results' not in st.session_state:
    st.session_state.last_scan_results = None

def initialize_scout():
    """Initialize the Literature Scout agent."""
    if st.session_state.scout_initialized:
        return True
    
    try:
        with st.spinner("Initializing Literature Scout..."):
            # Validate API keys
            if not Config.validate_keys():
                st.error("❌ Missing required API keys. Please check your .env file.")
                st.info("💡 Make sure you have set OPENAI_API_KEY in your .env file.")
                return False
            
            # Initialize components
            st.session_state.scout = LiteratureScout(verbose=False)
            st.session_state.report_generator = ReportGenerator()
            st.session_state.scout_initialized = True
            
            st.success("✅ Literature Scout initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize Literature Scout: {e}")
        return False

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Agteria Literature Scout</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent Research Discovery for Climate Technology**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Control Panel")
        
        # Initialize button
        if st.button("🚀 Initialize Scout", disabled=st.session_state.scout_initialized):
            initialize_scout()
        
        if st.session_state.scout_initialized:
            st.success("✅ Scout Ready")
            
            # Show status
            try:
                status = st.session_state.scout.get_agent_status()
                st.markdown("### 📊 Status")
                st.metric("Papers in Memory", status.get('papers_in_memory', 0))
                st.metric("Available Tools", status.get('available_tools', 0))
                
                if status.get('current_focus'):
                    st.info(f"🎯 Current Focus: {status['current_focus']}")
                
            except Exception as e:
                st.error(f"Error getting status: {e}")
        else:
            st.warning("⚠️ Scout not initialized")
        
        # Navigation
        st.markdown("### 📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🔍 Research", "📊 Daily Scan", "🚀 Breakthrough Analysis", 
             "🔍 Gap Explorer", "👥 Competitor Tracking", "📈 Analytics", "⚙️ Settings"]
        )
    
    # Main content based on selected page
    if not st.session_state.scout_initialized and page != "⚙️ Settings":
        st.warning("⚠️ Please initialize the Literature Scout first using the sidebar.")
        return
    
    if page == "🔍 Research":
        research_page()
    elif page == "📊 Daily Scan":
        daily_scan_page()
    elif page == "🚀 Breakthrough Analysis":
        breakthrough_analysis_page()
    elif page == "🔍 Gap Explorer":
        gap_explorer_page()
    elif page == "👥 Competitor Tracking":
        competitor_tracking_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

def research_page():
    """Research page for conducting literature searches."""
    st.markdown("## 🔍 Research Assistant")
    st.markdown("Conduct intelligent research on any topic related to methane reduction and climate technology.")
    
    # Research form
    with st.form("research_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Research Query:",
                placeholder="e.g., novel methane inhibitors for cattle",
                help="Enter your research question or topic"
            )
        
        with col2:
            focus_areas = st.multiselect(
                "Focus Areas:",
                ["Novel mechanisms", "Practical applications", "Cross-domain insights", 
                 "Molecular targets", "Scalability", "Commercial viability"],
                help="Select specific areas to focus on"
            )
        
        submitted = st.form_submit_button("🔍 Start Research", use_container_width=True)
    
    if submitted and query:
        with st.spinner(f"Researching: {query}..."):
            try:
                result = st.session_state.scout.conduct_research(query, focus_areas)
                
                # Store in history
                st.session_state.research_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'result': result
                })
                
                # Display results
                display_research_results(result, 0)
                
            except Exception as e:
                st.error(f"❌ Research failed: {e}")
    
    # Research history
    if st.session_state.research_history:
        st.markdown("## 📚 Research History")
        
        for i, item in enumerate(reversed(st.session_state.research_history[-5:]), 1):
            with st.expander(f"{item['timestamp'].strftime('%Y-%m-%d %H:%M')} - {item['query'][:50]}..."):
                display_research_results(item['result'], i)

def display_research_results(result: Dict[str, Any], index: int = 0):
    """Display research results in a structured format."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Papers Found", result.get('papers_found', 0))
    with col2:
        st.metric("Novel Insights", len(result.get('novel_insights', [])))
    with col3:
        st.metric("Hypotheses", len(result.get('hypotheses', [])))
    with col4:
        st.metric("Next Steps", len(result.get('next_steps', [])))
    
    # Main response
    st.markdown("### 📄 Research Findings")
    st.markdown(result.get('response', 'No response available'))
    
    # Novel insights
    if result.get('novel_insights'):
        st.markdown("### 💡 Novel Insights")
        for i, insight in enumerate(result['novel_insights'], 1):
            st.markdown(f'<div class="insight-box">{i}. {insight}</div>', unsafe_allow_html=True)
    
    # Hypotheses
    if result.get('hypotheses'):
        st.markdown("### 🧪 Generated Hypotheses")
        for i, hypothesis in enumerate(result['hypotheses'], 1):
            st.markdown(f'<div class="hypothesis-box">{i}. {hypothesis}</div>', unsafe_allow_html=True)
    
    # Next steps
    if result.get('next_steps'):
        st.markdown("### 📋 Recommended Next Steps")
        for i, step in enumerate(result['next_steps'], 1):
            st.markdown(f"{i}. {step}")
    
    # Download results
    if st.button("💾 Download Results", key=f"download_{index}"):
        json_str = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def daily_scan_page():
    """Daily scan page for automated research monitoring."""
    st.markdown("## 📊 Daily Research Scan")
    st.markdown("Automated daily scanning of scientific literature for breakthrough discoveries.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Custom queries
        st.markdown("### 🔧 Custom Scan Configuration")
        custom_queries = st.text_area(
            "Additional Queries (one per line):",
            placeholder="marine biology methane\npharmaceutical enzyme inhibitors\nindustrial catalyst methane",
            help="Add custom search queries beyond the default Agteria-focused searches"
        )
        
        # Scan options
        generate_report = st.checkbox("📄 Generate Report", value=True)
        
    with col2:
        # Last scan info
        if st.session_state.last_scan_results:
            st.markdown("### 📈 Last Scan")
            last_scan = st.session_state.last_scan_results
            st.metric("Discoveries", len(last_scan.get('novel_discoveries', [])))
            st.metric("Hypotheses", len(last_scan.get('generated_hypotheses', [])))
            st.metric("Queries", last_scan.get('queries_processed', 0))
    
    # Start scan button
    if st.button("🌅 Start Daily Scan", use_container_width=True):
        
        # Prepare custom queries
        custom_query_list = None
        if custom_queries.strip():
            custom_query_list = [q.strip() for q in custom_queries.split('\n') if q.strip()]
        
        with st.spinner("Performing daily research scan... This may take several minutes."):
            try:
                scan_results = st.session_state.scout.daily_research_scan(custom_query_list)
                st.session_state.last_scan_results = scan_results
                
                # Display scan results
                display_scan_results(scan_results, generate_report, index=1)
                
            except Exception as e:
                st.error(f"❌ Daily scan failed: {e}")
    
    # Display last scan results if available
    if st.session_state.last_scan_results:
        st.markdown("---")
        st.markdown("## 📊 Latest Scan Results")
        display_scan_results(st.session_state.last_scan_results, show_generate_button=False, index=2)

def display_scan_results(scan_results: Dict[str, Any], generate_report: bool = False, show_generate_button: bool = True, index: int = 0):
    """Display daily scan results."""
    
    # Summary
    st.markdown("### 📈 Scan Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Queries Processed", scan_results.get('queries_processed', 0))
    with col2:
        st.metric("Novel Discoveries", len(scan_results.get('novel_discoveries', [])))
    with col3:
        st.metric("Generated Hypotheses", len(scan_results.get('generated_hypotheses', [])))
    
    # Summary text
    if scan_results.get('summary'):
        st.markdown("### 📝 Executive Summary")
        st.markdown(scan_results['summary'])
    
    # Novel discoveries
    discoveries = scan_results.get('novel_discoveries', [])
    if discoveries:
        st.markdown("### 🔬 Novel Discoveries")
        for i, discovery in enumerate(discoveries, 1):
            st.markdown(f'<div class="insight-box"><strong>Discovery {i}:</strong> {discovery}</div>', unsafe_allow_html=True)
    
    # Generated hypotheses
    hypotheses = scan_results.get('generated_hypotheses', [])
    if hypotheses:
        st.markdown("### 💡 Generated Hypotheses")
        for i, hypothesis in enumerate(hypotheses, 1):
            st.markdown(f'<div class="hypothesis-box"><strong>Hypothesis {i}:</strong> {hypothesis}</div>', unsafe_allow_html=True)
    
    # Generate report
    if (generate_report or show_generate_button) and st.session_state.report_generator:
        if st.button("📄 Generate Detailed Report", key=f"report_{index}"):
            with st.spinner("Generating report..."):
                try:
                    report_path = st.session_state.report_generator.generate_daily_digest(scan_results)
                    if report_path:
                        st.success(f"✅ Report generated: {report_path}")
                        
                        # Read and display report
                        try:
                            with open(report_path, 'r') as f:
                                report_content = f.read()
                            
                            st.markdown("### 📄 Generated Report")
                            st.markdown(report_content)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Report",
                                data=report_content,
                                file_name=f"daily_digest_{datetime.now().strftime('%Y%m%d')}.md",
                                mime="text/markdown"
                            )
                            
                        except Exception as e:
                            st.error(f"Error reading report: {e}")
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")

def breakthrough_analysis_page():
    """Breakthrough analysis page."""
    st.markdown("## 🚀 Breakthrough Potential Analysis")
    st.markdown("Analyze the breakthrough potential of research findings for commercial applications.")
    
    # Input form
    with st.form("breakthrough_form"):
        findings = st.text_area(
            "Research Findings:",
            placeholder="Enter research findings, discoveries, or technologies to analyze...",
            height=150,
            help="Describe the research findings you want to analyze for breakthrough potential"
        )
        
        submitted = st.form_submit_button("🚀 Analyze Breakthrough Potential", use_container_width=True)
    
    if submitted and findings:
        with st.spinner("Analyzing breakthrough potential..."):
            try:
                result = st.session_state.scout.analyze_breakthrough_potential(findings)
                
                # Display analysis
                st.markdown("### 🎯 Breakthrough Analysis")
                st.markdown(result.get('response', 'No analysis available'))
                
                # Additional insights
                if result.get('novel_insights'):
                    st.markdown("### 💡 Key Insights")
                    for insight in result['novel_insights']:
                        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                
                # Recommendations
                if result.get('next_steps'):
                    st.markdown("### 📋 Recommendations")
                    for step in result['next_steps']:
                        st.markdown(f"• {step}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

def gap_explorer_page():
    """Research gap exploration page."""
    st.markdown("## 🔍 Research Gap Explorer")
    st.markdown("Identify unexplored research opportunities and generate novel hypotheses.")
    
    # Input form
    with st.form("gap_form"):
        research_area = st.text_input(
            "Research Area:",
            placeholder="e.g., methane inhibition in ruminants",
            help="Enter the research area you want to explore for gaps"
        )
        
        submitted = st.form_submit_button("🔍 Explore Research Gaps", use_container_width=True)
    
    if submitted and research_area:
        with st.spinner(f"Exploring research gaps in {research_area}..."):
            try:
                result = st.session_state.scout.explore_research_gaps(research_area)
                
                # Display gap analysis
                st.markdown("### 🎯 Gap Analysis")
                st.markdown(result.get('response', 'No analysis available'))
                
                # Identified opportunities
                if result.get('novel_insights'):
                    st.markdown("### 🔬 Identified Opportunities")
                    for insight in result['novel_insights']:
                        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                
                # Generated hypotheses
                if result.get('hypotheses'):
                    st.markdown("### 💡 Research Hypotheses to Fill Gaps")
                    for hypothesis in result['hypotheses']:
                        st.markdown(f'<div class="hypothesis-box">{hypothesis}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Gap exploration failed: {e}")

def competitor_tracking_page():
    """Competitor tracking page."""
    st.markdown("## 👥 Competitor Intelligence")
    st.markdown("Track competitor research and identify market opportunities.")
    
    # Competitor input
    with st.form("competitor_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            competitors_text = st.text_area(
                "Competitor Companies (one per line):",
                placeholder="DSM\nCargill\nAlltech\nBovaer",
                help="Enter competitor company names to track"
            )
        
        with col2:
            track_patents = st.checkbox("📄 Track Patents", value=True)
            track_publications = st.checkbox("📚 Track Publications", value=True)
            track_news = st.checkbox("📰 Track News", value=True)
        
        submitted = st.form_submit_button("👥 Track Competitors", use_container_width=True)
    
    if submitted and competitors_text.strip():
        competitors = [c.strip() for c in competitors_text.split('\n') if c.strip()]
        
        with st.spinner(f"Tracking {len(competitors)} competitors..."):
            try:
                intelligence = st.session_state.scout.track_competitor_research(competitors)
                
                # Display intelligence summary
                st.markdown("### 📊 Intelligence Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Competitors Analyzed", len(intelligence.get('competitors_analyzed', [])))
                with col2:
                    st.metric("Findings", len(intelligence.get('findings', [])))
                with col3:
                    st.metric("Opportunities", len(intelligence.get('collaboration_opportunities', [])))
                
                # Key findings
                findings = intelligence.get('findings', [])
                if findings:
                    st.markdown("### 🔍 Key Findings")
                    for finding in findings[:5]:  # Show top 5
                        query = finding.get('query', 'Unknown Query')
                        response = finding.get('response', 'No response')[:300]
                        st.markdown(f"**Query:** {query}")
                        st.markdown(f"**Finding:** {response}...")
                        st.markdown("---")
                
                # Threats and opportunities
                threats = intelligence.get('competitive_threats', [])
                opportunities = intelligence.get('collaboration_opportunities', [])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if threats:
                        st.markdown("### ⚠️ Competitive Threats")
                        for threat in threats:
                            st.markdown(f'<div class="warning-box">{threat}</div>', unsafe_allow_html=True)
                
                with col2:
                    if opportunities:
                        st.markdown("### 🤝 Collaboration Opportunities")
                        for opportunity in opportunities:
                            st.markdown(f'<div class="success-box">{opportunity}</div>', unsafe_allow_html=True)
                
                # Generate report
                if st.button("📄 Generate Intelligence Report"):
                    with st.spinner("Generating competitive intelligence report..."):
                        try:
                            report_path = st.session_state.report_generator.generate_competitive_intelligence(intelligence)
                            if report_path:
                                st.success(f"✅ Report generated: {report_path}")
                                
                                # Show download option
                                with open(report_path, 'r') as f:
                                    report_content = f.read()
                                
                                st.download_button(
                                    label="📥 Download Intelligence Report",
                                    data=report_content,
                                    file_name=f"competitive_intelligence_{datetime.now().strftime('%Y%m%d')}.md",
                                    mime="text/markdown"
                                )
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
                
            except Exception as e:
                st.error(f"❌ Competitor tracking failed: {e}")

def analytics_page():
    """Analytics and visualization page."""
    st.markdown("## 📈 Research Analytics")
    st.markdown("Visualize research trends and insights from your Literature Scout activities.")
    
    try:
        # Get memory statistics
        if st.session_state.scout:
            memory_stats = st.session_state.scout.memory.get_memory_stats()
            
            # Overview metrics
            st.markdown("### 📊 Memory Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", memory_stats.get('total_documents', 0))
            with col2:
                st.metric("Total Analyses", memory_stats.get('total_analyses', 0))
            with col3:
                st.metric("Total Searches", memory_stats.get('total_searches', 0))
            with col4:
                avg_quality = memory_stats.get('average_quality_score', 0)
                st.metric("Avg Quality Score", f"{avg_quality:.2f}")
            
            # Documents by source
            sources = memory_stats.get('documents_by_source', {})
            if sources:
                st.markdown("### 📚 Documents by Source")
                
                # Create pie chart
                fig = px.pie(
                    values=list(sources.values()),
                    names=list(sources.keys()),
                    title="Distribution of Papers by Source Database"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Research history trends
            if st.session_state.research_history:
                st.markdown("### 🔍 Research Activity")
                
                # Create timeline of research queries
                history_df = pd.DataFrame([
                    {
                        'timestamp': item['timestamp'],
                        'query': item['query'][:50] + '...' if len(item['query']) > 50 else item['query'],
                        'papers_found': item['result'].get('papers_found', 0),
                        'insights': len(item['result'].get('novel_insights', [])),
                        'hypotheses': len(item['result'].get('hypotheses', []))
                    }
                    for item in st.session_state.research_history
                ])
                
                if not history_df.empty:
                    # Timeline chart
                    fig = px.scatter(
                        history_df,
                        x='timestamp',
                        y='papers_found',
                        size='insights',
                        color='hypotheses',
                        hover_data=['query'],
                        title="Research Activity Timeline",
                        labels={
                            'timestamp': 'Time',
                            'papers_found': 'Papers Found',
                            'insights': 'Novel Insights',
                            'hypotheses': 'Hypotheses Generated'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary table
                    st.markdown("### 📋 Recent Research Summary")
                    st.dataframe(history_df[['timestamp', 'query', 'papers_found', 'insights', 'hypotheses']])
            
            # Quality distribution
            st.markdown("### 📊 Quality Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                min_quality = memory_stats.get('min_quality_score', 0)
                max_quality = memory_stats.get('max_quality_score', 1)
                
                st.metric("Quality Range", f"{min_quality:.2f} - {max_quality:.2f}")
                
                # Quality gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_quality,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Average Quality Score"},
                    gauge = {'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "yellow"},
                                {'range': [0.8, 1], 'color': "green"}],
                            'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 0.9}}))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Research Recommendations")
                
                if memory_stats.get('total_documents', 0) < 10:
                    st.info("💡 Run more daily scans to build your research database")
                
                if avg_quality < 0.7:
                    st.warning("⚠️ Consider refining search queries for higher quality papers")
                
                if not st.session_state.research_history:
                    st.info("🔍 Start conducting research to see activity analytics")
                
                if memory_stats.get('total_documents', 0) > 100:
                    st.success("🎉 Great! You have a substantial research database")
        
        else:
            st.warning("⚠️ Literature Scout not initialized. Please initialize to view analytics.")
    
    except Exception as e:
        st.error(f"❌ Error loading analytics: {e}")

def settings_page():
    """Settings and configuration page."""
    st.markdown("## ⚙️ Settings & Configuration")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Required API Keys")
        
        # Check OpenAI key
        openai_status = "✅ Configured" if Config.OPENAI_API_KEY else "❌ Missing"
        st.markdown(f"**OpenAI API Key:** {openai_status}")
        
        if not Config.OPENAI_API_KEY:
            st.warning("OpenAI API key is required for the Literature Scout to function.")
            st.markdown("1. Get your API key from [OpenAI](https://platform.openai.com/api-keys)")
            st.markdown("2. Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`")
    
    with col2:
        st.markdown("#### Optional API Keys")
        
        # Check Serper key
        serper_status = "✅ Configured" if Config.SERPER_API_KEY else "❌ Missing"
        st.markdown(f"**Serper API Key (Web Search):** {serper_status}")
        
        if not Config.SERPER_API_KEY:
            st.info("Serper API key enables enhanced web search capabilities.")
            st.markdown("Get your key from [Serper](https://serper.dev/)")
    
    # System Configuration
    st.markdown("### 🛠️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Settings")
        st.text(f"Model: {Config.DEFAULT_MODEL}")
        st.text(f"Temperature: {Config.TEMPERATURE}")
        st.text(f"Max ArXiv Results: {Config.MAX_ARXIV_RESULTS}")
        st.text(f"Max PubMed Results: {Config.MAX_PUBMED_RESULTS}")
        st.text(f"Chunk Size: {Config.CHUNK_SIZE}")
    
    with col2:
        st.markdown("#### Storage Paths")
        st.text(f"Vector DB: {Config.VECTOR_DB_PATH}")
        st.text(f"Reports: {Config.REPORTS_DIR}")
        st.text(f"Collection: {Config.COLLECTION_NAME}")
    
    # Memory Management
    st.markdown("### 💾 Memory Management")
    
    if st.session_state.scout_initialized:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Old Documents"):
                with st.spinner("Clearing old documents..."):
                    try:
                        st.session_state.scout.memory.clear_old_documents(days=90)
                        st.success("✅ Old documents cleared")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("💾 Backup Memory"):
                with st.spinner("Creating backup..."):
                    try:
                        backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.scout.memory.backup_memory(backup_path)
                        st.success(f"✅ Backup created: {backup_path}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col3:
            if st.button("📊 Memory Stats"):
                try:
                    stats = st.session_state.scout.memory.get_memory_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Help & Documentation
    st.markdown("### 📚 Help & Documentation")
    
    with st.expander("🚀 Getting Started"):
        st.markdown("""
        1. **Setup API Keys**: Add your OpenAI API key to the `.env` file
        2. **Initialize Scout**: Click the "Initialize Scout" button in the sidebar
        3. **Start Researching**: Use the Research page to query scientific literature
        4. **Daily Scans**: Set up automated daily scans to monitor new publications
        5. **Analyze Results**: Use the breakthrough analysis to evaluate findings
        """)
    
    with st.expander("🔍 Research Tips"):
        st.markdown("""
        - **Be Specific**: Use specific terms like "methane inhibitors cattle" rather than general terms
        - **Cross-Domain**: Explore connections between different fields (e.g., "marine biology methane")
        - **Focus Areas**: Select relevant focus areas to guide the research direction
        - **Regular Scans**: Run daily scans to stay updated with the latest research
        - **Gap Analysis**: Use the gap explorer to find unexplored research opportunities
        """)
    
    with st.expander("⚙️ Configuration Options"):
        st.markdown("""
        - **Temperature**: Controls creativity vs. consistency (0.0-1.0)
        - **Search Limits**: Adjust maximum results from each database
        - **Quality Filters**: Set minimum quality scores for papers
        - **Memory Retention**: Configure how long to keep analyzed papers
        - **Report Formats**: Choose between Markdown, HTML, or JSON outputs
        """)

if __name__ == "__main__":
    main()