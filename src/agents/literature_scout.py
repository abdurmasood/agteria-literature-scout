"""Main Literature Scout agent using ReAct framework for intelligent research assistance."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain.agents import AgentType, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import FileCallbackHandler
from langchain_core.documents import Document

from ..config import Config, AGTERIA_KEYWORDS, CROSS_DOMAIN_KEYWORDS
from ..tools.search_tools import create_langchain_search_tools
from ..tools.analysis_tools import create_langchain_analysis_tools
from ..tools.hypothesis_tools import create_langchain_hypothesis_tools
from ..memory.research_memory import ResearchMemory, KnowledgeGraph
from ..processors.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiteratureScout:
    """Intelligent Literature Scout agent for automated research discovery."""
    
    def __init__(self, memory_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the Literature Scout agent.
        
        Args:
            memory_path: Optional custom path for memory storage
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize core components
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize memory and processors
        self.memory = ResearchMemory(persist_directory=memory_path)
        self.knowledge_graph = KnowledgeGraph(self.memory)
        self.document_processor = DocumentProcessor()
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Initialize agent memory
        self.agent_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True
        )
        
        # Create the agent
        self.agent = self._create_agent()
        
        # Initialize session state
        self.session_state = {
            "current_research_focus": None,
            "recent_discoveries": [],
            "hypothesis_count": 0,
            "papers_analyzed": 0
        }
        
        logger.info("Literature Scout agent initialized successfully")
    
    def _setup_tools(self) -> List:
        """Set up all available tools for the agent."""
        tools = []
        
        # Add search tools
        tools.extend(create_langchain_search_tools())
        
        # Add analysis tools
        tools.extend(create_langchain_analysis_tools())
        
        # Add hypothesis generation tools
        tools.extend(create_langchain_hypothesis_tools())
        
        # Add memory tools
        tools.extend(self._create_memory_tools())
        
        logger.info(f"Initialized {len(tools)} tools for the agent")
        return tools
    
    def _create_memory_tools(self) -> List:
        """Create tools for interacting with memory."""
        from langchain_core.tools import Tool
        
        def search_memory_func(query: str) -> str:
            """Search stored papers in memory."""
            results = self.memory.search_papers(query, k=5)
            if not results:
                return f"No papers found in memory for: {query}"
            
            formatted = f"Found {len(results)} papers in memory:\n\n"
            for i, doc in enumerate(results, 1):
                title = doc.metadata.get('title', 'Unknown Title')
                authors = doc.metadata.get('authors', 'Unknown Authors')
                quality = doc.metadata.get('quality_score', 0)
                formatted += f"{i}. {title}\n"
                formatted += f"   Authors: {authors}\n"
                formatted += f"   Quality Score: {quality:.2f}\n"
                formatted += f"   Preview: {doc.page_content[:200]}...\n\n"
            
            return formatted
        
        def get_memory_stats_func(_: str) -> str:
            """Get memory statistics."""
            stats = self.memory.get_memory_stats()
            formatted = "Memory Statistics:\n\n"
            formatted += f"📚 Total Documents: {stats.get('total_documents', 0)}\n"
            formatted += f"🔍 Total Analyses: {stats.get('total_analyses', 0)}\n"
            formatted += f"📊 Average Quality: {stats.get('average_quality_score', 0):.2f}\n"
            
            sources = stats.get('documents_by_source', {})
            if sources:
                formatted += "\nDocuments by Source:\n"
                for source, count in sources.items():
                    formatted += f"  • {source}: {count}\n"
            
            return formatted
        
        def find_related_concepts_func(concept: str) -> str:
            """Find concepts related to the given concept."""
            related = self.knowledge_graph.find_related_concepts(concept, max_related=8)
            if not related:
                return f"No related concepts found for: {concept}"
            
            formatted = f"Concepts related to '{concept}':\n\n"
            for related_concept, co_occurrence in related:
                formatted += f"• {related_concept} (appears together in {co_occurrence} papers)\n"
            
            return formatted
        
        return [
            Tool(
                name="search_memory",
                description="Search previously analyzed papers stored in memory. Use this to avoid re-analyzing the same papers.",
                func=search_memory_func
            ),
            Tool(
                name="memory_stats",
                description="Get statistics about papers and analyses stored in memory.",
                func=get_memory_stats_func
            ),
            Tool(
                name="find_related_concepts",
                description="Find concepts related to a given research concept based on co-occurrence in papers.",
                func=find_related_concepts_func
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with custom prompt."""
        
        react_prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template="""You are the Agteria Literature Scout, an expert AI research assistant specializing in climate technology and methane reduction in livestock. Your primary goal is to help Agteria Biotech discover novel research insights, identify breakthrough opportunities, and generate innovative hypotheses.

CORE EXPERTISE:
- Methane inhibition in cattle and livestock
- Enzyme inhibitors and fermentation control
- Agricultural climate solutions
- Cross-domain innovation (marine biology, pharmaceuticals, materials science)
- Scientific literature analysis and synthesis

AVAILABLE TOOLS:
{tools}

Tool Names: {tool_names}

CRITICAL SOURCE ATTRIBUTION REQUIREMENTS:
🚨 MANDATORY: When you use search tools and get paper results with IDs, you MUST cite them in your final answer.

CITATION FORMAT: Use [ID: paper_id] after each claim
✅ EXAMPLE: "Marine enzymes reduce methane by 45% [ID: arxiv_2024001] and show safety in trials [ID: pubmed_12345]"

IMPORTANT: Always include paper IDs from tool responses in your final answer.

RESEARCH APPROACH:
1. SEARCH STRATEGY: Use multiple databases (ArXiv, PubMed, Web) to ensure comprehensive coverage
2. ANALYSIS DEPTH: Thoroughly analyze papers for novel mechanisms, molecules, and methodologies
3. CROSS-DOMAIN THINKING: Actively look for connections between different research fields
4. HYPOTHESIS GENERATION: After analyzing papers, use hypothesis generation tools:
   - Use 'cross_domain_hypotheses' tool with format: 'Domain1: findings1 | Domain2: findings2'
   - Use 'analogical_hypotheses' tool with format: 'source_mechanism | source_domain'
   - Use 'creative_ideation' tool for breakthrough ideas
   - Use 'research_gap_analysis' tool with format: 'research_area | existing_approaches | limitations'
5. MEMORY UTILIZATION: Check memory to avoid duplicate work and build on previous discoveries
6. SOURCE TRACKING: Pay attention to Paper IDs returned by search tools and reference them in your analysis

RESPONSE FORMAT:
Use this format for your reasoning:

Thought: [Your reasoning about what to do next - note any Paper IDs from previous tool responses]
Action: [The tool to use]
Action Input: [The input to the tool]
Observation: [The tool's response - extract Paper IDs if present]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to provide a comprehensive response with full source attribution
Final Answer: [Your detailed analysis with proper citations for EVERY claim, insight, and hypothesis]

KEY PRIORITIES FOR AGTERIA:
- Novel methane inhibition mechanisms
- Scalable and cost-effective solutions
- Safety and efficacy in livestock
- Cross-industry applicable technologies
- Breakthrough molecular discoveries

FINAL ANSWER STRUCTURE:
Your Final Answer must be formatted as clean markdown with the following sections:

## Overview
Overview of the research query and the main findings.

## 💡 Novel Insights

1. First insight with [Paper ID: title] citation
2. Second insight with [Paper ID: title] citation

## 🧪 Generated Hypotheses

1. First hypothesis with supporting [Paper ID: title] citations
2. Second hypothesis with supporting [Paper ID: title] citations

## 📋 Recommended Next Steps

1. First action step with relevant source references
2. Second action step with relevant source references

## 📖 Source Summary

List of all Paper IDs referenced with their titles

Current Request: {input}

Begin your research and remember: NO CLAIM WITHOUT CITATION!

{agent_scratchpad}"""
        )
        
        # Create React agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        # Create agent executor with increased limits for source attribution workflow
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.agent_memory,
            verbose=self.verbose,
            max_iterations=25,  # Increased for source attribution workflow
            max_execution_time=600,  # 10 minutes timeout for comprehensive research
            handle_parsing_errors=True,
            early_stopping_method="generate"  # Allow agent to stop early if task is complete
        )
        
        return agent_executor
    
    def conduct_research(self, research_query: str, focus_areas: Optional[List[str]] = None, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a query.
        
        Args:
            research_query: The research question or topic
            focus_areas: Optional specific areas to focus on
            callbacks: Optional list of callback handlers for streaming
        
        Returns:
            Dictionary with research results and insights
        """
        logger.info(f"Starting research on: {research_query}")
        
        # Update session state
        self.session_state["current_research_focus"] = research_query
        
        # Enhance query with Agteria context
        enhanced_query = self._enhance_query_with_context(research_query, focus_areas)
        
        try:
            logger.info(f"Starting agent execution for query: {research_query}")
            
            # Create a temporary agent executor with callbacks if provided
            if callbacks:
                # Create new agent executor with callbacks
                temp_agent_executor = AgentExecutor(
                    agent=self.agent.agent,
                    tools=self.tools,
                    memory=self.agent_memory,
                    verbose=self.verbose,
                    max_iterations=25,
                    max_execution_time=600,
                    handle_parsing_errors=True,
                    early_stopping_method="generate",
                    callbacks=callbacks
                )
                result = temp_agent_executor.invoke({"input": enhanced_query})
            else:
                # Use existing agent executor
                result = self.agent.invoke({"input": enhanced_query})
            
            # Check if agent completed successfully
            agent_output = result.get("output", "")
            if not agent_output or "Agent stopped due to" in agent_output:
                logger.warning(f"Agent may have hit limits. Output: {agent_output[:200]}...")
                
                # Return partial results with helpful message
                return {
                    "query": research_query,
                    "response": agent_output or "Agent execution incomplete - may have hit iteration or time limits",
                    "error": "Agent execution incomplete",
                    "papers_found": 0,
                    "novel_insights": [],
                    "hypotheses": [],
                    "next_steps": ["Try a more specific query", "Check API keys and network connectivity"],
                    "sources": [],
                    "paper_ids_referenced": [],
                    "timestamp": datetime.now().isoformat(),
                    "troubleshooting_tips": [
                        "The agent may need more time for complex research queries",
                        "Try breaking down the query into smaller, more specific questions",
                        "Check that all API keys are properly configured",
                        "Verify network connectivity to research databases"
                    ]
                }
            
            # Process and structure the results
            structured_result = self._structure_research_result(
                query=research_query,
                agent_response=agent_output,
                focus_areas=focus_areas
            )
            
            # Update knowledge graph
            self._update_knowledge_from_research(structured_result)
            
            logger.info(f"Research completed successfully for: {research_query}")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            error_result = {
                "query": research_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "troubleshooting_tips": [
                    "Check API keys in .env file",
                    "Verify network connectivity", 
                    "Try a simpler query first",
                    "Check agent logs for detailed error information"
                ]
            }
            
            # Add specific guidance based on error type
            if "API" in str(e):
                error_result["troubleshooting_tips"].insert(0, "API key issue detected - check OPENAI_API_KEY in .env")
            elif "timeout" in str(e).lower():
                error_result["troubleshooting_tips"].insert(0, "Timeout detected - try a more specific query")
            elif "rate limit" in str(e).lower():
                error_result["troubleshooting_tips"].insert(0, "Rate limit reached - wait a moment and try again")
            
            return error_result
    
    def daily_research_scan(self, custom_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform daily automated research scan.
        
        Args:
            custom_queries: Optional custom queries to include
        
        Returns:
            Dictionary with scan results
        """
        logger.info("Starting daily research scan")
        
        # Default scan queries
        default_queries = [
            "novel methane inhibitors cattle livestock 2024",
            "enzyme inhibition ruminant fermentation recent advances",
            "feed additives reduce greenhouse gas emissions",
            "methanogenesis archaea inhibition breakthrough"
        ]
        
        # Add cross-domain queries
        cross_domain_queries = [
            f"methane reduction {keyword}" for keyword in CROSS_DOMAIN_KEYWORDS[:3]
        ]
        
        # Combine all queries
        all_queries = default_queries + cross_domain_queries
        if custom_queries:
            all_queries.extend(custom_queries)
        
        scan_results = {
            "scan_date": datetime.now().isoformat(),
            "queries_processed": len(all_queries),
            "results": [],
            "summary": "",
            "novel_discoveries": [],
            "generated_hypotheses": []
        }
        
        for query in all_queries:
            try:
                logger.info(f"Processing scan query: {query}")
                
                # Conduct focused research
                query_result = self.conduct_research(
                    research_query=query,
                    focus_areas=["novel mechanisms", "practical applications"]
                )
                
                scan_results["results"].append(query_result)
                
                # Extract novel discoveries
                if query_result.get("novel_insights"):
                    scan_results["novel_discoveries"].extend(query_result["novel_insights"])
                
                # Extract hypotheses
                if query_result.get("hypotheses"):
                    scan_results["generated_hypotheses"].extend(query_result["hypotheses"])
                
            except Exception as e:
                logger.error(f"Error processing query {query}: {e}")
                continue
        
        # Generate scan summary
        scan_results["summary"] = self._generate_scan_summary(scan_results)
        
        logger.info("Daily research scan completed")
        return scan_results
    
    def analyze_breakthrough_potential(self, research_findings: str, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Analyze the breakthrough potential of research findings.
        
        Args:
            research_findings: Description of research findings
            callbacks: Optional list of callback handlers for streaming
        
        Returns:
            Dictionary with breakthrough analysis
        """
        analysis_query = f"""
        Analyze the breakthrough potential of these research findings for Agteria's methane reduction goals:
        
        {research_findings}
        
        Consider:
        1. Technical feasibility and scalability
        2. Commercial viability and cost-effectiveness
        3. Competitive advantages
        4. Timeline to implementation
        5. Risk factors and mitigation strategies
        6. Market impact potential
        
        Provide a comprehensive breakthrough assessment.
        """
        
        return self.conduct_research(analysis_query, focus_areas=["breakthrough analysis"], callbacks=callbacks)
    
    def explore_research_gaps(self, research_area: str, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Explore research gaps in a specific area.
        
        Args:
            research_area: Area of research to explore
            callbacks: Optional list of callback handlers for streaming
        
        Returns:
            Dictionary with gap analysis and opportunities
        """
        gap_query = f"""
        Identify research gaps and unexplored opportunities in: {research_area}
        
        Focus on:
        1. What current approaches are missing
        2. Underexplored biological mechanisms
        3. Novel application possibilities
        4. Cross-disciplinary opportunities
        5. Technology transfer possibilities
        
        Generate specific research hypotheses to fill these gaps.
        """
        
        return self.conduct_research(gap_query, focus_areas=["gap analysis", "opportunity identification"], callbacks=callbacks)
    
    def track_competitor_research(self, competitors: List[str]) -> Dict[str, Any]:
        """
        Track competitor research and developments.
        
        Args:
            competitors: List of competitor names or research groups
        
        Returns:
            Dictionary with competitive intelligence
        """
        competitor_queries = []
        
        for competitor in competitors:
            competitor_queries.extend([
                f"{competitor} methane reduction research recent",
                f"{competitor} livestock emissions patent",
                f"{competitor} feed additive development"
            ])
        
        intelligence_results = {
            "analysis_date": datetime.now().isoformat(),
            "competitors_analyzed": competitors,
            "findings": [],
            "competitive_threats": [],
            "collaboration_opportunities": []
        }
        
        for query in competitor_queries:
            try:
                result = self.conduct_research(
                    research_query=query,
                    focus_areas=["competitive analysis"]
                )
                intelligence_results["findings"].append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing competitor query {query}: {e}")
        
        return intelligence_results
    
    def _enhance_query_with_context(self, query: str, focus_areas: Optional[List[str]] = None) -> str:
        """Enhance research query with Agteria-specific context."""
        enhanced = f"""
        Research Query: {query}
        
        Context: You are researching for Agteria Biotech, which focuses on reducing methane emissions from cattle through novel feed additives and enzyme inhibitors. Their goal is to reduce 1% of global greenhouse gas emissions by 2035.
        
        Current Research Focus: {', '.join(AGTERIA_KEYWORDS)}
        
        Focus Areas: {', '.join(focus_areas) if focus_areas else 'General research'}
        
        Instructions:
        1. Search multiple databases for comprehensive coverage
        2. Analyze papers for practical applications to cattle methane reduction
        3. Look for cross-domain innovations from other fields
        4. Generate novel, testable hypotheses
        5. Identify potential collaborations or partnerships
        6. Assess commercial viability and scalability
        
        Provide detailed insights and actionable recommendations.
        """
        
        return enhanced
    
    def _structure_research_result(self, query: str, agent_response: str, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Structure the agent's research response with source attribution."""
        # Extract paper IDs from agent response
        paper_ids_from_response = self._extract_paper_ids(agent_response)
        
        # Get all paper IDs from citation tracker (more reliable)
        from ..tools.search_tools import get_global_citation_tracker
        citation_tracker = get_global_citation_tracker()
        all_stored_sources = citation_tracker.get_all_sources()
        all_paper_ids = [source.id for source in all_stored_sources]
        
        # Use stored IDs if agent response doesn't have any
        paper_ids = paper_ids_from_response if paper_ids_from_response else all_paper_ids
        sources = self._get_sources_for_ids(paper_ids)
        
        # Extract insights
        
        return {
            "query": query,
            "focus_areas": focus_areas or [],
            "response": agent_response,
            "timestamp": datetime.now().isoformat(),
            "session_id": id(self),
            "papers_found": len(sources),
            "novel_insights": self._extract_novel_insights(agent_response),
            "hypotheses": self._extract_hypotheses(agent_response),
            "next_steps": self._extract_next_steps(agent_response),
            # Source attribution fields
            "sources": sources,
            "paper_ids_referenced": paper_ids,
            "bibliography": self._generate_bibliography(sources)
        }
    
    def _extract_paper_count(self, response: str) -> int:
        """Extract number of papers found from response."""
        import re
        match = re.search(r"found (\d+) papers", response.lower())
        return int(match.group(1)) if match else 0
    
    def _extract_novel_insights(self, response: str) -> List[str]:
        """Extract novel insights from response."""
        insights = []
        
        # Look for insight keywords
        insight_keywords = ["novel", "breakthrough", "innovative", "unprecedented", "unique"]
        
        sentences = response.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in insight_keywords):
                insights.append(sentence.strip())
        
        return insights[:5]  # Limit to top 5
    
    def _extract_hypotheses(self, response: str) -> List[str]:
        """Extract hypotheses from response."""
        hypotheses = []
        
        # Look for hypothesis indicators
        hypothesis_patterns = [
            r"hypothesis:?\s*(.+?)(?=\n|$)",
            r"propose:?\s*(.+?)(?=\n|$)",
            r"suggest:?\s*(.+?)(?=\n|$)"
        ]
        
        import re
        for pattern in hypothesis_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            hypotheses.extend([m.strip() for m in matches])
        
        return hypotheses[:3]  # Limit to top 3
    
    def _extract_next_steps(self, response: str) -> List[str]:
        """Extract next steps from response."""
        next_steps = []
        
        # Look for action items
        action_keywords = ["next step", "recommend", "should", "could", "future work"]
        
        sentences = response.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in action_keywords):
                next_steps.append(sentence.strip())
        
        return next_steps[:3]  # Limit to top 3
    
    def _update_knowledge_from_research(self, research_result: Dict[str, Any]):
        """Update knowledge graph from research results."""
        try:
            # This would extract concepts and update the knowledge graph
            # Implementation depends on the specific format of research results
            pass
        except Exception as e:
            logger.warning(f"Could not update knowledge graph: {e}")
    
    def _generate_scan_summary(self, scan_results: Dict[str, Any]) -> str:
        """Generate a summary of the daily scan results."""
        total_queries = scan_results["queries_processed"]
        novel_count = len(scan_results["novel_discoveries"])
        hypothesis_count = len(scan_results["generated_hypotheses"])
        
        summary = f"""
        Daily Research Scan Summary ({scan_results['scan_date']})
        
        📊 Scan Statistics:
        • Queries Processed: {total_queries}
        • Novel Discoveries: {novel_count}
        • Hypotheses Generated: {hypothesis_count}
        
        🔬 Key Discoveries:
        """
        
        # Add top discoveries
        for i, discovery in enumerate(scan_results["novel_discoveries"][:3], 1):
            summary += f"\n{i}. {discovery}"
        
        summary += "\n\n💡 Generated Hypotheses:"
        
        # Add top hypotheses
        for i, hypothesis in enumerate(scan_results["generated_hypotheses"][:3], 1):
            summary += f"\n{i}. {hypothesis}"
        
        return summary
    
    def _extract_paper_ids(self, response: str) -> List[str]:
        """Extract paper IDs mentioned in the agent response."""
        import re
        
        paper_ids = []
        
        # Pattern 1: [ID: paper_id] format (simplified)
        id_pattern = r'\[ID:\s*([^\]]+?)\]'
        matches = re.findall(id_pattern, response, re.IGNORECASE)
        for match in matches:
            clean_id = match.split(':')[0].strip()
            if clean_id and clean_id not in paper_ids:
                paper_ids.append(clean_id)
        
        # Pattern 2: [Paper ID: title] format (original)
        paper_id_pattern = r'\[(?:Paper )?ID:\s*([^\]]+?)\]'
        matches = re.findall(paper_id_pattern, response, re.IGNORECASE)
        for match in matches:
            clean_id = match.split(':')[0].strip()
            if clean_id and clean_id not in paper_ids:
                paper_ids.append(clean_id)
        
        # Pattern 3: PAPER_IDS: lists from tool responses
        list_pattern = r'(?:PAPER_IDS|ALL_PAPER_IDS|WEB_PAPER_IDS):\s*([^\n]+)'
        list_matches = re.findall(list_pattern, response, re.IGNORECASE)
        for match in list_matches:
            ids = [id.strip() for id in match.split(',') if id.strip()]
            for id in ids:
                if id not in paper_ids:
                    paper_ids.append(id)
        
        # Pattern 4: Extract arxiv, pubmed, doi, web IDs directly
        direct_patterns = [
            r'(arxiv_[a-zA-Z0-9_]+)',
            r'(pubmed_[a-zA-Z0-9_]+)', 
            r'(doi_[a-zA-Z0-9_]+)',
            r'(web_[a-zA-Z0-9_]+)'
        ]
        for pattern in direct_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if match not in paper_ids:
                    paper_ids.append(match)
        
        logger.info(f"Extracted {len(paper_ids)} paper IDs from response")
        return paper_ids
    
    def _get_sources_for_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Get source information for the given paper IDs."""
        from ..tools.search_tools import get_global_citation_tracker
        
        citation_tracker = get_global_citation_tracker()
        sources = []
        
        for paper_id in paper_ids:
            source = citation_tracker.get_source(paper_id)
            if source:
                sources.append(source.to_dict())
            else:
                logger.warning(f"Source not found for paper ID: {paper_id}")
        
        logger.info(f"Retrieved {len(sources)} sources for {len(paper_ids)} paper IDs")
        return sources
    
    def _generate_bibliography(self, sources: List[Dict[str, Any]]) -> str:
        """Generate formatted bibliography from sources."""
        if not sources:
            return "No sources available for bibliography."
        
        bibliography = "## Bibliography\n\n"
        
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown Title')
            authors = source.get('authors', [])
            journal = source.get('journal', '')
            published = source.get('published', '')
            database = source.get('database', 'unknown')
            
            # Format authors
            if authors:
                if len(authors) == 1:
                    author_str = authors[0]
                elif len(authors) <= 3:
                    author_str = ', '.join(authors[:-1]) + f', & {authors[-1]}'
                else:
                    author_str = f"{authors[0]} et al."
            else:
                author_str = "Unknown Authors"
            
            # Extract year
            year = "n.d."
            if published:
                import re
                year_match = re.search(r'(\d{4})', str(published))
                if year_match:
                    year = year_match.group(1)
            
            # Create citation
            citation = f"{author_str} ({year}). {title}."
            
            if journal:
                citation += f" {journal}."
            
            # Add database info
            citation += f" Retrieved from {database.title()}."
            
            # Add DOI or URL if available
            if source.get('doi'):
                citation += f" https://doi.org/{source['doi']}"
            elif source.get('url'):
                citation += f" {source['url']}"
            
            bibliography += f"{i}. {citation}\n\n"
        
        return bibliography

    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic agent functionality with a simple query."""
        try:
            logger.info("Testing basic agent functionality...")
            
            # Simple test query
            test_query = "What is methane?"
            
            # Use a simplified prompt for testing
            simple_prompt = f"""You are a research assistant. Answer this question briefly: {test_query}
            
Use this format:
Thought: I need to answer this question about methane
Action: (no tools needed for basic definition)
Final Answer: [Your brief answer about methane]"""
            
            # Create a simple agent for testing
            from langchain.agents import create_react_agent
            from langchain_core.prompts import PromptTemplate
            
            test_prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template=f"""Answer this question: {{input}}

Available tools: {{tools}}

{{agent_scratchpad}}"""
            )
            
            test_agent = create_react_agent(
                llm=self.llm,
                tools=[],  # No tools for basic test
                prompt=test_prompt
            )
            
            from langchain.agents import AgentExecutor
            test_executor = AgentExecutor(
                agent=test_agent,
                tools=[],
                verbose=True,
                max_iterations=3,
                max_execution_time=30,
                handle_parsing_errors=True
            )
            
            result = test_executor.invoke({"input": test_query})
            
            return {
                "test_status": "success",
                "test_query": test_query,
                "test_response": result.get("output", ""),
                "message": "Basic agent functionality is working"
            }
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return {
                "test_status": "failed", 
                "error": str(e),
                "message": "Basic agent functionality has issues"
            }
    
    def conduct_simple_research(self, research_query: str) -> Dict[str, Any]:
        """Conduct research with simplified prompt (fallback mode)."""
        logger.info(f"Using simplified research mode for: {research_query}")
        
        # Simplified prompt without complex source attribution
        simple_query = f"""Research this topic and provide a brief summary: {research_query}

Focus on:
1. Key findings from recent papers
2. Main research trends  
3. Practical applications

Use available search tools and provide a concise summary."""
        
        try:
            # Create simplified agent executor with lower complexity
            from langchain.agents import create_react_agent, AgentExecutor
            from langchain_core.prompts import PromptTemplate
            
            simple_prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template="""You are a research assistant. Use the available tools to research the topic and provide a summary.

Available tools:
{tools}

Tool names: {tool_names}

Research this: {input}

Use this format:
Thought: [your reasoning]
Action: [tool to use]
Action Input: [input to tool]
Observation: [tool response]
... (repeat as needed)
Thought: I have enough information to provide a summary
Final Answer: [your research summary]

{agent_scratchpad}"""
            )
            
            simple_agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=simple_prompt
            )
            
            simple_executor = AgentExecutor(
                agent=simple_agent,
                tools=self.tools,
                verbose=self.verbose,
                max_iterations=10,
                max_execution_time=300,
                handle_parsing_errors=True
            )
            
            result = simple_executor.invoke({"input": simple_query})
            
            return {
                "query": research_query,
                "response": result.get("output", ""),
                "mode": "simplified",
                "timestamp": datetime.now().isoformat(),
                "papers_found": 0,  # Basic structure for compatibility
                "novel_insights": [],
                "hypotheses": [],
                "next_steps": []
            }
            
        except Exception as e:
            logger.error(f"Simplified research also failed: {e}")
            return {
                "query": research_query,
                "error": f"Both normal and simplified research failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "troubleshooting_tips": [
                    "Check OpenAI API key configuration",
                    "Verify internet connectivity",
                    "Try restarting the application",
                    "Check system logs for detailed errors"
                ]
            }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of the agent."""
        memory_stats = self.memory.get_memory_stats()
        
        return {
            "agent_initialized": True,
            "current_focus": self.session_state["current_research_focus"],
            "papers_in_memory": memory_stats.get("total_documents", 0),
            "recent_discoveries": len(self.session_state["recent_discoveries"]),
            "session_stats": self.session_state.copy(),
            "available_tools": len(self.tools),
            "memory_health": "good" if memory_stats.get("total_documents", 0) > 0 else "empty"
        }