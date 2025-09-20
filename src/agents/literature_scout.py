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

RESEARCH APPROACH:
1. SEARCH STRATEGY: Use multiple databases (ArXiv, PubMed, Web) to ensure comprehensive coverage
2. ANALYSIS DEPTH: Thoroughly analyze papers for novel mechanisms, molecules, and methodologies
3. CROSS-DOMAIN THINKING: Actively look for connections between different research fields
4. HYPOTHESIS GENERATION: Generate creative but scientifically grounded research hypotheses
5. MEMORY UTILIZATION: Check memory to avoid duplicate work and build on previous discoveries

RESPONSE FORMAT:
Use this format for your reasoning:

Thought: [Your reasoning about what to do next]
Action: [The tool to use]
Action Input: [The input to the tool]
Observation: [The tool's response]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to provide a comprehensive response
Final Answer: [Your detailed analysis and recommendations]

KEY PRIORITIES FOR AGTERIA:
- Novel methane inhibition mechanisms
- Scalable and cost-effective solutions
- Safety and efficacy in livestock
- Cross-industry applicable technologies
- Breakthrough molecular discoveries

Current Request: {input}

Begin your research:

{agent_scratchpad}"""
        )
        
        # Create React agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.agent_memory,
            verbose=self.verbose,
            max_iterations=15,
            max_execution_time=300,  # 5 minutes timeout
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def conduct_research(self, research_query: str, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a query.
        
        Args:
            research_query: The research question or topic
            focus_areas: Optional specific areas to focus on
        
        Returns:
            Dictionary with research results and insights
        """
        logger.info(f"Starting research on: {research_query}")
        
        # Update session state
        self.session_state["current_research_focus"] = research_query
        
        # Enhance query with Agteria context
        enhanced_query = self._enhance_query_with_context(research_query, focus_areas)
        
        try:
            # Run the agent
            result = self.agent.invoke({"input": enhanced_query})
            
            # Process and structure the results
            structured_result = self._structure_research_result(
                query=research_query,
                agent_response=result.get("output", ""),
                focus_areas=focus_areas
            )
            
            # Update knowledge graph
            self._update_knowledge_from_research(structured_result)
            
            logger.info(f"Research completed for: {research_query}")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            return {
                "query": research_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
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
    
    def analyze_breakthrough_potential(self, research_findings: str) -> Dict[str, Any]:
        """
        Analyze the breakthrough potential of research findings.
        
        Args:
            research_findings: Description of research findings
        
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
        
        return self.conduct_research(analysis_query, focus_areas=["breakthrough analysis"])
    
    def explore_research_gaps(self, research_area: str) -> Dict[str, Any]:
        """
        Explore research gaps in a specific area.
        
        Args:
            research_area: Area of research to explore
        
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
        
        return self.conduct_research(gap_query, focus_areas=["gap analysis", "opportunity identification"])
    
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
        """Structure the agent's research response."""
        return {
            "query": query,
            "focus_areas": focus_areas or [],
            "response": agent_response,
            "timestamp": datetime.now().isoformat(),
            "session_id": id(self),
            "papers_found": self._extract_paper_count(agent_response),
            "novel_insights": self._extract_novel_insights(agent_response),
            "hypotheses": self._extract_hypotheses(agent_response),
            "next_steps": self._extract_next_steps(agent_response)
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