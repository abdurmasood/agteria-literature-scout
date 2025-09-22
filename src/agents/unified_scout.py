"""Unified Literature Scout agent with pluggable capabilities."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from .core import AgentCore
from .capabilities.research import ResearchCapability
from .capabilities.business import BusinessCapability

logger = logging.getLogger(__name__)

class UnifiedLiteratureScout:
    """
    Unified Literature Scout with pluggable capability modules.
    
    This agent provides a single interface for all research and analysis tasks,
    automatically routing queries to the appropriate specialized capability.
    
    Capabilities:
    - Research: Literature discovery, analysis, hypothesis generation
    - Business: Breakthrough analysis, commercial viability assessment
    - Analytics: (Future) Trend analysis, competitive intelligence
    """
    
    def __init__(self, memory_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the unified agent with all capabilities.
        
        Args:
            memory_path: Optional custom path for memory storage
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize shared core services
        self.core = AgentCore(memory_path=memory_path, verbose=verbose)
        
        # Load all available capabilities
        self.capabilities = self._load_capabilities()
        
        # Track initialization
        self.initialized_at = datetime.now().isoformat()
        
        logger.info(f"UnifiedLiteratureScout initialized with {len(self.capabilities)} capabilities")
    
    def _load_capabilities(self) -> Dict[str, Any]:
        """Load all available capability modules."""
        capabilities = {}
        
        try:
            # Load research capability
            capabilities['research'] = ResearchCapability(self.core)
            logger.info("Research capability loaded")
        except Exception as e:
            logger.error(f"Failed to load research capability: {e}")
        
        try:
            # Load business capability
            capabilities['business'] = BusinessCapability(self.core)
            logger.info("Business capability loaded")
        except Exception as e:
            logger.error(f"Failed to load business capability: {e}")
        
        # Future capabilities can be added here
        # capabilities['analytics'] = AnalyticsCapability(self.core)
        
        return capabilities
    
    def execute(self, query: str, capability: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the appropriate capability.
        
        Args:
            query: The task query or research question
            capability: Optional explicit capability selection
            **kwargs: Additional parameters passed to the capability
        
        Returns:
            Result dictionary from the selected capability
        """
        try:
            # Detect capability if not explicitly provided
            capability_name = capability or self._detect_capability(query)
            
            # Validate capability exists
            if capability_name not in self.capabilities:
                available = list(self.capabilities.keys())
                raise ValueError(f"Capability '{capability_name}' not available. Available: {available}")
            
            # Get the capability module
            capability_module = self.capabilities[capability_name]
            
            # Prepare task
            task = {
                'query': query,
                'context': self.core.get_context(),
                **kwargs
            }
            
            logger.info(f"Routing task to {capability_name} capability")
            
            # Execute task
            result = capability_module.execute(task)
            
            # Add routing information to result
            result['routing_info'] = {
                'capability_used': capability_name,
                'explicit_routing': capability is not None,
                'available_capabilities': list(self.capabilities.keys())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Unified execution error: {e}")
            return self._handle_unified_error(query, capability, e)
    
    def _detect_capability(self, query: str) -> str:
        """
        Detect the appropriate capability for a given query.
        
        Args:
            query: Query string to analyze
        
        Returns:
            Capability name to use
        """
        query_lower = query.lower()
        
        # Business analysis indicators
        business_keywords = [
            'breakthrough', 'commercial', 'viability', 'market', 'business',
            'feasibility', 'competitive', 'investment', 'analyze', 'potential',
            'revenue', 'cost', 'regulatory', 'approval', 'timeline', 'risk'
        ]
        
        # Research indicators
        research_keywords = [
            'research', 'literature', 'papers', 'studies', 'find', 'search',
            'discover', 'hypothesis', 'mechanism', 'molecular', 'enzyme',
            'novel', 'innovation', 'scientific', 'publication'
        ]
        
        # Count keyword matches
        business_score = sum(1 for kw in business_keywords if kw in query_lower)
        research_score = sum(1 for kw in research_keywords if kw in query_lower)
        
        # Special patterns for business analysis
        if any(phrase in query_lower for phrase in [
            'breakthrough potential', 'commercial potential', 'market analysis',
            'feasibility study', 'investment recommendation', 'business case'
        ]):
            business_score += 3
        
        # Special patterns for research
        if any(phrase in query_lower for phrase in [
            'literature review', 'research papers', 'find studies',
            'scientific discovery', 'novel research', 'hypothesis generation'
        ]):
            research_score += 3
        
        # Default to research for ambiguous queries
        if business_score > research_score:
            return 'business'
        else:
            return 'research'
    
    # Convenience methods for backward compatibility
    
    def conduct_research(self, research_query: str, focus_areas: Optional[List[str]] = None, 
                        callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Conduct research using the research capability.
        
        Args:
            research_query: Research question or topic
            focus_areas: Optional specific areas to focus on
            callbacks: Optional callback handlers for streaming
        
        Returns:
            Research results
        """
        return self.execute(
            query=research_query,
            capability='research',
            focus_areas=focus_areas,
            callbacks=callbacks
        )
    
    def analyze_breakthrough_potential(self, research_findings: str, 
                                     callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Analyze breakthrough potential using the business capability.
        
        Args:
            research_findings: Research findings to analyze
            callbacks: Optional callback handlers for streaming
        
        Returns:
            Business analysis results
        """
        return self.execute(
            query=research_findings,
            capability='business',
            callbacks=callbacks
        )
    
    def daily_research_scan(self, custom_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform daily research scan using the research capability.
        
        Args:
            custom_queries: Optional custom queries to include in scan
        
        Returns:
            Daily scan results
        """
        from ..config import AGTERIA_KEYWORDS
        
        # Use default queries if none provided
        queries = custom_queries or [
            f"latest research on {keyword}"
            for keyword in AGTERIA_KEYWORDS[:3]  # Use top 3 keywords
        ]
        
        scan_results = {
            "scan_date": datetime.now().isoformat(),
            "queries_processed": len(queries),
            "results": [],
            "summary": "",
            "novel_discoveries": [],
            "generated_hypotheses": []
        }
        
        # Execute research for each query
        for query in queries:
            try:
                result = self.conduct_research(query)
                scan_results["results"].append(result)
                
                # Aggregate novel discoveries and hypotheses
                if result.get("novel_insights"):
                    scan_results["novel_discoveries"].extend(result["novel_insights"])
                if result.get("hypotheses"):
                    scan_results["generated_hypotheses"].extend(result["hypotheses"])
                    
            except Exception as e:
                logger.error(f"Error in daily scan query '{query}': {e}")
                scan_results["results"].append({
                    "query": query,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Generate summary
        successful_results = [r for r in scan_results["results"] if r.get("status") != "failed"]
        scan_results["summary"] = f"Processed {len(queries)} queries with {len(successful_results)} successful results. Found {len(scan_results['novel_discoveries'])} novel discoveries and generated {len(scan_results['generated_hypotheses'])} hypotheses."
        
        return scan_results
    
    def explore_research_gaps(self, research_area: str, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Explore research gaps using the research capability.
        
        Args:
            research_area: Research area to explore for gaps
            callbacks: Optional callback handlers for streaming
        
        Returns:
            Gap analysis results
        """
        gap_query = f"Identify research gaps and unexplored opportunities in {research_area}. Focus on areas where current solutions are inadequate or where novel approaches could provide breakthrough innovations."
        
        return self.execute(
            query=gap_query,
            capability='research',
            focus_areas=["research gaps", "unexplored opportunities"],
            callbacks=callbacks
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the unified agent.
        
        Returns:
            Status dictionary with all information
        """
        # Get core status
        core_status = self.core.get_agent_status()
        
        # Get capability status
        capability_status = {}
        for name, capability in self.capabilities.items():
            try:
                capability_status[name] = {
                    'info': capability.get_capability_info(),
                    'usage_stats': capability.get_usage_stats()
                }
            except Exception as e:
                capability_status[name] = {'error': str(e)}
        
        return {
            'unified_agent': {
                'initialized_at': self.initialized_at,
                'total_capabilities': len(self.capabilities),
                'available_capabilities': list(self.capabilities.keys())
            },
            'core_services': core_status,
            'capabilities': capability_status
        }
    
    def get_capability_info(self, capability_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about capabilities.
        
        Args:
            capability_name: Optional specific capability to get info for
        
        Returns:
            Capability information
        """
        if capability_name:
            if capability_name in self.capabilities:
                return self.capabilities[capability_name].get_capability_info()
            else:
                raise ValueError(f"Capability '{capability_name}' not found")
        else:
            # Return info for all capabilities
            return {
                name: capability.get_capability_info()
                for name, capability in self.capabilities.items()
            }
    
    def _handle_unified_error(self, query: str, capability: Optional[str], error: Exception) -> Dict[str, Any]:
        """Handle errors at the unified agent level."""
        error_msg = str(error)
        logger.error(f"Unified agent error: {error_msg}")
        
        return {
            "status": "failed",
            "error": error_msg,
            "response": f"Unified agent execution failed: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "requested_capability": capability,
            "available_capabilities": list(self.capabilities.keys()),
            "troubleshooting": [
                "Check that all required dependencies are installed",
                "Verify API keys are properly configured",
                "Try specifying a capability explicitly if auto-detection failed",
                "Check the agent status for more detailed error information",
                "Contact support if the issue persists"
            ]
        }
    
    def export_session_data(self) -> str:
        """
        Export current session data as JSON.
        
        Returns:
            JSON string with session data
        """
        try:
            session_data = {
                "agent_type": "UnifiedLiteratureScout",
                "initialized_at": self.initialized_at,
                "export_timestamp": datetime.now().isoformat(),
                "capabilities": list(self.capabilities.keys()),
                "session_state": self.core.session_state,
                "memory_stats": self.core.memory.get_memory_stats(),
                "citation_data": self.core.citation_tracker.export_sources_json()
            }
            
            return json.dumps(session_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting session data: {e}")
            return f'{{"error": "Failed to export session data: {str(e)}"}}'
    
    def cleanup_session(self, days: int = 7):
        """
        Clean up old session data.
        
        Args:
            days: Remove data older than this many days
        """
        try:
            # Clean up core session state
            self.core.cleanup_old_state(days)
            
            # Clean up old documents from memory if needed
            # self.core.memory.clear_old_documents(days * 3)  # Keep documents longer
            
            logger.info(f"Cleaned up session data older than {days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    # Memory property for backward compatibility
    @property
    def memory(self):
        """Access to the research memory for backward compatibility."""
        return self.core.memory
    
    # Additional methods for backward compatibility with LiteratureScout
    
    def track_competitor_research(self, competitors: List[str]) -> Dict[str, Any]:
        """
        Track competitor research and developments.
        
        Args:
            competitors: List of competitor names or research groups
        
        Returns:
            Dictionary with competitive intelligence
        """
        from ..config import AGTERIA_KEYWORDS
        
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
    
    def conduct_simple_research(self, research_query: str) -> Dict[str, Any]:
        """Conduct research with simplified prompt (fallback mode)."""
        logger.info(f"Using simplified research mode for: {research_query}")
        
        try:
            # Use the research capability but with simplified query
            simple_query = f"""Research this topic and provide a brief summary: {research_query}
Focus on:
1. Key findings from recent papers
2. Main research trends  
3. Practical applications
Use available search tools and provide a concise summary."""
            
            result = self.execute(
                query=simple_query,
                capability='research'
            )
            
            # Adjust result format for compatibility
            result.update({
                "query": research_query,
                "mode": "simplified",
                "papers_found": result.get("papers_found", 0),
                "novel_insights": result.get("novel_insights", []),
                "hypotheses": result.get("hypotheses", []),
                "next_steps": result.get("next_steps", [])
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Simplified research failed: {e}")
            return {
                "query": research_query,
                "error": f"Simplified research failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "troubleshooting_tips": [
                    "Check OpenAI API key configuration",
                    "Verify internet connectivity",
                    "Try restarting the application",
                    "Check system logs for detailed errors"
                ]
            }
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic agent functionality with a simple query."""
        try:
            logger.info("Testing basic unified agent functionality...")
            
            # Simple test query
            test_query = "What is methane?"
            
            # Test using research capability with a very simple query
            result = self.execute(
                query=test_query,
                capability='research'
            )
            
            if result.get('status') == 'completed':
                return {
                    "test_status": "success",
                    "test_query": test_query,
                    "test_response": result.get("response", ""),
                    "message": "Unified agent basic functionality test passed",
                    "capabilities_tested": list(self.capabilities.keys())
                }
            else:
                return {
                    "test_status": "failed",
                    "test_query": test_query,
                    "error": result.get("error", "Unknown error"),
                    "message": "Unified agent basic functionality test failed"
                }
                
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return {
                "test_status": "failed",
                "test_query": "What is methane?",
                "error": str(e),
                "message": "Basic functionality test encountered an exception"
            }