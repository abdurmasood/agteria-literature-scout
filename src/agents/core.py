"""Core infrastructure for unified agent architecture."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.tools import Tool

from ..config import Config
from ..memory.research_memory import ResearchMemory, KnowledgeGraph
from ..processors.document_processor import DocumentProcessor
from ..utils.citation_tracker import CitationTracker

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry for managing tools across capabilities."""
    
    def __init__(self):
        self._tools = {}
        self._registered_sets = set()
    
    def register_tool_set(self, name: str, tools: List[Tool]):
        """Register a set of tools by name."""
        if name not in self._registered_sets:
            self._tools[name] = tools
            self._registered_sets.add(name)
            logger.info(f"Registered {len(tools)} tools for {name}")
    
    def get_tools(self, tool_sets: List[str]) -> List[Tool]:
        """Get tools from specified tool sets."""
        all_tools = []
        for tool_set in tool_sets:
            if tool_set in self._tools:
                all_tools.extend(self._tools[tool_set])
        return all_tools
    
    def get_all_tools(self) -> Dict[str, List[Tool]]:
        """Get all registered tools."""
        return self._tools.copy()

class AgentCore:
    """
    Shared core services for all agent capabilities.
    
    This class provides common infrastructure that all capabilities can use:
    - Memory management
    - LLM instances
    - Citation tracking
    - Document processing
    - Tool registry
    - Session state
    """
    
    def __init__(self, memory_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize core agent services.
        
        Args:
            memory_path: Optional custom path for memory storage
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize memory system
        self.memory = ResearchMemory(persist_directory=memory_path)
        self.knowledge_graph = KnowledgeGraph(self.memory)
        
        # Initialize processing services
        self.document_processor = DocumentProcessor()
        self.citation_tracker = CitationTracker()
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Initialize session state
        self.session_state = {
            "current_research_focus": None,
            "recent_discoveries": [],
            "hypothesis_count": 0,
            "papers_analyzed": 0,
            "active_capabilities": [],
            "created_at": datetime.now().isoformat()
        }
        
        logger.info("AgentCore initialized successfully")
    
    def get_llm(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs) -> ChatOpenAI:
        """
        Get configured LLM instance with optional overrides.
        
        Args:
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional LLM parameters
            
        Returns:
            Configured ChatOpenAI instance
        """
        llm_config = {
            "model": Config.DEFAULT_MODEL,
            "temperature": temperature or Config.TEMPERATURE,
            "openai_api_key": Config.OPENAI_API_KEY,
            **kwargs
        }
        
        if max_tokens:
            llm_config["max_tokens"] = max_tokens
            
        return ChatOpenAI(**llm_config)
    
    def get_memory(self) -> ConversationSummaryBufferMemory:
        """
        Get conversation memory for agent interactions.
        
        Returns:
            Configured conversation memory
        """
        return ConversationSummaryBufferMemory(
            llm=self.get_llm(temperature=0.3),  # Lower temp for summaries
            max_token_limit=2000,
            return_messages=True
        )
    
    def create_memory_tools(self) -> List[Tool]:
        """
        Create tools for interacting with research memory.
        
        Returns:
            List of memory-related tools
        """
        def search_memory_func(query: str) -> str:
            """Search stored papers in memory."""
            results = self.memory.search_papers(query, k=5)
            if not results:
                return f"No papers found in memory for: {query}"
            
            # Format results with source attribution
            formatted_results = []
            for doc in results:
                title = doc.metadata.get('title', 'Unknown')
                source_id = doc.metadata.get('source_id', 'Unknown')
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                formatted_results.append(f"[ID: {source_id}] {title}\nContent: {content_preview}")
            
            return f"Found {len(results)} papers in memory:\n" + "\n\n".join(formatted_results)
        
        def get_memory_stats_func() -> str:
            """Get memory statistics."""
            stats = self.memory.get_memory_stats()
            return f"""Memory Statistics:
- Total documents: {stats.get('total_documents', 0)}
- Total analyses: {stats.get('total_analyses', 0)}
- Total searches: {stats.get('total_searches', 0)}
- Average quality score: {stats.get('average_quality_score', 0):.2f}
- Documents by source: {stats.get('documents_by_source', {})}
"""
        
        memory_tools = [
            Tool(
                name="search_memory",
                description="Search through previously analyzed papers in memory. Use this to avoid duplicate work and reference previous findings.",
                func=search_memory_func
            ),
            Tool(
                name="memory_stats",
                description="Get statistics about papers and analyses stored in memory.",
                func=get_memory_stats_func
            )
        ]
        
        # Register memory tools
        self.tool_registry.register_tool_set("memory", memory_tools)
        return memory_tools
    
    def update_session_state(self, updates: Dict[str, Any]):
        """
        Update session state with new information.
        
        Args:
            updates: Dictionary of state updates
        """
        self.session_state.update(updates)
        self.session_state["last_updated"] = datetime.now().isoformat()
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get current context for capability execution.
        
        Returns:
            Context dictionary with session state and statistics
        """
        memory_stats = self.memory.get_memory_stats()
        
        return {
            "session_state": self.session_state.copy(),
            "memory_stats": memory_stats,
            "available_tools": list(self.tool_registry.get_all_tools().keys()),
            "context_generated_at": datetime.now().isoformat()
        }
    
    def log_capability_usage(self, capability_name: str, task: str, result_summary: str):
        """
        Log usage of a capability for analytics.
        
        Args:
            capability_name: Name of the capability used
            task: Description of the task performed
            result_summary: Brief summary of the result
        """
        usage_entry = {
            "capability": capability_name,
            "task": task,
            "result": result_summary,
            "timestamp": datetime.now().isoformat()
        }
        
        # Track in session state
        if "capability_usage" not in self.session_state:
            self.session_state["capability_usage"] = []
        
        self.session_state["capability_usage"].append(usage_entry)
        
        # Keep only last 50 entries to prevent memory bloat
        if len(self.session_state["capability_usage"]) > 50:
            self.session_state["capability_usage"] = self.session_state["capability_usage"][-50:]
        
        logger.info(f"Logged usage: {capability_name} - {task}")
    
    def get_shared_citations(self) -> CitationTracker:
        """
        Get shared citation tracker for source attribution.
        
        Returns:
            Shared CitationTracker instance
        """
        return self.citation_tracker
    
    def cleanup_old_state(self, days: int = 7):
        """
        Clean up old session state to prevent memory bloat.
        
        Args:
            days: Remove entries older than this many days
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        # Clean up capability usage
        if "capability_usage" in self.session_state:
            cleaned_usage = []
            for entry in self.session_state["capability_usage"]:
                try:
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time > cutoff:
                        cleaned_usage.append(entry)
                except (KeyError, ValueError):
                    # Keep entries without valid timestamps
                    cleaned_usage.append(entry)
            
            self.session_state["capability_usage"] = cleaned_usage
        
        logger.info(f"Cleaned up session state older than {days} days")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the agent core.
        
        Returns:
            Status dictionary with all core service information
        """
        memory_stats = self.memory.get_memory_stats()
        tool_stats = {name: len(tools) for name, tools in self.tool_registry.get_all_tools().items()}
        
        return {
            "core_initialized": True,
            "memory_status": {
                "total_documents": memory_stats.get("total_documents", 0),
                "total_analyses": memory_stats.get("total_analyses", 0),
                "health": "good" if memory_stats.get("total_documents", 0) > 0 else "empty"
            },
            "tool_registry": tool_stats,
            "session_state": {
                "current_focus": self.session_state.get("current_research_focus"),
                "papers_analyzed": self.session_state.get("papers_analyzed", 0),
                "active_capabilities": self.session_state.get("active_capabilities", []),
                "usage_entries": len(self.session_state.get("capability_usage", []))
            },
            "citation_tracker": {
                "total_sources": len(self.citation_tracker.get_all_sources()),
                "total_insights": len(self.citation_tracker.insights)
            }
        }