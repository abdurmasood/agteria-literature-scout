"""Base class for agent capabilities."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

class CapabilityError(Exception):
    """Exception raised by capability modules."""
    pass

class CapabilityModule(ABC):
    """
    Abstract base class for agent capabilities.
    
    Each capability module represents a specialized skill set that the unified agent
    can use to handle specific types of tasks. Examples include:
    - ResearchCapability: Literature discovery and analysis
    - BusinessCapability: Commercial viability assessment
    - AnalyticsCapability: Trend analysis and reporting
    """
    
    def __init__(self, core, capability_name: str):
        """
        Initialize capability module.
        
        Args:
            core: AgentCore instance providing shared services
            capability_name: Unique name for this capability
        """
        self.core = core
        self.capability_name = capability_name
        self.initialized_at = datetime.now().isoformat()
        
        # Register tools specific to this capability
        self.tools = self._register_tools()
        self.core.tool_registry.register_tool_set(capability_name, self.tools)
        
        # Create the agent for this capability
        self.agent = self._create_agent()
        
        # Track this capability as active
        active_capabilities = self.core.session_state.get("active_capabilities", [])
        if capability_name not in active_capabilities:
            active_capabilities.append(capability_name)
            self.core.update_session_state({"active_capabilities": active_capabilities})
        
        logger.info(f"{capability_name} capability initialized with {len(self.tools)} tools")
    
    @abstractmethod
    def _register_tools(self) -> List[Tool]:
        """
        Register tools specific to this capability.
        
        Returns:
            List of tools this capability provides
        """
        pass
    
    @abstractmethod
    def _create_agent(self):
        """
        Create the agent instance for this capability.
        
        Returns:
            Agent instance (AgentExecutor, LangGraph agent, etc.)
        """
        pass
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using this capability.
        
        Args:
            task: Task specification dictionary containing:
                - query: The main task query
                - context: Optional context information
                - callbacks: Optional callback handlers
                - **kwargs: Additional parameters
        
        Returns:
            Result dictionary with standardized format:
                - status: 'completed', 'failed', or 'partial'
                - response: Main response content
                - timestamp: Execution timestamp
                - capability: Name of capability used
                - **additional fields specific to capability
        """
        pass
    
    def get_capability_info(self) -> Dict[str, Any]:
        """
        Get information about this capability.
        
        Returns:
            Dictionary with capability metadata
        """
        return {
            "name": self.capability_name,
            "initialized_at": self.initialized_at,
            "tools_count": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "description": self.__class__.__doc__ or "No description available",
            "agent_type": type(self.agent).__name__
        }
    
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate that task contains required fields.
        
        Args:
            task: Task dictionary to validate
        
        Returns:
            True if task is valid
        
        Raises:
            CapabilityError: If task is invalid
        """
        required_fields = ['query']
        missing_fields = [field for field in required_fields if field not in task]
        
        if missing_fields:
            raise CapabilityError(f"Missing required task fields: {missing_fields}")
        
        return True
    
    def _structure_result(self, result: Any, task: Dict[str, Any], status: str = "completed") -> Dict[str, Any]:
        """
        Structure the result in standardized format.
        
        Args:
            result: Raw result from agent execution
            task: Original task specification  
            status: Execution status
        
        Returns:
            Standardized result dictionary
        """
        # Extract response content based on result type
        if isinstance(result, dict):
            response = result.get('output', result.get('response', str(result)))
        elif hasattr(result, 'content'):
            response = result.content
        else:
            response = str(result)
        
        # Log capability usage
        result_summary = f"{status}: {len(response) if response else 0} chars"
        self.core.log_capability_usage(
            capability_name=self.capability_name,
            task=task.get('query', 'Unknown task')[:100],
            result_summary=result_summary
        )
        
        return {
            "status": status,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "capability": self.capability_name,
            "task_query": task.get('query', ''),
            "execution_info": {
                "agent_type": type(self.agent).__name__,
                "tools_available": len(self.tools),
                "context_provided": bool(task.get('context'))
            }
        }
    
    def _handle_execution_error(self, error: Exception, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle execution errors gracefully.
        
        Args:
            error: Exception that occurred
            task: Original task specification
        
        Returns:
            Error result dictionary
        """
        error_msg = str(error)
        logger.error(f"{self.capability_name} execution error: {error_msg}")
        
        # Log the error
        self.core.log_capability_usage(
            capability_name=self.capability_name,
            task=task.get('query', 'Unknown task')[:100],
            result_summary=f"ERROR: {error_msg[:100]}"
        )
        
        return {
            "status": "failed",
            "error": error_msg,
            "response": f"Capability execution failed: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "capability": self.capability_name,
            "task_query": task.get('query', ''),
            "troubleshooting": self._get_troubleshooting_tips(),
            "execution_info": {
                "agent_type": type(self.agent).__name__,
                "tools_available": len(self.tools),
                "error_type": type(error).__name__
            }
        }
    
    def _get_troubleshooting_tips(self) -> List[str]:
        """
        Get troubleshooting tips for this capability.
        
        Returns:
            List of troubleshooting suggestions
        """
        return [
            f"Check that {self.capability_name} capability is properly initialized",
            "Verify that all required API keys are configured",
            "Try simplifying the query if it's too complex",
            "Check network connectivity for external service calls",
            f"Review {self.capability_name} capability logs for detailed error information"
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this capability.
        
        Returns:
            Usage statistics dictionary
        """
        usage_history = self.core.session_state.get("capability_usage", [])
        capability_usage = [entry for entry in usage_history if entry.get("capability") == self.capability_name]
        
        if not capability_usage:
            return {
                "total_uses": 0,
                "success_rate": 0.0,
                "average_response_length": 0,
                "last_used": None
            }
        
        total_uses = len(capability_usage)
        successful_uses = len([entry for entry in capability_usage if not entry.get("result", "").startswith("ERROR")])
        success_rate = successful_uses / total_uses if total_uses > 0 else 0.0
        
        # Calculate average response length from successful uses
        response_lengths = []
        for entry in capability_usage:
            if not entry.get("result", "").startswith("ERROR"):
                # Extract length from result summary like "completed: 1234 chars"
                result = entry.get("result", "")
                if "chars" in result:
                    try:
                        length = int(result.split(":")[1].strip().split(" ")[0])
                        response_lengths.append(length)
                    except (ValueError, IndexError):
                        pass
        
        avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        return {
            "total_uses": total_uses,
            "success_rate": success_rate,
            "average_response_length": avg_response_length,
            "last_used": capability_usage[-1].get("timestamp") if capability_usage else None,
            "recent_tasks": [entry.get("task", "")[:50] + "..." for entry in capability_usage[-5:]]
        }