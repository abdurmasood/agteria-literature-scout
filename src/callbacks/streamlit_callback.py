"""Streamlit callback handler for real-time agent execution streaming."""

import time
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming agent execution to Streamlit UI."""
    
    def __init__(self, status_container=None, history_container=None, max_history: int = 20, update_throttle: float = 0.5):
        """
        Initialize the Streamlit callback handler.
        
        Args:
            status_container: Streamlit container for current status
            history_container: Streamlit container for step history
            max_history: Maximum number of steps to keep in history
            update_throttle: Minimum seconds between UI updates for performance
        """
        super().__init__()
        self.status_container = status_container
        self.history_container = history_container
        self.max_history = max_history
        self.update_throttle = update_throttle
        
        # Internal state
        self.steps = []
        self.start_time = None
        self.current_tool = None
        self.step_count = 0
        self.last_update_time = 0
        
        # Formatting configuration
        self.icons = {
            'start': '🚀',
            'thinking': '🤔',
            'searching': '🔍',
            'analyzing': '🧠',
            'found': '📄',
            'complete': '✅',
            'error': '❌',
            'tool': '🔧'
        }
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when the agent chain starts."""
        self.start_time = time.time()
        message = "Starting research analysis..."
        self._update_status(self.icons['start'], message)
        self._add_to_history(self.icons['start'], message)
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when the agent takes an action."""
        self.step_count += 1
        
        # Extract and format the agent's thought process
        thought = getattr(action, 'log', '')
        if thought:
            # Parse the thought to extract the reasoning
            thought_text = self._extract_thought(thought)
            if thought_text:
                message = f"Thinking: {thought_text}"
                self._update_status(self.icons['thinking'], message)
                self._add_to_history(self.icons['thinking'], message)
        
        # Show which tool is being used
        tool_name = action.tool
        tool_input = str(action.tool_input)
        
        # Format tool usage message
        if tool_name == 'arxiv_search':
            message = f"Searching ArXiv: {self._truncate(tool_input, 60)}"
            icon = self.icons['searching']
        elif tool_name == 'pubmed_search':
            message = f"Searching PubMed: {self._truncate(tool_input, 60)}"
            icon = self.icons['searching']
        elif tool_name == 'web_search':
            message = f"Searching Web: {self._truncate(tool_input, 60)}"
            icon = self.icons['searching']
        elif 'analysis' in tool_name.lower():
            message = f"Analyzing: {self._truncate(tool_input, 60)}"
            icon = self.icons['analyzing']
        elif 'hypothesis' in tool_name.lower():
            message = f"Generating hypotheses: {self._truncate(tool_input, 50)}"
            icon = self.icons['analyzing']
        else:
            message = f"Using {tool_name}: {self._truncate(tool_input, 50)}"
            icon = self.icons['tool']
        
        self.current_tool = tool_name
        self._update_status(icon, message)
        self._add_to_history(icon, message)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        # This provides additional granularity if needed
        pass
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes running."""
        if self.current_tool:
            # Format tool output summary
            output_summary = self._summarize_tool_output(output, self.current_tool)
            message = f"Found: {output_summary}"
            self._update_status(self.icons['found'], message)
            self._add_to_history(self.icons['found'], message)
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when a tool encounters an error."""
        message = f"Tool error: {str(error)[:100]}"
        self._update_status(self.icons['error'], message)
        self._add_to_history(self.icons['error'], message)
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when the agent finishes."""
        elapsed = self._get_elapsed_time()
        message = f"Research complete in {elapsed}"
        self._update_status(self.icons['complete'], message)
        self._add_to_history(self.icons['complete'], message)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating."""
        # Optional: could show when the agent is thinking deeper
        pass
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes generating."""
        # Optional: could show completion of reasoning
        pass
    
    def _update_status(self, icon: str, message: str) -> None:
        """Update the current status display with throttling."""
        current_time = time.time()
        
        # Throttle updates for performance
        if current_time - self.last_update_time < self.update_throttle:
            return
        
        if self.status_container:
            try:
                formatted_message = f"{icon} {message}"
                self.status_container.markdown(f"**{formatted_message}**")
                self.last_update_time = current_time
            except Exception as e:
                logger.warning(f"Failed to update status container: {e}")
                # Continue execution even if UI update fails
                pass
    
    def _add_to_history(self, icon: str, message: str) -> None:
        """Add a step to the history display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        step = {
            'timestamp': timestamp,
            'icon': icon,
            'message': message,
            'step_number': self.step_count
        }
        
        self.steps.append(step)
        
        # Limit history size
        if len(self.steps) > self.max_history:
            self.steps = self.steps[-self.max_history:]
        
        # Update history display
        if self.history_container:
            try:
                self._render_history()
            except Exception as e:
                logger.warning(f"Failed to update history container: {e}")
                # Continue execution even if UI update fails
                pass
    
    def _render_history(self) -> None:
        """Render the complete step history."""
        if not self.steps:
            return
        
        history_text = ""
        for step in self.steps[-10:]:  # Show last 10 steps
            history_text += f"`{step['timestamp']}` {step['icon']} {step['message']}\n\n"
        
        self.history_container.markdown(history_text)
    
    def _extract_thought(self, log_text: str) -> str:
        """Extract the agent's thought from the log text."""
        lines = log_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Thought:'):
                thought = line.replace('Thought:', '').strip()
                return self._truncate(thought, 120)
        return ""
    
    def _summarize_tool_output(self, output: str, tool_name: str) -> str:
        """Summarize tool output for display."""
        if not output:
            return "No results"
        
        # Different summaries based on tool type
        if 'search' in tool_name.lower():
            # Try to extract paper count
            if 'papers found' in output.lower():
                import re
                match = re.search(r'(\d+)\s+papers?\s+found', output.lower())
                if match:
                    count = match.group(1)
                    return f"{count} papers found"
            
            # Fallback: count lines or estimate
            lines = output.split('\n')
            non_empty_lines = [l for l in lines if l.strip()]
            if len(non_empty_lines) > 5:
                return f"~{len(non_empty_lines)} results found"
            else:
                return "Search results retrieved"
        
        elif 'analysis' in tool_name.lower():
            return "Analysis completed"
        
        elif 'hypothesis' in tool_name.lower():
            # Try to count hypotheses
            if 'hypothesis' in output.lower():
                count = output.lower().count('hypothesis')
                return f"{count} hypotheses generated"
            return "Hypotheses generated"
        
        else:
            return self._truncate(output, 80)
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _get_elapsed_time(self) -> str:
        """Get formatted elapsed time."""
        if not self.start_time:
            return "unknown time"
        
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f} seconds"
        else:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
    
    def get_step_summary(self) -> Dict[str, Any]:
        """Get a summary of all steps taken."""
        return {
            'total_steps': len(self.steps),
            'elapsed_time': self._get_elapsed_time(),
            'steps': self.steps.copy()
        }
    
    def clear_history(self) -> None:
        """Clear the step history."""
        self.steps = []
        self.step_count = 0
        if self.history_container:
            self.history_container.empty()