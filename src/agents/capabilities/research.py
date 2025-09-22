"""Research capability module for literature discovery and analysis."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langchain_core.documents import Document

from .base import CapabilityModule, CapabilityError
from ...config import Config, AGTERIA_KEYWORDS, CROSS_DOMAIN_KEYWORDS
from ...tools.search_tools import create_langchain_search_tools
from ...tools.analysis_tools import create_langchain_analysis_tools
from ...tools.hypothesis_tools import create_langchain_hypothesis_tools

logger = logging.getLogger(__name__)

class ResearchCapability(CapabilityModule):
    """
    Research capability focused on literature discovery and analysis.
    
    This capability handles:
    - Multi-database literature search (ArXiv, PubMed, Web)
    - Paper analysis and content extraction
    - Cross-domain hypothesis generation
    - Research gap identification
    - Knowledge graph building
    """
    
    def __init__(self, core):
        """Initialize research capability."""
        super().__init__(core, "research")
    
    def _register_tools(self) -> List[Tool]:
        """Register research-specific tools."""
        tools = []
        
        # Add search tools with shared citation tracker
        tools.extend(create_langchain_search_tools(self.core.get_shared_citations()))
        
        # Add analysis tools  
        tools.extend(create_langchain_analysis_tools())
        
        # Add hypothesis generation tools
        tools.extend(create_langchain_hypothesis_tools())
        
        # Add memory tools from core
        tools.extend(self.core.create_memory_tools())
        
        return tools
    
    def _create_agent(self):
        """Create LangGraph agent for research tasks."""
        # Get LLM optimized for research
        llm = self.core.get_llm(temperature=Config.TEMPERATURE)
        
        # Create memory checkpointer for stateful execution
        checkpointer = MemorySaver()
        
        # Create LangGraph agent
        agent = create_react_agent(
            model=llm,
            tools=self.tools,
            checkpointer=checkpointer
        )
        
        return agent
    
    def _create_research_message(self, query: str, focus_areas: List[str]) -> str:
        """Create research message for LangGraph agent."""
        # Build the complete research instruction message
        message = f"""You are the Agteria Literature Scout Research Specialist, an expert AI research assistant specializing in climate technology and methane reduction in livestock. Your primary goal is to help Agteria Biotech discover novel research insights, identify breakthrough opportunities, and generate innovative hypotheses.

CORE EXPERTISE:
- Methane inhibition in cattle and livestock
- Enzyme inhibitors and fermentation control
- Agricultural climate solutions
- Cross-domain innovation (marine biology, pharmaceuticals, materials science)
- Scientific literature analysis and synthesis

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

KEY PRIORITIES FOR AGTERIA:
- Novel methane inhibition mechanisms
- Scalable and cost-effective solutions
- Safety and efficacy in livestock
- Cross-industry applicable technologies
- Breakthrough molecular discoveries

FINAL ANSWER STRUCTURE:
Your final response must be formatted as clean markdown with the following sections:
## Overview
Overview of the research query and the main findings.

## 💡 Novel Insights
1. First insight with [ID: paper_id] citation
2. Second insight with [ID: paper_id] citation

## 🧪 Generated Hypotheses
1. First hypothesis with supporting [ID: paper_id] citations
2. Second hypothesis with supporting [ID: paper_id] citations

## 📋 Recommended Next Steps
1. First action step with relevant source references
2. Second action step with relevant source references

## 📖 Source Summary
List of all Paper IDs referenced with their titles

Focus Areas: {', '.join(focus_areas) if focus_areas else 'General research'}

Current Research Query: {query}

Begin your research and remember: NO CLAIM WITHOUT CITATION!"""
        return message
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research task using LangGraph agent.
        
        Args:
            task: Research task containing:
                - query: Research question or topic
                - focus_areas: Optional specific areas to focus on
                - callbacks: Optional callback handlers for streaming
        
        Returns:
            Research results dictionary
        """
        try:
            # Validate task
            self._validate_task(task)
            
            query = task['query']
            focus_areas = task.get('focus_areas', [])
            callbacks = task.get('callbacks', [])
            
            logger.info(f"Starting research on: {query}")
            
            # Update session state
            self.core.update_session_state({
                "current_research_focus": query,
                "research_start_time": datetime.now().isoformat()
            })
            
            # Create unique thread ID for this research session
            thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Enhance query with Agteria context and create research message
            enhanced_query = self._enhance_query_with_context(query, focus_areas)
            research_message = self._create_research_message(query, focus_areas)
            
            # Execute LangGraph agent with streaming
            final_response = None
            for event in self.agent.stream(
                {"messages": [("human", f"{research_message}\n\n{enhanced_query}")]},
                config=config,
                stream_mode="values"
            ):
                if self.core.verbose:
                    logger.info(f"Research agent event: {event.keys()}")
                
                # Handle streaming callbacks for UI
                if callbacks:
                    self._handle_streaming_event(event, callbacks)
                
                # Capture final response
                if "messages" in event:
                    final_response = event["messages"][-1].content
            
            # Check if agent completed successfully
            if not final_response or "Agent stopped due to" in final_response:
                logger.warning(f"Agent may have hit limits. Output: {final_response[:200] if final_response else 'No response'}...")
                return self._handle_incomplete_execution(query, final_response)
            
            # Structure and enhance results
            structured_result = self._structure_research_result(
                query=query,
                agent_response=final_response,
                focus_areas=focus_areas,
                task=task
            )
            
            # Add thread_id for tracking
            structured_result["thread_id"] = thread_id
            
            # Update knowledge graph with findings
            self._update_knowledge_from_research(structured_result)
            
            # Update session statistics
            self.core.update_session_state({
                "papers_analyzed": self.core.session_state.get("papers_analyzed", 0) + structured_result.get("papers_found", 0),
                "last_research_completed": datetime.now().isoformat()
            })
            
            logger.info(f"Research completed successfully for: {query}")
            return structured_result
            
        except Exception as e:
            return self._handle_execution_error(e, task)
    
    def _enhance_query_with_context(self, query: str, focus_areas: List[str]) -> str:
        """Enhance research query with Agteria context."""
        # Add domain context
        domain_context = "Context: You are researching for Agteria Biotech, focused on methane reduction in livestock through innovative biotechnology solutions.\n\n"
        
        # Add focus areas if provided
        if focus_areas:
            domain_context += f"Focus Areas: {', '.join(focus_areas)}\n\n"
        
        # Add relevant keywords
        keyword_context = f"Key Areas of Interest: {', '.join(AGTERIA_KEYWORDS[:5])}\n\n"
        
        return domain_context + keyword_context + f"Research Query: {query}"
    
    def _handle_streaming_event(self, event: Dict[str, Any], callbacks: List):
        """Handle streaming events for UI callbacks."""
        try:
            if "messages" in event and callbacks:
                latest_message = event["messages"][-1]
                
                # Parse message to extract tool information
                tool_info = self._extract_tool_info_from_message(latest_message)
                
                for callback in callbacks:
                    if hasattr(callback, 'on_agent_action'):
                        # Create mock agent action for callback compatibility
                        class MockAction:
                            def __init__(self, tool, tool_input, log):
                                self.tool = tool
                                self.tool_input = tool_input
                                self.log = log
                        
                        mock_action = MockAction(
                            tool=tool_info['tool_name'],
                            tool_input=tool_info['tool_input'],
                            log=tool_info['reasoning']
                        )
                        
                        callback.on_agent_action(mock_action)
                        
                        # Handle tool results if present
                        if tool_info.get('has_results') and hasattr(callback, 'on_tool_end'):
                            callback.on_tool_end(tool_info['results'])
                        
        except Exception as e:
            logger.warning(f"Error handling streaming event: {e}")
    
    def _extract_tool_info_from_message(self, message) -> Dict[str, Any]:
        """Extract tool call information from LangGraph message."""
        import re
        
        content = message.content if hasattr(message, 'content') else str(message)
        
        # Default values
        tool_info = {
            'tool_name': 'research_analysis',
            'tool_input': 'Processing research...',
            'reasoning': '',
            'has_results': False,
            'results': ''
        }
        
        # Detect tool calls in message content
        tool_patterns = {
            'arxiv_search': r"(?:searching|search|query).*?arxiv.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'pubmed_search': r"(?:searching|search|query).*?pubmed.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'web_search': r"(?:searching|search).*?(?:the\s+)?web.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'comprehensive_search': r"(?:comprehensive|broad).*?search.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'analyze_paper': r"analyz(?:ing|e).*?(?:paper|document|research).*?(?:titled|called|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'cross_domain_hypotheses': r"generat(?:ing|e).*?(?:cross-domain\s+)?hypothes[ei]s",
            'analogical_hypotheses': r"generat(?:ing|e).*?analogical.*?hypothes[ei]s",
            'research_gap_analysis': r"(?:identifying|analyzing).*?research.*?gaps",
            'creative_ideation': r"generat(?:ing|e).*?creative.*?ideas",
            'extract_molecules': r"extract(?:ing)?.*?(?:molecular|chemical).*?information",
            'score_relevance': r"scor(?:ing|e).*?relevance",
            'memory_stats': r"(?:checking|retrieving).*?memory.*?stats",
            'search_memory': r"search(?:ing)?.*?memory"
        }
        
        # Check for specific tool mentions first
        for tool_name, pattern in tool_patterns.items():
            if re.search(pattern, content.lower()) or tool_name in content.lower():
                tool_info['tool_name'] = tool_name
                match = re.search(pattern, content.lower())
                if match and match.groups():
                    tool_info['tool_input'] = match.group(1).strip()[:80]
                else:
                    # Extract content after common trigger words
                    for trigger in ["for", "about", "on", "regarding"]:
                        if trigger in content.lower():
                            after_trigger = content.lower().split(trigger, 1)[1]
                            clean_input = re.sub(r'[^\w\s]', ' ', after_trigger).strip()[:80]
                            if clean_input:
                                tool_info['tool_input'] = clean_input
                                break
                break
        
        # Extract reasoning from common agent patterns
        reasoning_patterns = [
            r"I'll (.+?)(?:\.|$|\n)",
            r"Let me (.+?)(?:\.|$|\n)",
            r"I need to (.+?)(?:\.|$|\n)",
            r"I will (.+?)(?:\.|$|\n)",
            r"Now I'll (.+?)(?:\.|$|\n)"
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, content)
            if match:
                tool_info['reasoning'] = f"Agent: {match.group(1).strip()}"
                break
        
        # Check for results indicators
        result_indicators = [
            "found", "papers", "results", "discovered", "identified", "generated",
            "completed", "extracted", "analyzed", "scored"
        ]
        
        if any(indicator in content.lower() for indicator in result_indicators):
            tool_info['has_results'] = True
            tool_info['results'] = content[:200]
        
        # Handle special cases where tool name is explicitly mentioned
        if "using" in content.lower() and "tool" in content.lower():
            tool_mention = re.search(r"using\s+(\w+).*?tool", content.lower())
            if tool_mention:
                tool_info['tool_name'] = tool_mention.group(1)
        
        # Make tool input more descriptive if it's still generic
        if tool_info['tool_input'] in ['Processing research...', ''] and len(content) > 20:
            # Extract key terms from the content
            words = re.findall(r'\b\w{4,}\b', content.lower())
            if words:
                meaningful_words = [w for w in words[:5] if w not in 
                                  ['will', 'search', 'using', 'tool', 'need', 'analyze', 'generate']]
                if meaningful_words:
                    tool_info['tool_input'] = " ".join(meaningful_words[:3])
        
        return tool_info
    
    def _structure_research_result(self, query: str, agent_response: str, focus_areas: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Structure research results into standardized format."""
        # Base result from parent class
        result = self._structure_result(agent_response, task, "completed")
        
        # Add research-specific fields
        result.update({
            "query": query,
            "focus_areas": focus_areas,
            "papers_found": self._count_papers_in_response(agent_response),
            "novel_insights": self._extract_insights_from_response(agent_response),
            "hypotheses": self._extract_hypotheses_from_response(agent_response),
            "next_steps": self._extract_next_steps_from_response(agent_response),
            "sources": self._extract_sources_from_response(agent_response),
            "paper_ids_referenced": self._extract_paper_ids_from_response(agent_response),
            "analysis_type": "literature_research"
        })
        
        return result
    
    def _handle_incomplete_execution(self, query: str, agent_output: str) -> Dict[str, Any]:
        """Handle cases where agent execution was incomplete."""
        return {
            "status": "partial",
            "query": query,
            "response": agent_output or "Agent execution incomplete - may have hit iteration or time limits",
            "error": "Agent execution incomplete",
            "papers_found": 0,
            "novel_insights": [],
            "hypotheses": [],
            "next_steps": ["Try a more specific query", "Check API keys and network connectivity"],
            "sources": [],
            "paper_ids_referenced": [],
            "timestamp": datetime.now().isoformat(),
            "capability": self.capability_name,
            "troubleshooting_tips": [
                "The agent may need more time for complex research queries",
                "Try breaking down the query into smaller, more specific questions",
                "Check that all API keys are properly configured",
                "Verify network connectivity to research databases"
            ]
        }
    
    def _count_papers_in_response(self, response: str) -> int:
        """Count paper references in response."""
        import re
        # Count unique paper ID references
        paper_ids = re.findall(r'\[ID: ([^\]]+)\]', response)
        return len(set(paper_ids))
    
    def _extract_insights_from_response(self, response: str) -> List[str]:
        """Extract novel insights from research response."""
        insights = []
        lines = response.split('\n')
        
        in_insights_section = False
        for line in lines:
            if "💡 Novel Insights" in line or "## Novel Insights" in line:
                in_insights_section = True
                continue
            elif line.startswith("##") or line.startswith("🧪") or line.startswith("📋"):
                in_insights_section = False
            elif in_insights_section and line.strip() and (line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-"))):
                insight = line.strip()
                # Clean up numbering and formatting
                insight = insight.lstrip("123456789.-").strip()
                if insight and len(insight) > 10:
                    insights.append(insight)
        
        return insights[:5]  # Limit to top 5
    
    def _extract_hypotheses_from_response(self, response: str) -> List[str]:
        """Extract hypotheses from research response."""
        hypotheses = []
        lines = response.split('\n')
        
        in_hypotheses_section = False
        for line in lines:
            if "🧪 Generated Hypotheses" in line or "## Generated Hypotheses" in line:
                in_hypotheses_section = True
                continue
            elif line.startswith("##") or line.startswith("📋"):
                in_hypotheses_section = False
            elif in_hypotheses_section and line.strip() and (line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-"))):
                hypothesis = line.strip()
                hypothesis = hypothesis.lstrip("123456789.-").strip()
                if hypothesis and len(hypothesis) > 10:
                    hypotheses.append(hypothesis)
        
        return hypotheses[:5]  # Limit to top 5
    
    def _extract_next_steps_from_response(self, response: str) -> List[str]:
        """Extract next steps from research response."""
        next_steps = []
        lines = response.split('\n')
        
        in_steps_section = False
        for line in lines:
            if "📋 Recommended Next Steps" in line or "## Next Steps" in line:
                in_steps_section = True
                continue
            elif line.startswith("##") or line.startswith("📖"):
                in_steps_section = False
            elif in_steps_section and line.strip() and (line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-"))):
                step = line.strip()
                step = step.lstrip("123456789.-").strip()
                if step and len(step) > 5:
                    next_steps.append(step)
        
        return next_steps[:5]  # Limit to top 5
    
    def _extract_sources_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract source information from response."""
        sources = []
        
        # Get sources from shared citation tracker
        all_sources = self.core.get_shared_citations().get_all_sources()
        
        # Find which sources were referenced in this response
        import re
        referenced_ids = re.findall(r'\[ID: ([^\]]+)\]', response)
        
        # Match sources by exact ID
        for source in all_sources:
            if source.id in referenced_ids:
                sources.append(source.to_dict())
        
        return sources
    
    def _extract_paper_ids_from_response(self, response: str) -> List[str]:
        """Extract paper IDs referenced in response."""
        import re
        paper_ids = re.findall(r'\[ID: ([^\]]+)\]', response)
        return list(set(paper_ids))  # Remove duplicates
    
    def _update_knowledge_from_research(self, research_result: Dict[str, Any]):
        """Update knowledge graph with research findings."""
        try:
            # Update knowledge graph if we have new insights
            insights = research_result.get("novel_insights", [])
            if insights:
                # Add insights to recent discoveries
                recent_discoveries = self.core.session_state.get("recent_discoveries", [])
                recent_discoveries.extend(insights)
                # Keep only last 20 discoveries
                recent_discoveries = recent_discoveries[-20:]
                self.core.update_session_state({"recent_discoveries": recent_discoveries})
            
            # Update hypothesis count
            hypotheses_count = len(research_result.get("hypotheses", []))
            current_count = self.core.session_state.get("hypothesis_count", 0)
            self.core.update_session_state({"hypothesis_count": current_count + hypotheses_count})
            
        except Exception as e:
            logger.warning(f"Error updating knowledge graph: {e}")
    
    def get_research_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent research history."""
        usage_history = self.core.session_state.get("capability_usage", [])
        research_usage = [
            entry for entry in usage_history 
            if entry.get("capability") == self.capability_name
        ]
        
        return research_usage[-limit:]  # Return most recent
    
    def _get_troubleshooting_tips(self) -> List[str]:
        """Get research-specific troubleshooting tips."""
        return [
            "Check that API keys are properly configured for all databases",
            "Verify network connectivity to ArXiv, PubMed, and web search services",
            "Try using more specific research terms",
            "Break complex queries into smaller, focused questions",
            "Check if the research area has sufficient literature available",
            "Ensure the query is related to Agteria's focus areas for best results"
        ]