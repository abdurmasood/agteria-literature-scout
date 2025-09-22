"""Business capability module for breakthrough potential analysis."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool

from .base import CapabilityModule, CapabilityError
from ...config import Config
from ...tools.search_tools import create_langchain_search_tools
from ...tools.analysis_tools import create_langchain_analysis_tools
from ...tools.business_tools import create_langchain_business_tools

logger = logging.getLogger(__name__)

class BusinessCapability(CapabilityModule):
    """
    Business capability focused on breakthrough potential analysis.
    
    This capability handles:
    - Technical feasibility assessment
    - Market viability analysis
    - Competitive landscape evaluation
    - Regulatory pathway assessment
    - Investment recommendations
    - Strategic business intelligence
    """
    
    def __init__(self, core):
        """Initialize business capability."""
        super().__init__(core, "business")
    
    def _register_tools(self) -> List[Tool]:
        """Register business-specific tools."""
        tools = []
        
        # Add business analysis tools
        tools.extend(create_langchain_business_tools())
        
        # Add limited search tools for competitive intelligence
        search_tools = create_langchain_search_tools(self.core.get_shared_citations())
        essential_search_tools = [
            tool for tool in search_tools 
            if tool.name in ['pubmed_search', 'web_search']
        ]
        tools.extend(essential_search_tools)
        
        # Add focused analysis tools
        analysis_tools = create_langchain_analysis_tools()
        essential_analysis_tools = [
            tool for tool in analysis_tools 
            if tool.name in ['analyze_paper', 'extract_molecular_targets']
        ]
        tools.extend(essential_analysis_tools)
        
        # Add memory tools from core
        tools.extend(self.core.create_memory_tools())
        
        return tools
    
    def _create_agent(self):
        """Create LangGraph agent for business analysis."""
        # Get LLM optimized for business analysis
        llm = self.core.get_llm(
            temperature=0.4,  # Balanced creativity and accuracy
            max_tokens=4000   # Allow for comprehensive analysis
        )
        
        # Create memory checkpointer for stateful execution
        checkpointer = MemorySaver()
        
        # Create LangGraph agent
        agent = create_react_agent(
            model=llm,
            tools=self.tools,
            checkpointer=checkpointer
        )
        
        return agent
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute business analysis task.
        
        Args:
            task: Business analysis task containing:
                - query: Research findings to analyze
                - callbacks: Optional callback handlers for streaming
        
        Returns:
            Business analysis results dictionary
        """
        try:
            # Validate task
            self._validate_task(task)
            
            research_findings = task['query']
            callbacks = task.get('callbacks', [])
            
            logger.info(f"Starting business analysis for: {research_findings[:100]}...")
            
            # Create unique thread ID for this analysis with increased limits
            thread_id = f"business_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": Config.LANGGRAPH_RECURSION_LIMIT
            }
            
            # Create focused business analysis prompt
            analysis_prompt = self._create_business_prompt(research_findings)
            
            # Execute LangGraph agent with streaming
            final_response = None
            for event in self.agent.stream(
                {"messages": [("human", analysis_prompt)]},
                config=config,
                stream_mode="values"
            ):
                if self.core.verbose:
                    logger.info(f"Business agent event: {event.keys()}")
                
                # Handle streaming callbacks for UI
                if callbacks:
                    self._handle_streaming_event(event, callbacks)
                
                # Capture final response
                if "messages" in event:
                    final_response = event["messages"][-1].content
            
            # Structure the results
            structured_result = self._structure_business_result(
                research_findings=research_findings,
                agent_response=final_response,
                thread_id=thread_id,
                task=task
            )
            
            # Store analysis in memory for future reference
            self._store_analysis_in_memory(structured_result)
            
            logger.info("Business analysis completed successfully")
            return structured_result
            
        except Exception as e:
            return self._handle_execution_error(e, task)
    
    def _create_business_prompt(self, research_findings: str) -> str:
        """Create focused prompt for business analysis."""
        return f"""
You are an expert business analyst for Agteria Biotech, specializing in evaluating breakthrough potential of climate technology research. Your goal is to provide comprehensive business intelligence on research findings.

RESEARCH FINDINGS TO ANALYZE:
{research_findings}

YOUR ANALYSIS APPROACH:
1. **Technical Validation**: Use search tools to find similar technologies and validate claims
2. **Competitive Intelligence**: Research existing solutions and competitive landscape  
3. **Business Assessment**: Evaluate commercial viability, market potential, and strategic fit
4. **Risk Analysis**: Identify technical, market, and regulatory risks
5. **Strategic Recommendation**: Provide clear proceed/pause/pivot decision

AVAILABLE TOOLS FOR ANALYSIS:
- assess_technical_feasibility: Evaluate technical readiness and scalability
- analyze_market_viability: Assess commercial potential and market opportunity
- evaluate_competitive_landscape: Research competition and positioning
- assess_regulatory_pathway: Evaluate approval requirements and timeline
- generate_investment_recommendation: Synthesize final recommendation
- pubmed_search: Search medical literature for validation
- web_search: Search for market and competitive intelligence
- search_memory: Check previous research on similar topics

REQUIRED OUTPUT FORMAT:
Provide a comprehensive breakthrough analysis report with the following structure:

## 🚀 Breakthrough Potential Analysis

### Executive Summary
[2-3 sentences summarizing the opportunity and recommendation]

### Technical Feasibility Assessment
[Use assess_technical_feasibility tool and summarize findings]

### Market Viability Analysis  
[Use analyze_market_viability tool and summarize findings]

### Competitive Landscape
[Use evaluate_competitive_landscape tool and summarize findings]

### Regulatory Pathway
[Use assess_regulatory_pathway tool and summarize findings]

### Investment Recommendation
[Use generate_investment_recommendation tool with all previous analyses]

### Strategic Priorities
- **Immediate Actions**: [Next 3-6 months]
- **Development Phase**: [6-18 months] 
- **Market Entry**: [18-36 months]

Begin your analysis by using the appropriate tools to gather intelligence, then synthesize into the structured report format.
"""
    
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
        """Extract tool call information from LangGraph message for business analysis."""
        import re
        
        content = message.content if hasattr(message, 'content') else str(message)
        
        # Default values for business analysis
        tool_info = {
            'tool_name': 'business_analysis',
            'tool_input': 'Processing business analysis...',
            'reasoning': '',
            'has_results': False,
            'results': ''
        }
        
        # Business-specific tool patterns
        tool_patterns = {
            'analyze_market_potential': r"analyz(?:ing|e).*?market.*?potential",
            'assess_competitive_landscape': r"assess(?:ing)?.*?competitiv(?:e|tion)",
            'evaluate_technical_feasibility': r"evaluat(?:ing|e).*?technical.*?feasibilit",
            'identify_key_stakeholders': r"identif(?:ying|y).*?stakeholders",
            'analyze_regulatory_environment': r"analyz(?:ing|e).*?regulat(?:ory|ions)",
            'estimate_market_size': r"estimat(?:ing|e).*?market.*?size",
            'assess_monetization_strategies': r"assess(?:ing)?.*?monetization",
            'identify_partnerships': r"identif(?:ying|y).*?partnerships",
            'analyze_implementation_timeline': r"analyz(?:ing|e).*?implementation.*?timeline",
            'evaluate_investment_requirements': r"evaluat(?:ing|e).*?investment",
            'arxiv_search': r"(?:searching|search|query).*?arxiv.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'pubmed_search': r"(?:searching|search|query).*?pubmed.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)",
            'web_search': r"(?:searching|search).*?(?:the\s+)?web.*?(?:for|about)\s+(.+?)(?:\.|$|:|;|\n)"
        }
        
        # Check for specific tool mentions
        for tool_name, pattern in tool_patterns.items():
            if re.search(pattern, content.lower()) or tool_name in content.lower():
                tool_info['tool_name'] = tool_name
                match = re.search(pattern, content.lower())
                if match and match.groups():
                    tool_info['tool_input'] = match.group(1).strip()[:80]
                else:
                    # Extract content after common trigger words
                    for trigger in ["for", "about", "on", "regarding", "of"]:
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
            "found", "analyzed", "evaluated", "assessed", "identified", "estimated",
            "completed", "determined", "discovered", "concluded"
        ]
        
        if any(indicator in content.lower() for indicator in result_indicators):
            tool_info['has_results'] = True
            tool_info['results'] = content[:200]
        
        # Handle special business analysis cases
        if "business" in content.lower() and "analysis" in content.lower():
            tool_info['tool_name'] = 'business_analysis'
        
        # Make tool input more descriptive if it's still generic
        if tool_info['tool_input'] in ['Processing business analysis...', ''] and len(content) > 20:
            # Extract key business terms from the content
            words = re.findall(r'\b\w{4,}\b', content.lower())
            if words:
                meaningful_words = [w for w in words[:5] if w not in 
                                  ['will', 'analysis', 'using', 'tool', 'need', 'business', 'evaluate']]
                if meaningful_words:
                    tool_info['tool_input'] = " ".join(meaningful_words[:3])
        
        return tool_info
    
    def _structure_business_result(self, research_findings: str, agent_response: str, 
                                 thread_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Structure business analysis results."""
        try:
            # Base result from parent class
            result = self._structure_result(agent_response, task, "completed")
            
            # Add business-specific fields
            result.update({
                "research_findings": research_findings,
                "analysis_response": agent_response,  # Keep for backward compatibility
                "thread_id": thread_id,
                "analysis_type": "breakthrough_potential",
                "executive_summary": self._extract_executive_summary(agent_response),
                "recommendation": self._extract_recommendation(agent_response)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error structuring business result: {e}")
            return {
                "research_findings": research_findings,
                "response": agent_response,
                "error": str(e),
                "status": "completed_with_errors",
                "timestamp": datetime.now().isoformat(),
                "capability": self.capability_name
            }
    
    def _extract_executive_summary(self, response: str) -> str:
        """Extract executive summary from response."""
        lines = response.split('\n')
        in_summary = False
        summary_lines = []
        
        for line in lines:
            if "Executive Summary" in line:
                in_summary = True
                continue
            elif line.startswith("###") and in_summary:
                break
            elif in_summary and line.strip():
                summary_lines.append(line.strip())
        
        return ' '.join(summary_lines) if summary_lines else "No executive summary available"
    
    def _extract_recommendation(self, response: str) -> str:
        """Extract investment recommendation from response."""
        if "Investment Recommendation:" in response:
            recommendation_section = response.split("Investment Recommendation:")[1]
            if "PROCEED" in recommendation_section.upper():
                return "PROCEED"
            elif "CAUTIOUS" in recommendation_section.upper():
                return "CAUTIOUS"
            elif "PASS" in recommendation_section.upper():
                return "PASS"
        return "REVIEW"
    
    def _store_analysis_in_memory(self, result: Dict[str, Any]):
        """Store business analysis results in research memory."""
        try:
            from langchain_core.documents import Document
            
            analysis_doc = Document(
                page_content=result.get("response", ""),
                metadata={
                    "document_type": "business_analysis",
                    "research_findings": result.get("research_findings", ""),
                    "recommendation": result.get("recommendation", "unknown"),
                    "thread_id": result.get("thread_id", ""),
                    "analysis_date": result.get("timestamp", ""),
                    "source_database": "business_analyzer",
                    "title": f"Business Analysis: {result.get('research_findings', '')[:50]}..."
                }
            )
            
            # Store in memory
            self.core.memory.add_papers([analysis_doc])
            logger.info("Stored business analysis in memory")
            
        except Exception as e:
            logger.warning(f"Failed to store analysis in memory: {e}")
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent business analyses from memory."""
        try:
            # Search memory for previous analyses
            analyses = self.core.memory.search_papers(
                query="business analysis breakthrough",
                k=limit,
                filter_dict={"document_type": "business_analysis"}
            )
            
            history = []
            for doc in analyses:
                history.append({
                    "research_findings": doc.metadata.get("research_findings", ""),
                    "recommendation": doc.metadata.get("recommendation", ""),
                    "analysis_date": doc.metadata.get("analysis_date", ""),
                    "thread_id": doc.metadata.get("thread_id", "")
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
    
    def generate_fallback_analysis(self, research_findings: str) -> str:
        """Generate simple fallback analysis when LangGraph fails."""
        try:
            # Use direct LLM call as fallback
            llm = self.core.get_llm(temperature=0.3)
            
            fallback_prompt = f"""
            Provide a basic breakthrough analysis for these research findings:
            
            {research_findings}
            
            Include:
            1. Technical feasibility assessment (1-10 score)
            2. Market potential evaluation  
            3. Key risks and challenges
            4. Recommendation (Proceed/Cautious/Pass)
            """
            
            response = llm.invoke(fallback_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Fallback analysis also failed: {e}")
            return f"Unable to complete analysis due to technical issues: {str(e)}"
    
    def _get_troubleshooting_tips(self) -> List[str]:
        """Get business-specific troubleshooting tips."""
        return [
            "Check that LangGraph is installed: pip install langgraph>=0.2.0",
            "Verify OpenAI API key is working and has sufficient credits",
            "Check internet connectivity for competitive intelligence searches",
            "Try with simpler, more focused research findings",
            "Ensure the research findings are clearly described",
            "Contact support if LangGraph execution continues to fail"
        ]