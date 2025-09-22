"""LangGraph-based breakthrough analysis agent for business intelligence evaluation."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import FileCallbackHandler

from ..config import Config
from ..tools.search_tools import create_langchain_search_tools
from ..tools.analysis_tools import create_langchain_analysis_tools
from ..tools.business_tools import create_langchain_business_tools
from ..memory.research_memory import ResearchMemory

logger = logging.getLogger(__name__)

class BreakthroughAnalyzer:
    """LangGraph-based agent specifically designed for breakthrough potential analysis."""
    
    def __init__(self, memory: ResearchMemory, verbose: bool = True):
        """
        Initialize the breakthrough analyzer.
        
        Args:
            memory: Shared research memory instance
            verbose: Enable verbose logging
        """
        self.memory = memory
        self.verbose = verbose
        
        # Initialize LLM with focused parameters for business analysis
        self.llm = ChatOpenAI(
            model_name=Config.DEFAULT_MODEL,
            temperature=0.4,  # Balanced creativity and accuracy for business analysis
            api_key=Config.OPENAI_API_KEY,
            max_tokens=4000  # Allow for comprehensive analysis
        )
        
        # Create tools for breakthrough analysis
        self.tools = self._create_business_tools()
        
        # Create LangGraph agent with memory
        self.agent = self._create_langgraph_agent()
        
        logger.info("BreakthroughAnalyzer initialized successfully")
    
    def _create_business_tools(self) -> List:
        """Create tools optimized for business analysis."""
        tools = []
        
        # Add business analysis tools
        tools.extend(create_langchain_business_tools())
        
        # Add limited search tools for competitive intelligence
        search_tools = create_langchain_search_tools()
        # Only include essential search tools to avoid complexity
        essential_search_tools = [
            tool for tool in search_tools 
            if tool.name in ['pubmed_search', 'web_search', 'search_memory']
        ]
        tools.extend(essential_search_tools)
        
        # Add focused analysis tools
        analysis_tools = create_langchain_analysis_tools()
        essential_analysis_tools = [
            tool for tool in analysis_tools 
            if tool.name in ['analyze_paper', 'extract_molecular_targets']
        ]
        tools.extend(essential_analysis_tools)
        
        logger.info(f"Created {len(tools)} business analysis tools")
        return tools
    
    def _create_langgraph_agent(self):
        """Create LangGraph agent with checkpointing for stateful execution."""
        # Create memory checkpointer
        checkpointer = MemorySaver()
        
        # Create the LangGraph agent
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=checkpointer
        )
        
        return agent
    
    def analyze_breakthrough_potential(self, research_findings: str, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Analyze breakthrough potential using structured LangGraph workflow.
        
        Args:
            research_findings: Description of research findings to analyze
            callbacks: Optional callback handlers for streaming
            
        Returns:
            Structured breakthrough analysis results
        """
        logger.info("Starting LangGraph breakthrough analysis")
        
        # Create unique thread ID for this analysis
        thread_id = f"breakthrough_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create focused breakthrough analysis prompt
        analysis_prompt = self._create_breakthrough_prompt(research_findings)
        
        try:
            # Execute LangGraph agent with streaming
            final_response = None
            intermediate_results = {}
            
            for event in self.agent.stream(
                {"messages": [("human", analysis_prompt)]},
                config=config,
                stream_mode="values"
            ):
                if self.verbose:
                    logger.info(f"Agent event: {event.keys()}")
                
                # Handle streaming callbacks for UI
                if callbacks:
                    self._handle_streaming_event(event, callbacks)
                
                # Capture final response
                if "messages" in event:
                    final_response = event["messages"][-1].content
            
            # Structure the results
            structured_result = self._structure_breakthrough_result(
                research_findings=research_findings,
                agent_response=final_response,
                thread_id=thread_id
            )
            
            logger.info("LangGraph breakthrough analysis completed successfully")
            return structured_result
            
        except Exception as e:
            logger.error(f"LangGraph breakthrough analysis error: {e}")
            return self._handle_analysis_error(research_findings, e)
    
    def _create_breakthrough_prompt(self, research_findings: str) -> str:
        """Create focused prompt for breakthrough analysis."""
        prompt = f"""
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
        return prompt
    
    def _handle_streaming_event(self, event: Dict[str, Any], callbacks: List):
        """Handle streaming events for UI callbacks."""
        try:
            if "messages" in event and callbacks:
                # Extract latest message for streaming
                latest_message = event["messages"][-1]
                
                # Update callbacks with progress
                for callback in callbacks:
                    if hasattr(callback, 'on_agent_action'):
                        # Simulate agent action for callback compatibility
                        action_info = {
                            'tool': 'breakthrough_analysis',
                            'tool_input': 'Processing breakthrough analysis...',
                            'log': latest_message.content[:200] + "..."
                        }
                        
                        # Create mock agent action object
                        class MockAction:
                            def __init__(self, tool, tool_input, log):
                                self.tool = tool
                                self.tool_input = tool_input
                                self.log = log
                        
                        mock_action = MockAction(
                            action_info['tool'],
                            action_info['tool_input'], 
                            action_info['log']
                        )
                        
                        callback.on_agent_action(mock_action)
                        
        except Exception as e:
            logger.warning(f"Error handling streaming event: {e}")
    
    def _structure_breakthrough_result(
        self, 
        research_findings: str, 
        agent_response: str, 
        thread_id: str
    ) -> Dict[str, Any]:
        """Structure the breakthrough analysis results."""
        try:
            # Extract key components from agent response
            result = {
                "research_findings": research_findings,
                "analysis_response": agent_response,
                "response": agent_response,  # For Streamlit compatibility
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "breakthrough_potential",
                "status": "completed"
            }
            
            # Try to extract structured data from response
            if "Investment Recommendation:" in agent_response:
                # Extract recommendation
                recommendation_section = agent_response.split("Investment Recommendation:")[1]
                if "PROCEED" in recommendation_section.upper():
                    result["recommendation"] = "PROCEED"
                elif "CAUTIOUS" in recommendation_section.upper():
                    result["recommendation"] = "CAUTIOUS"  
                elif "PASS" in recommendation_section.upper():
                    result["recommendation"] = "PASS"
                else:
                    result["recommendation"] = "REVIEW"
            
            # Extract executive summary if present
            if "Executive Summary" in agent_response:
                summary_section = agent_response.split("Executive Summary")[1].split("###")[0]
                result["executive_summary"] = summary_section.strip()
            
            # Store in memory for future reference
            self._store_analysis_in_memory(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error structuring breakthrough result: {e}")
            return {
                "research_findings": research_findings,
                "analysis_response": agent_response,
                "error": str(e),
                "status": "completed_with_errors",
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_analysis_error(self, research_findings: str, error: Exception) -> Dict[str, Any]:
        """Handle analysis errors with fallback response."""
        logger.error(f"Breakthrough analysis failed: {error}")
        
        return {
            "research_findings": research_findings,
            "error": str(error),
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "fallback_analysis": self._generate_fallback_analysis(research_findings),
            "troubleshooting": [
                "Check internet connectivity for competitive intelligence searches",
                "Verify OpenAI API key is working and has sufficient credits", 
                "Try breaking down complex findings into simpler statements",
                "Contact support if the issue persists"
            ]
        }
    
    def _generate_fallback_analysis(self, research_findings: str) -> str:
        """Generate simple fallback analysis when LangGraph fails."""
        try:
            # Use direct LLM call as fallback
            fallback_prompt = f"""
            Provide a basic breakthrough analysis for these research findings:
            
            {research_findings}
            
            Include:
            1. Technical feasibility assessment (1-10 score)
            2. Market potential evaluation  
            3. Key risks and challenges
            4. Recommendation (Proceed/Cautious/Pass)
            """
            
            response = self.llm.invoke(fallback_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Fallback analysis also failed: {e}")
            return f"Unable to complete analysis due to technical issues: {str(e)}"
    
    def _store_analysis_in_memory(self, result: Dict[str, Any]):
        """Store analysis results in research memory for future reference."""
        try:
            # Create document for storage
            from langchain_core.documents import Document
            
            analysis_doc = Document(
                page_content=result.get("analysis_response", ""),
                metadata={
                    "document_type": "breakthrough_analysis",
                    "research_findings": result.get("research_findings", ""),
                    "recommendation": result.get("recommendation", "unknown"),
                    "thread_id": result.get("thread_id", ""),
                    "analysis_date": result.get("timestamp", ""),
                    "source_database": "breakthrough_analyzer"
                }
            )
            
            # Store in memory
            self.memory.add_papers([analysis_doc])
            logger.info(f"Stored breakthrough analysis in memory")
            
        except Exception as e:
            logger.warning(f"Failed to store analysis in memory: {e}")
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent breakthrough analyses from memory."""
        try:
            # Search memory for previous analyses
            analyses = self.memory.search_papers(
                query="breakthrough analysis",
                k=limit,
                filter_dict={"document_type": "breakthrough_analysis"}
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