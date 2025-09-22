"""Business analysis tools for breakthrough potential evaluation."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from ..config import Config

logger = logging.getLogger(__name__)

# Initialize LLM for business analysis
business_llm = ChatOpenAI(
    model_name=Config.DEFAULT_MODEL,
    temperature=0.3,  # Lower temperature for more analytical responses
    api_key=Config.OPENAI_API_KEY
)

@tool
def assess_technical_feasibility(research_findings: str) -> str:
    """
    Assess the technical feasibility and scalability of research findings.
    
    Args:
        research_findings: Description of the research findings to evaluate
        
    Returns:
        Structured technical feasibility assessment
    """
    try:
        feasibility_prompt = PromptTemplate(
            input_variables=["findings"],
            template="""
            You are a technical expert evaluating the feasibility of research findings for commercial implementation.
            
            Research Findings: {findings}
            
            Provide a structured technical feasibility assessment:
            
            ## Technical Feasibility Score: [1-10]
            
            ## Key Technical Factors:
            - **Mechanism Validation**: How well-established is the underlying science?
            - **Scalability**: Can this be manufactured/implemented at scale?
            - **Stability**: Are there stability or degradation concerns?
            - **Safety Profile**: What are the known/potential safety issues?
            
            ## Implementation Challenges:
            1. [Challenge 1]
            2. [Challenge 2]
            3. [Challenge 3]
            
            ## Technical Readiness Level: [1-9 TRL scale]
            
            ## Recommendations:
            - [Specific technical validation steps needed]
            """
        )
        
        chain = feasibility_prompt | business_llm
        result = chain.invoke({"findings": research_findings})
        
        logger.info("Completed technical feasibility assessment")
        return result.content
        
    except Exception as e:
        logger.error(f"Error in technical feasibility assessment: {e}")
        return f"Technical feasibility assessment error: {str(e)}"

@tool
def analyze_market_viability(research_findings: str) -> str:
    """
    Analyze commercial market viability and potential for research findings.
    
    Args:
        research_findings: Description of the research findings to evaluate
        
    Returns:
        Structured market viability analysis
    """
    try:
        market_prompt = PromptTemplate(
            input_variables=["findings"],
            template="""
            You are a market analyst evaluating the commercial potential of research findings in the climate technology sector.
            
            Research Findings: {findings}
            
            Provide a structured market viability assessment:
            
            ## Market Viability Score: [1-10]
            
            ## Market Analysis:
            - **Addressable Market**: Size and characteristics of target market
            - **Customer Need**: How urgent/important is this solution?
            - **Pricing Potential**: What could customers pay for this?
            - **Market Timing**: Is the market ready for this solution?
            
            ## Competitive Landscape:
            - **Existing Solutions**: What alternatives currently exist?
            - **Competitive Advantage**: What makes this unique/better?
            - **Market Position**: First mover or fast follower opportunity?
            
            ## Revenue Potential:
            - **Market Size**: Estimated total addressable market
            - **Market Share**: Realistic capture potential
            - **Revenue Timeline**: When could meaningful revenue start?
            
            ## Market Risks:
            1. [Risk 1]
            2. [Risk 2]
            3. [Risk 3]
            """
        )
        
        chain = market_prompt | business_llm
        result = chain.invoke({"findings": research_findings})
        
        logger.info("Completed market viability analysis")
        return result.content
        
    except Exception as e:
        logger.error(f"Error in market viability analysis: {e}")
        return f"Market viability analysis error: {str(e)}"

@tool
def evaluate_competitive_landscape(research_findings: str) -> str:
    """
    Evaluate the competitive landscape and positioning for research findings.
    
    Args:
        research_findings: Description of the research findings to evaluate
        
    Returns:
        Competitive landscape analysis
    """
    try:
        competitive_prompt = PromptTemplate(
            input_variables=["findings"],
            template="""
            You are a competitive intelligence analyst evaluating research findings in the context of existing market players.
            
            Research Findings: {findings}
            
            Provide competitive landscape evaluation:
            
            ## Competitive Position Score: [1-10]
            
            ## Existing Players Analysis:
            - **Direct Competitors**: Who offers similar solutions?
            - **Indirect Competitors**: Alternative approaches to the same problem?
            - **Technology Leaders**: Who dominates this space currently?
            
            ## Competitive Advantages:
            - **Unique Differentiators**: What makes this approach superior?
            - **Barriers to Entry**: What protects this advantage?
            - **Speed to Market**: First mover vs. fast follower potential?
            
            ## Competitive Threats:
            - **Technology Disruption**: Could existing players adapt quickly?
            - **Resource Advantages**: Do competitors have superior resources?
            - **Market Access**: Do competitors have better market access?
            
            ## Strategic Positioning:
            - **Blue Ocean vs. Red Ocean**: New market or existing competition?
            - **Partnership Opportunities**: Potential allies or acquisition targets?
            - **IP Landscape**: Patent considerations and freedom to operate?
            """
        )
        
        chain = competitive_prompt | business_llm
        result = chain.invoke({"findings": research_findings})
        
        logger.info("Completed competitive landscape evaluation")
        return result.content
        
    except Exception as e:
        logger.error(f"Error in competitive landscape evaluation: {e}")
        return f"Competitive landscape evaluation error: {str(e)}"

@tool
def assess_regulatory_pathway(research_findings: str) -> str:
    """
    Assess regulatory approval pathway and timeline for research findings.
    
    Args:
        research_findings: Description of the research findings to evaluate
        
    Returns:
        Regulatory pathway assessment
    """
    try:
        regulatory_prompt = PromptTemplate(
            input_variables=["findings"],
            template="""
            You are a regulatory affairs expert evaluating the approval pathway for research findings.
            
            Research Findings: {findings}
            
            Provide regulatory pathway assessment:
            
            ## Regulatory Complexity Score: [1-10]
            
            ## Regulatory Framework:
            - **Primary Regulators**: Which agencies would oversee approval?
            - **Approval Type**: Food additive, drug, GRAS, or other classification?
            - **Precedent Cases**: Similar products that have been approved?
            
            ## Approval Timeline:
            - **Pre-submission**: Preparation and data generation phase
            - **Submission to Approval**: Regulatory review timeline
            - **Total Timeline**: Realistic time from today to market
            
            ## Required Studies:
            - **Safety Studies**: Toxicology, animal studies needed
            - **Efficacy Studies**: Proof of performance requirements
            - **Environmental Impact**: Environmental safety assessments
            
            ## Regulatory Risks:
            1. [Risk 1 and mitigation]
            2. [Risk 2 and mitigation]
            3. [Risk 3 and mitigation]
            
            ## Cost Estimates:
            - **Study Costs**: Estimated cost for required studies
            - **Regulatory Fees**: Filing and review fees
            - **Total Investment**: Complete regulatory approval cost
            """
        )
        
        chain = regulatory_prompt | business_llm
        result = chain.invoke({"findings": research_findings})
        
        logger.info("Completed regulatory pathway assessment")
        return result.content
        
    except Exception as e:
        logger.error(f"Error in regulatory pathway assessment: {e}")
        return f"Regulatory pathway assessment error: {str(e)}"

@tool
def generate_investment_recommendation(
    research_findings: str, 
    technical_assessment: str = "",
    market_analysis: str = "",
    competitive_analysis: str = "",
    regulatory_assessment: str = ""
) -> str:
    """
    Generate final investment recommendation based on all assessments.
    
    Args:
        research_findings: Original research findings
        technical_assessment: Technical feasibility results
        market_analysis: Market viability results
        competitive_analysis: Competitive landscape results
        regulatory_assessment: Regulatory pathway results
        
    Returns:
        Final investment recommendation
    """
    try:
        investment_prompt = PromptTemplate(
            input_variables=[
                "findings", "technical", "market", "competitive", "regulatory"
            ],
            template="""
            You are a senior investment analyst providing final recommendations on breakthrough research opportunities.
            
            Research Findings: {findings}
            
            Supporting Analyses:
            Technical Assessment: {technical}
            Market Analysis: {market}
            Competitive Analysis: {competitive}
            Regulatory Assessment: {regulatory}
            
            Provide final investment recommendation:
            
            ## Overall Breakthrough Score: [1-10]
            
            ## Investment Recommendation: [PROCEED / CAUTIOUS / PASS]
            
            ## Executive Summary:
            [2-3 sentence summary of opportunity and recommendation]
            
            ## Key Success Factors:
            1. [Critical success factor 1]
            2. [Critical success factor 2]
            3. [Critical success factor 3]
            
            ## Risk-Adjusted Timeline:
            - **Phase 1**: [Immediate next steps - 6 months]
            - **Phase 2**: [Development phase - 12-18 months]
            - **Phase 3**: [Market entry phase - 2-3 years]
            
            ## Investment Requirements:
            - **Initial Investment**: $XXX,000 for validation phase
            - **Development Investment**: $X,XXX,000 for full development
            - **Total to Market**: $X,XXX,000 complete investment
            
            ## Strategic Value:
            - **Alignment with Agteria Goals**: How well does this fit our mission?
            - **Portfolio Impact**: How does this strengthen our overall portfolio?
            - **Strategic Options**: What future opportunities does this create?
            
            ## Immediate Actions Required:
            1. [Specific action 1]
            2. [Specific action 2]
            3. [Specific action 3]
            """
        )
        
        chain = investment_prompt | business_llm
        result = chain.invoke({
            "findings": research_findings,
            "technical": technical_assessment,
            "market": market_analysis,
            "competitive": competitive_analysis,
            "regulatory": regulatory_assessment
        })
        
        logger.info("Generated final investment recommendation")
        return result.content
        
    except Exception as e:
        logger.error(f"Error generating investment recommendation: {e}")
        return f"Investment recommendation error: {str(e)}"

def create_langchain_business_tools() -> List:
    """Create list of LangChain business analysis tools."""
    return [
        assess_technical_feasibility,
        analyze_market_viability,
        evaluate_competitive_landscape,
        assess_regulatory_pathway,
        generate_investment_recommendation
    ]