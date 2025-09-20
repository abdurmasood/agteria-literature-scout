"""Analysis tools for scientific paper processing and insight extraction."""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from ..config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperAnalyzer:
    """Advanced paper analysis and content extraction."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Analysis prompt templates
        self.analysis_prompt = PromptTemplate(
            input_variables=["paper_content", "focus_area"],
            template="""
            Analyze this scientific paper with focus on {focus_area}.
            
            Paper Content:
            {paper_content}
            
            Please provide a structured analysis including:
            
            1. MAIN OBJECTIVE: What is the primary research goal?
            
            2. METHODOLOGY: What methods/techniques were used?
            
            3. KEY FINDINGS: What are the most important results?
            
            4. NOVEL CONTRIBUTIONS: What's new or innovative?
            
            5. PRACTICAL APPLICATIONS: How could this be applied to methane reduction in cattle?
            
            6. LIMITATIONS: What are the study limitations?
            
            7. FUTURE DIRECTIONS: What follow-up research is suggested?
            
            8. CONFIDENCE SCORE: Rate relevance to methane reduction (0-10)
            
            Format as clear sections with bullet points.
            """
        )
        
        self.molecule_extraction_prompt = PromptTemplate(
            input_variables=["paper_content"],
            template="""
            Extract molecular information from this paper:
            
            {paper_content}
            
            Identify and list:
            
            1. CHEMICAL COMPOUNDS: All named molecules, chemicals, drugs
            2. MOLECULAR TARGETS: Enzymes, proteins, receptors mentioned
            3. CONCENTRATIONS: Any dosages or concentrations mentioned
            4. MECHANISMS: How the molecules work (if described)
            5. EFFICACY DATA: Any quantitative results (% reduction, IC50, etc.)
            
            Format as structured lists with molecular names and any associated data.
            """
        )
        
        self.hypothesis_prompt = PromptTemplate(
            input_variables=["analysis_results", "domain"],
            template="""
            Based on these research findings, generate novel hypotheses for methane reduction in cattle:
            
            Research Findings:
            {analysis_results}
            
            Domain Context: {domain}
            
            Generate 3-5 specific, testable hypotheses that:
            1. Build on the research findings
            2. Are relevant to cattle methane reduction
            3. Could be practically implemented
            4. Suggest specific next steps
            
            For each hypothesis, include:
            - Hypothesis statement
            - Scientific rationale
            - Proposed experiment
            - Expected outcome
            - Feasibility assessment (High/Medium/Low)
            
            Be creative but scientifically grounded.
            """
        )
    
    def analyze_paper(self, paper_content: str, focus_area: str = "methane reduction") -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a scientific paper.
        
        Args:
            paper_content: Full text or abstract of the paper
            focus_area: Specific research focus for analysis
        
        Returns:
            Dictionary containing structured analysis results
        """
        try:
            # Truncate content if too long
            if len(paper_content) > Config.MAX_DOC_LENGTH:
                paper_content = paper_content[:Config.MAX_DOC_LENGTH] + "... [truncated]"
            
            # Run analysis
            analysis_chain = self.analysis_prompt | self.llm
            analysis_result = analysis_chain.invoke({
                "paper_content": paper_content,
                "focus_area": focus_area
            })
            
            # Extract structured information
            structured_analysis = self._parse_analysis_result(analysis_result.content)
            
            # Add metadata
            structured_analysis.update({
                "analyzed_at": datetime.now().isoformat(),
                "focus_area": focus_area,
                "content_length": len(paper_content),
                "analysis_model": Config.DEFAULT_MODEL
            })
            
            logger.info(f"Successfully analyzed paper for {focus_area}")
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Paper analysis error: {e}")
            return {"error": str(e), "analyzed_at": datetime.now().isoformat()}
    
    def extract_molecules(self, paper_content: str) -> Dict[str, List[str]]:
        """
        Extract molecular and chemical information from paper.
        
        Args:
            paper_content: Paper text content
        
        Returns:
            Dictionary with categorized molecular information
        """
        try:
            molecule_chain = self.molecule_extraction_prompt | self.llm
            result = molecule_chain.invoke({"paper_content": paper_content})
            
            molecules = self._parse_molecule_result(result.content)
            
            logger.info(f"Extracted molecular information: {len(molecules)} categories")
            return molecules
            
        except Exception as e:
            logger.error(f"Molecule extraction error: {e}")
            return {"error": [str(e)]}
    
    def generate_hypotheses(self, analysis_results: str, domain: str = "agriculture") -> List[Dict[str, Any]]:
        """
        Generate research hypotheses based on analysis results.
        
        Args:
            analysis_results: Results from paper analysis
            domain: Research domain context
        
        Returns:
            List of hypothesis dictionaries
        """
        try:
            hypothesis_chain = self.hypothesis_prompt | self.llm
            result = hypothesis_chain.invoke({
                "analysis_results": analysis_results,
                "domain": domain
            })
            
            hypotheses = self._parse_hypothesis_result(result.content)
            
            logger.info(f"Generated {len(hypotheses)} research hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Hypothesis generation error: {e}")
            return [{"error": str(e)}]
    
    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured analysis result into dictionary."""
        sections = {
            "main_objective": "",
            "methodology": "",
            "key_findings": "",
            "novel_contributions": "",
            "practical_applications": "",
            "limitations": "",
            "future_directions": "",
            "confidence_score": 0
        }
        
        try:
            # Extract sections using regex patterns
            patterns = {
                "main_objective": r"1\.\s*MAIN OBJECTIVE:\s*(.+?)(?=2\.|$)",
                "methodology": r"2\.\s*METHODOLOGY:\s*(.+?)(?=3\.|$)",
                "key_findings": r"3\.\s*KEY FINDINGS:\s*(.+?)(?=4\.|$)",
                "novel_contributions": r"4\.\s*NOVEL CONTRIBUTIONS:\s*(.+?)(?=5\.|$)",
                "practical_applications": r"5\.\s*PRACTICAL APPLICATIONS:\s*(.+?)(?=6\.|$)",
                "limitations": r"6\.\s*LIMITATIONS:\s*(.+?)(?=7\.|$)",
                "future_directions": r"7\.\s*FUTURE DIRECTIONS:\s*(.+?)(?=8\.|$)",
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[key] = match.group(1).strip()
            
            # Extract confidence score
            score_match = re.search(r"CONFIDENCE SCORE:\s*.*?(\d+)", analysis_text, re.IGNORECASE)
            if score_match:
                sections["confidence_score"] = int(score_match.group(1))
                
        except Exception as e:
            logger.warning(f"Error parsing analysis result: {e}")
            sections["raw_analysis"] = analysis_text
        
        return sections
    
    def _parse_molecule_result(self, molecule_text: str) -> Dict[str, List[str]]:
        """Parse molecular extraction result into dictionary."""
        molecules = {
            "chemical_compounds": [],
            "molecular_targets": [],
            "concentrations": [],
            "mechanisms": [],
            "efficacy_data": []
        }
        
        try:
            # Extract sections
            patterns = {
                "chemical_compounds": r"1\.\s*CHEMICAL COMPOUNDS:\s*(.+?)(?=2\.|$)",
                "molecular_targets": r"2\.\s*MOLECULAR TARGETS:\s*(.+?)(?=3\.|$)",
                "concentrations": r"3\.\s*CONCENTRATIONS:\s*(.+?)(?=4\.|$)",
                "mechanisms": r"4\.\s*MECHANISMS:\s*(.+?)(?=5\.|$)",
                "efficacy_data": r"5\.\s*EFFICACY DATA:\s*(.+?)$"
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, molecule_text, re.DOTALL | re.IGNORECASE)
                if match:
                    # Split by lines and clean up
                    items = [line.strip("- ").strip() for line in match.group(1).split('\n') 
                            if line.strip() and not line.strip().startswith('-')]
                    molecules[key] = [item for item in items if item]
                    
        except Exception as e:
            logger.warning(f"Error parsing molecule result: {e}")
            molecules["raw_extraction"] = molecule_text
        
        return molecules
    
    def _parse_hypothesis_result(self, hypothesis_text: str) -> List[Dict[str, Any]]:
        """Parse hypothesis generation result into list of dictionaries."""
        hypotheses = []
        
        try:
            # Split by hypothesis indicators
            hypothesis_sections = re.split(r'(?=Hypothesis \d+|^\d+\.)', hypothesis_text)
            
            for section in hypothesis_sections:
                if not section.strip():
                    continue
                
                hypothesis = {}
                
                # Extract components
                statement_match = re.search(r"Hypothesis statement:\s*(.+?)(?=Scientific rationale:|$)", section, re.DOTALL | re.IGNORECASE)
                if statement_match:
                    hypothesis["statement"] = statement_match.group(1).strip()
                
                rationale_match = re.search(r"Scientific rationale:\s*(.+?)(?=Proposed experiment:|$)", section, re.DOTALL | re.IGNORECASE)
                if rationale_match:
                    hypothesis["rationale"] = rationale_match.group(1).strip()
                
                experiment_match = re.search(r"Proposed experiment:\s*(.+?)(?=Expected outcome:|$)", section, re.DOTALL | re.IGNORECASE)
                if experiment_match:
                    hypothesis["experiment"] = experiment_match.group(1).strip()
                
                outcome_match = re.search(r"Expected outcome:\s*(.+?)(?=Feasibility assessment:|$)", section, re.DOTALL | re.IGNORECASE)
                if outcome_match:
                    hypothesis["expected_outcome"] = outcome_match.group(1).strip()
                
                feasibility_match = re.search(r"Feasibility assessment:\s*(\w+)", section, re.IGNORECASE)
                if feasibility_match:
                    hypothesis["feasibility"] = feasibility_match.group(1).strip()
                
                if hypothesis:
                    hypothesis["generated_at"] = datetime.now().isoformat()
                    hypotheses.append(hypothesis)
                    
        except Exception as e:
            logger.warning(f"Error parsing hypothesis result: {e}")
            return [{"raw_hypotheses": hypothesis_text, "error": str(e)}]
        
        return hypotheses

class RelevanceScorer:
    """Score papers for relevance to Agteria's research focus."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=0.3,  # Lower temperature for consistent scoring
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.scoring_prompt = PromptTemplate(
            input_variables=["paper_title", "paper_abstract", "focus_keywords"],
            template="""
            Score this paper's relevance to methane reduction in cattle/livestock (0-10 scale):
            
            Title: {paper_title}
            Abstract: {paper_abstract}
            
            Focus Keywords: {focus_keywords}
            
            Scoring Criteria:
            - 9-10: Directly about cattle methane reduction or highly applicable methods
            - 7-8: Related to methane reduction in ruminants or relevant techniques
            - 5-6: General methane research or livestock studies that could be relevant
            - 3-4: Tangentially related (enzyme inhibition, fermentation, etc.)
            - 1-2: Minimal relevance
            - 0: Not relevant
            
            Provide:
            1. RELEVANCE SCORE: [0-10]
            2. REASONING: Brief explanation for the score
            3. KEY APPLICATIONS: How this could apply to Agteria's research
            
            Be concise but thorough.
            """
        )
    
    def score_paper(self, paper_title: str, paper_abstract: str) -> Dict[str, Any]:
        """
        Score a paper's relevance to Agteria's research.
        
        Args:
            paper_title: Title of the paper
            paper_abstract: Abstract or summary
        
        Returns:
            Dictionary with score and reasoning
        """
        try:
            from ..config import AGTERIA_KEYWORDS
            
            scoring_chain = self.scoring_prompt | self.llm
            result = scoring_chain.invoke({
                "paper_title": paper_title,
                "paper_abstract": paper_abstract,
                "focus_keywords": ", ".join(AGTERIA_KEYWORDS)
            })
            
            score_info = self._parse_score_result(result.content)
            score_info["scored_at"] = datetime.now().isoformat()
            
            return score_info
            
        except Exception as e:
            logger.error(f"Relevance scoring error: {e}")
            return {"relevance_score": 0, "error": str(e)}
    
    def _parse_score_result(self, score_text: str) -> Dict[str, Any]:
        """Parse scoring result into structured format."""
        result = {
            "relevance_score": 0,
            "reasoning": "",
            "key_applications": ""
        }
        
        try:
            # Extract score
            score_match = re.search(r"RELEVANCE SCORE:\s*(\d+)", score_text, re.IGNORECASE)
            if score_match:
                result["relevance_score"] = int(score_match.group(1))
            
            # Extract reasoning
            reasoning_match = re.search(r"REASONING:\s*(.+?)(?=KEY APPLICATIONS:|$)", score_text, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract applications
            apps_match = re.search(r"KEY APPLICATIONS:\s*(.+?)$", score_text, re.DOTALL | re.IGNORECASE)
            if apps_match:
                result["key_applications"] = apps_match.group(1).strip()
                
        except Exception as e:
            logger.warning(f"Error parsing score result: {e}")
            result["raw_score"] = score_text
        
        return result

# Create LangChain tools
def create_langchain_analysis_tools() -> List[Tool]:
    """Create LangChain-compatible analysis tools."""
    
    analyzer = PaperAnalyzer()
    scorer = RelevanceScorer()
    
    def analyze_paper_func(paper_content: str) -> str:
        """Analyze paper content and return structured results."""
        analysis = analyzer.analyze_paper(paper_content)
        
        if "error" in analysis:
            return f"Analysis error: {analysis['error']}"
        
        # Format results nicely
        formatted = f"""
Paper Analysis Results:

🎯 MAIN OBJECTIVE:
{analysis.get('main_objective', 'N/A')}

🔬 METHODOLOGY:
{analysis.get('methodology', 'N/A')}

✨ KEY FINDINGS:
{analysis.get('key_findings', 'N/A')}

🆕 NOVEL CONTRIBUTIONS:
{analysis.get('novel_contributions', 'N/A')}

🐄 PRACTICAL APPLICATIONS:
{analysis.get('practical_applications', 'N/A')}

⚠️ LIMITATIONS:
{analysis.get('limitations', 'N/A')}

🔮 FUTURE DIRECTIONS:
{analysis.get('future_directions', 'N/A')}

📊 CONFIDENCE SCORE: {analysis.get('confidence_score', 0)}/10
"""
        return formatted
    
    def extract_molecules_func(paper_content: str) -> str:
        """Extract molecular information from paper."""
        molecules = analyzer.extract_molecules(paper_content)
        
        if "error" in molecules:
            return f"Extraction error: {molecules['error'][0]}"
        
        formatted = "Molecular Information Extracted:\n\n"
        
        for category, items in molecules.items():
            if items and category != "raw_extraction":
                formatted += f"🧪 {category.replace('_', ' ').title()}:\n"
                for item in items:
                    formatted += f"  • {item}\n"
                formatted += "\n"
        
        return formatted
    
    def score_relevance_func(paper_info: str) -> str:
        """Score paper relevance to Agteria's research."""
        # Extract title and abstract from input
        lines = paper_info.split('\n')
        title = lines[0] if lines else ""
        abstract = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        
        score_info = scorer.score_paper(title, abstract)
        
        if "error" in score_info:
            return f"Scoring error: {score_info['error']}"
        
        return f"""
Relevance Score: {score_info['relevance_score']}/10

Reasoning: {score_info.get('reasoning', 'N/A')}

Key Applications: {score_info.get('key_applications', 'N/A')}
"""
    
    def generate_hypotheses_func(analysis_results: str) -> str:
        """Generate research hypotheses from analysis."""
        hypotheses = analyzer.generate_hypotheses(analysis_results)
        
        if not hypotheses or "error" in hypotheses[0]:
            return "Could not generate hypotheses from the provided analysis."
        
        formatted = "Generated Research Hypotheses:\n\n"
        
        for i, hyp in enumerate(hypotheses, 1):
            if "statement" in hyp:
                formatted += f"💡 HYPOTHESIS {i}:\n"
                formatted += f"Statement: {hyp.get('statement', 'N/A')}\n"
                formatted += f"Rationale: {hyp.get('rationale', 'N/A')}\n"
                formatted += f"Proposed Experiment: {hyp.get('experiment', 'N/A')}\n"
                formatted += f"Expected Outcome: {hyp.get('expected_outcome', 'N/A')}\n"
                formatted += f"Feasibility: {hyp.get('feasibility', 'N/A')}\n\n"
        
        return formatted
    
    return [
        Tool(
            name="analyze_paper",
            description="Analyze scientific paper content to extract key findings, methodologies, and applications to methane reduction research. Input should be paper text or abstract.",
            func=analyze_paper_func
        ),
        Tool(
            name="extract_molecules",
            description="Extract molecular and chemical information from scientific papers including compounds, targets, concentrations, and mechanisms.",
            func=extract_molecules_func
        ),
        Tool(
            name="score_relevance", 
            description="Score a paper's relevance to Agteria's methane reduction research (0-10). Input format: 'Title\\nAbstract'.",
            func=score_relevance_func
        ),
        Tool(
            name="generate_hypotheses",
            description="Generate novel research hypotheses based on paper analysis results. Use after analyzing papers to brainstorm new research directions.",
            func=generate_hypotheses_func
        )
    ]