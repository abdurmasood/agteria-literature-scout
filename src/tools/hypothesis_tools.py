"""Hypothesis generation tools for discovering novel research directions."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from ..config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HypothesisGenerator:
    """Advanced hypothesis generation using cross-domain insights."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.HYPOTHESIS_TEMPERATURE,  # Optimized temperature for hypothesis generation
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.cross_domain_prompt = PromptTemplate(
            input_variables=["domain_1", "findings_1", "domain_2", "findings_2", "target_application"],
            template="""
            Generate novel hypotheses by combining insights from two different research domains:
            
            DOMAIN 1: {domain_1}
            Key Findings: {findings_1}
            
            DOMAIN 2: {domain_2}  
            Key Findings: {findings_2}
            
            TARGET APPLICATION: {target_application}
            
            Generate 8-10 innovative hypotheses that combine concepts from both domains.
            
            For each hypothesis:
            1. HYPOTHESIS STATEMENT: Clear, testable hypothesis
            2. CROSS-DOMAIN CONNECTION: How the two domains relate
            3. SCIENTIFIC RATIONALE: Why this combination could work
            4. EXPERIMENTAL APPROACH: How to test this hypothesis
            5. EXPECTED IMPACT: Potential benefits if successful
            6. RISK ASSESSMENT: Potential challenges or limitations
            7. NOVELTY SCORE: Rate uniqueness (1-10)
            
            Focus on creative but scientifically plausible connections.
            """
        )
        
        self.analogy_prompt = PromptTemplate(
            input_variables=["source_mechanism", "source_domain", "target_domain", "target_challenge"],
            template="""
            Use analogical reasoning to generate research hypotheses:
            
            SOURCE MECHANISM: {source_mechanism}
            SOURCE DOMAIN: {source_domain}
            
            TARGET DOMAIN: {target_domain}
            TARGET CHALLENGE: {target_challenge}
            
            Generate hypotheses by adapting the source mechanism to the target domain.
            
            Think through:
            1. What are the key principles in the source mechanism?
            2. What analogous components exist in the target domain?
            3. How could the mechanism be adapted or modified?
            4. What would be the predicted outcomes?
            
            Provide 6-8 hypotheses with:
            - Analogy explanation
            - Adapted mechanism description  
            - Predicted outcomes
            - Testing strategy
            - Feasibility assessment
            """
        )
        
        self.gap_analysis_prompt = PromptTemplate(
            input_variables=["research_area", "existing_approaches", "limitations"],
            template="""
            Identify research gaps and generate hypotheses to fill them:
            
            RESEARCH AREA: {research_area}
            EXISTING APPROACHES: {existing_approaches}
            CURRENT LIMITATIONS: {limitations}
            
            Analyze what's missing and generate hypotheses for:
            
            1. UNEXPLORED MECHANISMS: What biological/chemical pathways haven't been investigated?
            2. OVERLOOKED COMBINATIONS: What approaches could be combined for synergistic effects?
            3. ALTERNATIVE TARGETS: What different molecular targets could be pursued?
            4. NOVEL DELIVERY METHODS: How could existing solutions be delivered more effectively?
            5. SCALING SOLUTIONS: How could lab findings be adapted for real-world application?
            
            For each gap, provide:
            - Gap description
            - Why it's been overlooked
            - Hypothesis to address it
            - Research strategy
            - Expected breakthrough potential
            """
        )
        
        self.contradiction_prompt = PromptTemplate(
            input_variables=["conflicting_findings", "studies"],
            template="""
            Generate hypotheses to resolve contradictory research findings:
            
            CONFLICTING FINDINGS: {conflicting_findings}
            STUDIES INVOLVED: {studies}
            
            Possible explanations for contradictions:
            1. Different experimental conditions
            2. Species/strain differences  
            3. Dosage/concentration effects
            4. Timing factors
            5. Environmental variables
            6. Measurement methods
            7. Hidden variables
            
            Generate hypotheses that could:
            - Reconcile the contradictory findings
            - Explain why different results occurred
            - Design experiments to resolve the contradiction
            - Identify the true underlying mechanism
            
            For each hypothesis:
            - Explanation of contradiction
            - Proposed resolution
            - Experimental design to test
            - Predicted outcome
            """
        )
    
    def generate_cross_domain_hypotheses(self, domain_1: str, findings_1: str, 
                                       domain_2: str, findings_2: str,
                                       target_application: str = "methane reduction in cattle") -> List[Dict[str, Any]]:
        """
        Generate hypotheses combining insights from two different domains.
        
        Args:
            domain_1: First research domain
            findings_1: Key findings from domain 1
            domain_2: Second research domain
            findings_2: Key findings from domain 2
            target_application: Target application area
        
        Returns:
            List of cross-domain hypothesis dictionaries
        """
        try:
            chain = self.cross_domain_prompt | self.llm
            result = chain.invoke({
                "domain_1": domain_1,
                "findings_1": findings_1,
                "domain_2": domain_2, 
                "findings_2": findings_2,
                "target_application": target_application
            })
            
            hypotheses = self._parse_cross_domain_result(result.content)
            
            # Add metadata
            for hyp in hypotheses:
                hyp.update({
                    "generation_type": "cross_domain",
                    "source_domains": [domain_1, domain_2],
                    "generated_at": datetime.now().isoformat()
                })
            
            logger.info(f"Generated {len(hypotheses)} cross-domain hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Cross-domain hypothesis generation error: {e}")
            return [{"error": str(e)}]
    
    def generate_analogical_hypotheses(self, source_mechanism: str, source_domain: str,
                                     target_domain: str = "cattle methane reduction",
                                     target_challenge: str = "inhibiting methanogenesis") -> List[Dict[str, Any]]:
        """
        Generate hypotheses using analogical reasoning.
        
        Args:
            source_mechanism: Mechanism from source domain
            source_domain: Source research domain
            target_domain: Target application domain
            target_challenge: Specific challenge to address
        
        Returns:
            List of analogical hypothesis dictionaries
        """
        try:
            chain = self.analogy_prompt | self.llm
            result = chain.invoke({
                "source_mechanism": source_mechanism,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "target_challenge": target_challenge
            })
            
            hypotheses = self._parse_analogical_result(result.content)
            
            # Add metadata
            for hyp in hypotheses:
                hyp.update({
                    "generation_type": "analogical",
                    "source_mechanism": source_mechanism,
                    "source_domain": source_domain,
                    "generated_at": datetime.now().isoformat()
                })
            
            logger.info(f"Generated {len(hypotheses)} analogical hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Analogical hypothesis generation error: {e}")
            return [{"error": str(e)}]
    
    def identify_research_gaps(self, research_area: str, existing_approaches: str,
                             limitations: str) -> List[Dict[str, Any]]:
        """
        Identify research gaps and generate hypotheses to fill them.
        
        Args:
            research_area: Area of research focus
            existing_approaches: Current research approaches
            limitations: Known limitations of existing work
        
        Returns:
            List of gap-filling hypothesis dictionaries
        """
        try:
            chain = self.gap_analysis_prompt | self.llm
            result = chain.invoke({
                "research_area": research_area,
                "existing_approaches": existing_approaches,
                "limitations": limitations
            })
            
            gaps = self._parse_gap_analysis_result(result.content)
            
            # Add metadata
            for gap in gaps:
                gap.update({
                    "generation_type": "gap_analysis",
                    "research_area": research_area,
                    "generated_at": datetime.now().isoformat()
                })
            
            logger.info(f"Identified {len(gaps)} research gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Gap analysis error: {e}")
            return [{"error": str(e)}]
    
    def resolve_contradictions(self, conflicting_findings: str, studies: str) -> List[Dict[str, Any]]:
        """
        Generate hypotheses to resolve contradictory research findings.
        
        Args:
            conflicting_findings: Description of contradictory results
            studies: Information about the conflicting studies
        
        Returns:
            List of contradiction-resolving hypothesis dictionaries
        """
        try:
            chain = self.contradiction_prompt | self.llm
            result = chain.invoke({
                "conflicting_findings": conflicting_findings,
                "studies": studies
            })
            
            hypotheses = self._parse_contradiction_result(result.content)
            
            # Add metadata
            for hyp in hypotheses:
                hyp.update({
                    "generation_type": "contradiction_resolution",
                    "conflicting_findings": conflicting_findings,
                    "generated_at": datetime.now().isoformat()
                })
            
            logger.info(f"Generated {len(hypotheses)} contradiction-resolving hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Contradiction resolution error: {e}")
            return [{"error": str(e)}]
    
    def _parse_cross_domain_result(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse cross-domain hypothesis generation result."""
        hypotheses = []
        
        # Split into individual hypotheses
        hypothesis_sections = result_text.split("HYPOTHESIS STATEMENT:")
        
        for section in hypothesis_sections[1:]:  # Skip first empty section
            hypothesis = {}
            
            # Extract each component
            components = {
                "statement": r"(.+?)(?=CROSS-DOMAIN CONNECTION:|$)",
                "connection": r"CROSS-DOMAIN CONNECTION:\s*(.+?)(?=SCIENTIFIC RATIONALE:|$)",
                "rationale": r"SCIENTIFIC RATIONALE:\s*(.+?)(?=EXPERIMENTAL APPROACH:|$)",
                "experimental_approach": r"EXPERIMENTAL APPROACH:\s*(.+?)(?=EXPECTED IMPACT:|$)",
                "expected_impact": r"EXPECTED IMPACT:\s*(.+?)(?=RISK ASSESSMENT:|$)",
                "risk_assessment": r"RISK ASSESSMENT:\s*(.+?)(?=NOVELTY SCORE:|$)",
                "novelty_score": r"NOVELTY SCORE:\s*(\d+)"
            }
            
            for key, pattern in components.items():
                import re
                match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if key == "novelty_score":
                        hypothesis[key] = int(value) if value.isdigit() else 0
                    else:
                        hypothesis[key] = value
            
            if hypothesis:
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _parse_analogical_result(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse analogical hypothesis generation result."""
        hypotheses = []
        
        # Split by hypothesis indicators
        sections = result_text.split("Hypothesis")
        
        for section in sections[1:]:  # Skip first section
            hypothesis = {}
            
            # Extract components
            lines = section.split('\n')
            current_field = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("- Analogy explanation:"):
                    current_field = "analogy_explanation"
                    current_content = [line.replace("- Analogy explanation:", "").strip()]
                elif line.startswith("- Adapted mechanism:"):
                    current_field = "adapted_mechanism"
                    current_content = [line.replace("- Adapted mechanism:", "").strip()]
                elif line.startswith("- Predicted outcomes:"):
                    current_field = "predicted_outcomes"
                    current_content = [line.replace("- Predicted outcomes:", "").strip()]
                elif line.startswith("- Testing strategy:"):
                    current_field = "testing_strategy"
                    current_content = [line.replace("- Testing strategy:", "").strip()]
                elif line.startswith("- Feasibility assessment:"):
                    current_field = "feasibility_assessment"
                    current_content = [line.replace("- Feasibility assessment:", "").strip()]
                elif current_field:
                    current_content.append(line)
                
                if current_field and (line.startswith("- ") or not current_content[-1]):
                    if current_field in hypothesis:
                        hypothesis[current_field] += " " + " ".join(current_content)
                    else:
                        hypothesis[current_field] = " ".join(current_content)
            
            if hypothesis:
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _parse_gap_analysis_result(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse gap analysis result."""
        gaps = []
        
        # Extract different types of gaps
        gap_types = [
            "UNEXPLORED MECHANISMS",
            "OVERLOOKED COMBINATIONS",
            "ALTERNATIVE TARGETS",
            "NOVEL DELIVERY METHODS",
            "SCALING SOLUTIONS"
        ]
        
        for gap_type in gap_types:
            import re
            pattern = f"{gap_type}:\s*(.+?)(?={'|'.join(gap_types[gap_types.index(gap_type)+1:])}|$)"
            match = re.search(pattern, result_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                gap_content = match.group(1).strip()
                
                # Parse individual gaps within this type
                gap_items = gap_content.split("- Gap description:")
                
                for item in gap_items[1:]:  # Skip first empty item
                    gap = {"gap_type": gap_type.lower().replace(" ", "_")}
                    
                    # Extract components
                    components = {
                        "description": r"(.+?)(?=Why it's been overlooked:|$)",
                        "why_overlooked": r"Why it's been overlooked:\s*(.+?)(?=Hypothesis to address:|$)",
                        "hypothesis": r"Hypothesis to address:\s*(.+?)(?=Research strategy:|$)",
                        "research_strategy": r"Research strategy:\s*(.+?)(?=Expected breakthrough potential:|$)",
                        "breakthrough_potential": r"Expected breakthrough potential:\s*(.+?)$"
                    }
                    
                    for key, pattern in components.items():
                        match = re.search(pattern, item, re.DOTALL | re.IGNORECASE)
                        if match:
                            gap[key] = match.group(1).strip()
                    
                    if gap.get("description"):
                        gaps.append(gap)
        
        return gaps
    
    def _parse_contradiction_result(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse contradiction resolution result."""
        hypotheses = []
        
        # Split by hypothesis indicators
        sections = result_text.split("Hypothesis")
        
        for section in sections[1:]:  # Skip first section
            hypothesis = {}
            
            # Extract components
            components = {
                "explanation": r"Explanation of contradiction:\s*(.+?)(?=Proposed resolution:|$)",
                "resolution": r"Proposed resolution:\s*(.+?)(?=Experimental design:|$)",
                "experimental_design": r"Experimental design:\s*(.+?)(?=Predicted outcome:|$)",
                "predicted_outcome": r"Predicted outcome:\s*(.+?)$"
            }
            
            import re
            for key, pattern in components.items():
                match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
                if match:
                    hypothesis[key] = match.group(1).strip()
            
            if hypothesis:
                hypotheses.append(hypothesis)
        
        return hypotheses

class IdeationEngine:
    """Creative ideation engine for breakthrough research concepts."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.HYPOTHESIS_TEMPERATURE + 0.05,  # Slightly higher than hypothesis generation for maximum creativity
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def brainstorm_wild_ideas(self, research_challenge: str, inspiration_sources: List[str]) -> List[Dict[str, Any]]:
        """
        Generate unconventional, creative research ideas.
        
        Args:
            research_challenge: The core challenge to address
            inspiration_sources: List of diverse inspiration sources
        
        Returns:
            List of creative idea dictionaries
        """
        prompt = f"""
        Brainstorm unconventional, potentially breakthrough ideas for: {research_challenge}
        
        Draw inspiration from: {', '.join(inspiration_sources)}
        
        Think outside the box. Consider:
        - Nature's solutions (biomimicry)
        - Technology from other industries
        - Historical breakthroughs in unrelated fields
        - Science fiction concepts that could become reality
        - Artistic or cultural approaches
        
        Generate 10-15 wildly creative but potentially feasible ideas:
        
        For each idea:
        1. WILD IDEA: The unconventional concept
        2. INSPIRATION SOURCE: What inspired this idea
        3. SCIENTIFIC BASIS: How it could actually work
        4. IMPLEMENTATION PATHWAY: Steps to make it reality
        5. BREAKTHROUGH POTENTIAL: Why this could be revolutionary
        6. FEASIBILITY ASSESSMENT: Realistic evaluation
        
        Be extremely creative while maintaining scientific plausibility.
        """
        
        try:
            result = self.llm.predict(prompt)
            ideas = self._parse_wild_ideas(result)
            
            for idea in ideas:
                idea.update({
                    "generation_type": "creative_ideation",
                    "challenge": research_challenge,
                    "generated_at": datetime.now().isoformat()
                })
            
            logger.info(f"Generated {len(ideas)} wild ideas")
            return ideas
            
        except Exception as e:
            logger.error(f"Creative ideation error: {e}")
            return [{"error": str(e)}]
    
    def _parse_wild_ideas(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse creative ideation result."""
        ideas = []
        
        # Split by idea indicators
        idea_sections = result_text.split("WILD IDEA:")
        
        for section in idea_sections[1:]:  # Skip first empty section
            idea = {}
            
            components = {
                "concept": r"(.+?)(?=INSPIRATION SOURCE:|$)",
                "inspiration": r"INSPIRATION SOURCE:\s*(.+?)(?=SCIENTIFIC BASIS:|$)",
                "scientific_basis": r"SCIENTIFIC BASIS:\s*(.+?)(?=IMPLEMENTATION PATHWAY:|$)",
                "implementation": r"IMPLEMENTATION PATHWAY:\s*(.+?)(?=BREAKTHROUGH POTENTIAL:|$)",
                "breakthrough_potential": r"BREAKTHROUGH POTENTIAL:\s*(.+?)(?=FEASIBILITY ASSESSMENT:|$)",
                "feasibility": r"FEASIBILITY ASSESSMENT:\s*(.+?)$"
            }
            
            import re
            for key, pattern in components.items():
                match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
                if match:
                    idea[key] = match.group(1).strip()
            
            if idea:
                ideas.append(idea)
        
        return ideas

# Create LangChain tools
def create_langchain_hypothesis_tools() -> List[Tool]:
    """Create LangChain-compatible hypothesis generation tools."""
    
    generator = HypothesisGenerator()
    ideation = IdeationEngine()
    
    def cross_domain_func(input_text: str) -> str:
        """Generate cross-domain hypotheses. Input format: 'Domain1: findings1 | Domain2: findings2'"""
        try:
            parts = input_text.split(" | ")
            if len(parts) != 2:
                return "Please use format: 'Domain1: findings1 | Domain2: findings2'"
            
            domain1_part = parts[0].split(": ", 1)
            domain2_part = parts[1].split(": ", 1)
            
            if len(domain1_part) != 2 or len(domain2_part) != 2:
                return "Please use format: 'Domain1: findings1 | Domain2: findings2'"
            
            hypotheses = generator.generate_cross_domain_hypotheses(
                domain1_part[0], domain1_part[1],
                domain2_part[0], domain2_part[1]
            )
            
            if not hypotheses or "error" in hypotheses[0]:
                return "Could not generate cross-domain hypotheses."
            
            formatted = "Cross-Domain Hypotheses:\n\n"
            for i, hyp in enumerate(hypotheses, 1):
                formatted += f"🔬 HYPOTHESIS {i}:\n"
                formatted += f"Statement: {hyp.get('statement', 'N/A')}\n"
                formatted += f"Connection: {hyp.get('connection', 'N/A')}\n"
                formatted += f"Rationale: {hyp.get('rationale', 'N/A')}\n"
                formatted += f"Novelty Score: {hyp.get('novelty_score', 0)}/10\n\n"
            
            return formatted
            
        except Exception as e:
            return f"Error generating cross-domain hypotheses: {e}"
    
    def analogical_func(input_text: str) -> str:
        """Generate analogical hypotheses. Input format: 'source_mechanism | source_domain'"""
        try:
            parts = input_text.split(" | ")
            if len(parts) < 2:
                return "Please use format: 'source_mechanism | source_domain'"
            
            hypotheses = generator.generate_analogical_hypotheses(parts[0], parts[1])
            
            if not hypotheses or "error" in hypotheses[0]:
                return "Could not generate analogical hypotheses."
            
            formatted = "Analogical Hypotheses:\n\n"
            for i, hyp in enumerate(hypotheses, 1):
                formatted += f"🧠 ANALOGY {i}:\n"
                formatted += f"Explanation: {hyp.get('analogy_explanation', 'N/A')}\n"
                formatted += f"Adapted Mechanism: {hyp.get('adapted_mechanism', 'N/A')}\n"
                formatted += f"Predicted Outcomes: {hyp.get('predicted_outcomes', 'N/A')}\n\n"
            
            return formatted
            
        except Exception as e:
            return f"Error generating analogical hypotheses: {e}"
    
    def gap_analysis_func(input_text: str) -> str:
        """Identify research gaps. Input format: 'research_area | existing_approaches | limitations'"""
        try:
            parts = input_text.split(" | ")
            if len(parts) < 3:
                return "Please use format: 'research_area | existing_approaches | limitations'"
            
            gaps = generator.identify_research_gaps(parts[0], parts[1], parts[2])
            
            if not gaps or "error" in gaps[0]:
                return "Could not identify research gaps."
            
            formatted = "Research Gaps Identified:\n\n"
            for i, gap in enumerate(gaps, 1):
                formatted += f"🔍 GAP {i} ({gap.get('gap_type', 'unknown').replace('_', ' ').title()}):\n"
                formatted += f"Description: {gap.get('description', 'N/A')}\n"
                formatted += f"Hypothesis: {gap.get('hypothesis', 'N/A')}\n"
                formatted += f"Strategy: {gap.get('research_strategy', 'N/A')}\n\n"
            
            return formatted
            
        except Exception as e:
            return f"Error analyzing research gaps: {e}"
    
    def creative_ideation_func(research_challenge: str) -> str:
        """Generate creative breakthrough ideas for a research challenge."""
        try:
            inspiration_sources = [
                "nature and biology", "space technology", "art and design",
                "ancient civilizations", "gaming and virtual reality",
                "cooking and food science", "music and acoustics"
            ]
            
            ideas = ideation.brainstorm_wild_ideas(research_challenge, inspiration_sources)
            
            if not ideas or "error" in ideas[0]:
                return "Could not generate creative ideas."
            
            formatted = f"Creative Ideas for: {research_challenge}\n\n"
            for i, idea in enumerate(ideas, 1):
                formatted += f"💡 WILD IDEA {i}:\n"
                formatted += f"Concept: {idea.get('concept', 'N/A')}\n"
                formatted += f"Inspiration: {idea.get('inspiration', 'N/A')}\n"
                formatted += f"Scientific Basis: {idea.get('scientific_basis', 'N/A')}\n"
                formatted += f"Feasibility: {idea.get('feasibility', 'N/A')}\n\n"
            
            return formatted
            
        except Exception as e:
            return f"Error generating creative ideas: {e}"
    
    return [
        Tool(
            name="cross_domain_hypotheses",
            description="Generate hypotheses by combining insights from two different research domains. Use format: 'Domain1: findings1 | Domain2: findings2'",
            func=cross_domain_func
        ),
        Tool(
            name="analogical_hypotheses",
            description="Generate hypotheses using analogical reasoning from other domains. Use format: 'source_mechanism | source_domain'",
            func=analogical_func
        ),
        Tool(
            name="research_gap_analysis",
            description="Identify research gaps and generate hypotheses to fill them. Use format: 'research_area | existing_approaches | limitations'",
            func=gap_analysis_func
        ),
        Tool(
            name="creative_ideation",
            description="Generate wildly creative but scientifically plausible breakthrough ideas for a research challenge.",
            func=creative_ideation_func
        )
    ]