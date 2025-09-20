"""Report generation utilities for Literature Scout findings."""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from ..config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate various types of reports from research findings."""
    
    def __init__(self, reports_dir: Optional[str] = None):
        self.reports_dir = Path(reports_dir or Config.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Report templates
        self.templates = {
            "daily_digest": self._daily_digest_template(),
            "research_summary": self._research_summary_template(),
            "hypothesis_report": self._hypothesis_report_template(),
            "competitive_intelligence": self._competitive_intelligence_template()
        }
    
    def generate_daily_digest(self, scan_results: Dict[str, Any], 
                            output_format: str = "markdown") -> str:
        """
        Generate daily research digest.
        
        Args:
            scan_results: Results from daily research scan
            output_format: Output format ("markdown", "html", "json")
        
        Returns:
            Path to generated report
        """
        try:
            report_data = self._prepare_daily_digest_data(scan_results)
            
            if output_format == "markdown":
                content = self._generate_markdown_digest(report_data)
                filename = f"daily_digest_{datetime.now().strftime('%Y%m%d')}.md"
            elif output_format == "html":
                content = self._generate_html_digest(report_data)
                filename = f"daily_digest_{datetime.now().strftime('%Y%m%d')}.html"
            elif output_format == "json":
                content = json.dumps(report_data, indent=2)
                filename = f"daily_digest_{datetime.now().strftime('%Y%m%d')}.json"
            else:
                raise ValueError(f"Unsupported format: {output_format}")
            
            # Write report
            report_path = self.reports_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated daily digest: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating daily digest: {e}")
            return ""
    
    def generate_research_summary(self, research_results: List[Dict[str, Any]], 
                                topic: str) -> str:
        """
        Generate comprehensive research summary.
        
        Args:
            research_results: List of research results
            topic: Research topic
        
        Returns:
            Path to generated report
        """
        try:
            report_data = self._prepare_research_summary_data(research_results, topic)
            content = self._generate_markdown_research_summary(report_data)
            
            # Generate filename
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')[:50]
            filename = f"research_summary_{safe_topic}_{datetime.now().strftime('%Y%m%d')}.md"
            
            # Write report
            report_path = self.reports_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated research summary: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating research summary: {e}")
            return ""
    
    def generate_hypothesis_report(self, hypotheses: List[Dict[str, Any]], 
                                 context: str) -> str:
        """
        Generate hypothesis report with prioritization.
        
        Args:
            hypotheses: List of generated hypotheses
            context: Research context
        
        Returns:
            Path to generated report
        """
        try:
            report_data = self._prepare_hypothesis_report_data(hypotheses, context)
            content = self._generate_markdown_hypothesis_report(report_data)
            
            filename = f"hypothesis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            
            # Write report
            report_path = self.reports_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated hypothesis report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating hypothesis report: {e}")
            return ""
    
    def generate_competitive_intelligence(self, intelligence_data: Dict[str, Any]) -> str:
        """
        Generate competitive intelligence report.
        
        Args:
            intelligence_data: Competitive intelligence findings
        
        Returns:
            Path to generated report
        """
        try:
            report_data = self._prepare_competitive_intelligence_data(intelligence_data)
            content = self._generate_markdown_competitive_report(report_data)
            
            filename = f"competitive_intelligence_{datetime.now().strftime('%Y%m%d')}.md"
            
            # Write report
            report_path = self.reports_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated competitive intelligence report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating competitive intelligence: {e}")
            return ""
    
    def generate_trend_analysis(self, historical_data: List[Dict[str, Any]], 
                              time_period: int = 30) -> str:
        """
        Generate trend analysis report with visualizations.
        
        Args:
            historical_data: Historical research data
            time_period: Time period in days
        
        Returns:
            Path to generated report
        """
        try:
            report_data = self._prepare_trend_analysis_data(historical_data, time_period)
            
            # Generate visualizations
            chart_paths = self._generate_trend_charts(report_data)
            
            # Generate markdown report
            content = self._generate_markdown_trend_analysis(report_data, chart_paths)
            
            filename = f"trend_analysis_{datetime.now().strftime('%Y%m%d')}.md"
            
            # Write report
            report_path = self.reports_dir / filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated trend analysis: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            return ""
    
    def _prepare_daily_digest_data(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for daily digest report."""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "scan_summary": scan_results.get("summary", ""),
            "queries_processed": scan_results.get("queries_processed", 0),
            "novel_discoveries": scan_results.get("novel_discoveries", []),
            "hypotheses": scan_results.get("generated_hypotheses", []),
            "results": scan_results.get("results", []),
            "statistics": {
                "total_papers_found": sum(r.get("papers_found", 0) for r in scan_results.get("results", [])),
                "novel_discovery_count": len(scan_results.get("novel_discoveries", [])),
                "hypothesis_count": len(scan_results.get("generated_hypotheses", []))
            }
        }
    
    def _prepare_research_summary_data(self, research_results: List[Dict[str, Any]], 
                                     topic: str) -> Dict[str, Any]:
        """Prepare data for research summary report."""
        return {
            "topic": topic,
            "generated_at": datetime.now().isoformat(),
            "total_results": len(research_results),
            "results": research_results,
            "key_insights": self._extract_key_insights(research_results),
            "methodology_overview": self._extract_methodologies(research_results),
            "future_directions": self._extract_future_directions(research_results)
        }
    
    def _prepare_hypothesis_report_data(self, hypotheses: List[Dict[str, Any]], 
                                      context: str) -> Dict[str, Any]:
        """Prepare data for hypothesis report."""
        # Prioritize hypotheses
        prioritized = self._prioritize_hypotheses(hypotheses)
        
        return {
            "context": context,
            "generated_at": datetime.now().isoformat(),
            "total_hypotheses": len(hypotheses),
            "prioritized_hypotheses": prioritized,
            "feasibility_analysis": self._analyze_feasibility(hypotheses),
            "resource_requirements": self._estimate_resources(hypotheses)
        }
    
    def _prepare_competitive_intelligence_data(self, intelligence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for competitive intelligence report."""
        return {
            "analysis_date": intelligence_data.get("analysis_date", datetime.now().isoformat()),
            "competitors": intelligence_data.get("competitors_analyzed", []),
            "findings": intelligence_data.get("findings", []),
            "threats": intelligence_data.get("competitive_threats", []),
            "opportunities": intelligence_data.get("collaboration_opportunities", []),
            "market_insights": self._extract_market_insights(intelligence_data)
        }
    
    def _prepare_trend_analysis_data(self, historical_data: List[Dict[str, Any]], 
                                   time_period: int) -> Dict[str, Any]:
        """Prepare data for trend analysis."""
        # Convert to DataFrame for analysis
        df = pd.DataFrame(historical_data)
        
        return {
            "time_period": time_period,
            "total_data_points": len(historical_data),
            "trend_data": df,
            "emerging_topics": self._identify_emerging_topics(df),
            "research_velocity": self._calculate_research_velocity(df),
            "topic_evolution": self._analyze_topic_evolution(df)
        }
    
    def _generate_markdown_digest(self, data: Dict[str, Any]) -> str:
        """Generate markdown daily digest."""
        template = self.templates["daily_digest"]
        
        # Build discoveries section
        discoveries_section = ""
        for i, discovery in enumerate(data["novel_discoveries"][:5], 1):
            discoveries_section += f"{i}. {discovery}\n"
        
        # Build hypotheses section
        hypotheses_section = ""
        for i, hypothesis in enumerate(data["hypotheses"][:5], 1):
            hypotheses_section += f"{i}. {hypothesis}\n"
        
        # Build statistics section
        stats = data["statistics"]
        stats_section = f"""
- **Papers Found**: {stats['total_papers_found']}
- **Novel Discoveries**: {stats['novel_discovery_count']}
- **Hypotheses Generated**: {stats['hypothesis_count']}
"""
        
        return template.format(
            date=data["date"],
            summary=data["scan_summary"],
            discoveries=discoveries_section,
            hypotheses=hypotheses_section,
            statistics=stats_section,
            queries_processed=data["queries_processed"]
        )
    
    def _generate_markdown_research_summary(self, data: Dict[str, Any]) -> str:
        """Generate markdown research summary."""
        template = self.templates["research_summary"]
        
        # Build key insights
        insights_section = ""
        for i, insight in enumerate(data["key_insights"], 1):
            insights_section += f"{i}. {insight}\n"
        
        # Build methodologies
        methods_section = ""
        for method in data["methodology_overview"]:
            methods_section += f"- {method}\n"
        
        return template.format(
            topic=data["topic"],
            date=datetime.now().strftime("%Y-%m-%d"),
            total_results=data["total_results"],
            insights=insights_section,
            methodologies=methods_section,
            future_directions="\n".join(f"- {fd}" for fd in data["future_directions"])
        )
    
    def _generate_markdown_hypothesis_report(self, data: Dict[str, Any]) -> str:
        """Generate markdown hypothesis report."""
        template = self.templates["hypothesis_report"]
        
        # Build prioritized hypotheses
        hypotheses_section = ""
        for i, hyp in enumerate(data["prioritized_hypotheses"], 1):
            hypotheses_section += f"""
### Hypothesis {i}: {hyp.get('statement', 'N/A')}

**Priority Score**: {hyp.get('priority_score', 0)}/10

**Rationale**: {hyp.get('rationale', 'N/A')}

**Feasibility**: {hyp.get('feasibility', 'N/A')}

**Expected Impact**: {hyp.get('expected_impact', 'N/A')}

**Next Steps**: {hyp.get('next_steps', 'N/A')}

---
"""
        
        return template.format(
            context=data["context"],
            date=datetime.now().strftime("%Y-%m-%d"),
            total_hypotheses=data["total_hypotheses"],
            hypotheses=hypotheses_section
        )
    
    def _generate_trend_charts(self, data: Dict[str, Any]) -> List[str]:
        """Generate trend analysis charts."""
        chart_paths = []
        
        try:
            # Research velocity chart
            plt.figure(figsize=(10, 6))
            velocity_data = data["research_velocity"]
            plt.plot(velocity_data.get("dates", []), velocity_data.get("papers_per_day", []))
            plt.title("Research Discovery Velocity")
            plt.xlabel("Date")
            plt.ylabel("Papers Found Per Day")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            velocity_chart = self.reports_dir / f"velocity_chart_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(velocity_chart)
            plt.close()
            chart_paths.append(str(velocity_chart))
            
            # Topic evolution chart
            plt.figure(figsize=(12, 8))
            topics = data["topic_evolution"]
            for topic, counts in topics.items():
                plt.plot(counts.get("dates", []), counts.get("frequency", []), label=topic)
            
            plt.title("Topic Evolution Over Time")
            plt.xlabel("Date")
            plt.ylabel("Frequency")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            evolution_chart = self.reports_dir / f"evolution_chart_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(evolution_chart)
            plt.close()
            chart_paths.append(str(evolution_chart))
            
        except Exception as e:
            logger.warning(f"Error generating charts: {e}")
        
        return chart_paths
    
    def _daily_digest_template(self) -> str:
        """Template for daily digest report."""
        return """# Agteria Literature Scout - Daily Digest

**Date**: {date}

## 📊 Scan Summary

{summary}

## 📈 Statistics

{statistics}

**Queries Processed**: {queries_processed}

## 🔬 Novel Discoveries

{discoveries}

## 💡 Generated Hypotheses

{hypotheses}

---

*Generated by Agteria Literature Scout on {date}*
"""
    
    def _research_summary_template(self) -> str:
        """Template for research summary report."""
        return """# Research Summary: {topic}

**Generated**: {date}  
**Total Results Analyzed**: {total_results}

## 🎯 Key Insights

{insights}

## 🔬 Methodologies Identified

{methodologies}

## 🔮 Future Research Directions

{future_directions}

---

*Agteria Literature Scout Research Summary*
"""
    
    def _hypothesis_report_template(self) -> str:
        """Template for hypothesis report."""
        return """# Research Hypothesis Report

**Context**: {context}  
**Generated**: {date}  
**Total Hypotheses**: {total_hypotheses}

## 🧪 Prioritized Research Hypotheses

{hypotheses}

---

*Generated by Agteria Literature Scout*
"""
    
    def _competitive_intelligence_template(self) -> str:
        """Template for competitive intelligence report."""
        return """# Competitive Intelligence Report

**Analysis Date**: {date}

## 🏢 Competitors Analyzed

{competitors}

## 📊 Key Findings

{findings}

## ⚠️ Competitive Threats

{threats}

## 🤝 Collaboration Opportunities

{opportunities}

---

*Agteria Competitive Intelligence Report*
"""
    
    # Helper methods for data processing
    def _extract_key_insights(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from research results."""
        insights = []
        for result in results:
            if result.get("novel_insights"):
                insights.extend(result["novel_insights"])
        return list(set(insights))[:10]  # Top 10 unique insights
    
    def _extract_methodologies(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract methodologies from research results."""
        methodologies = set()
        for result in results:
            response = result.get("response", "")
            # Simple extraction - could be enhanced
            if "method" in response.lower():
                sentences = response.split(".")
                for sentence in sentences:
                    if "method" in sentence.lower():
                        methodologies.add(sentence.strip())
        return list(methodologies)[:5]
    
    def _extract_future_directions(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract future research directions."""
        directions = []
        for result in results:
            if result.get("next_steps"):
                directions.extend(result["next_steps"])
        return list(set(directions))[:5]
    
    def _prioritize_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize hypotheses based on feasibility and impact."""
        for hyp in hypotheses:
            # Simple scoring system - could be enhanced
            feasibility_score = self._score_feasibility(hyp.get("feasibility", ""))
            impact_score = self._score_impact(hyp.get("expected_outcome", ""))
            hyp["priority_score"] = (feasibility_score + impact_score) / 2
        
        return sorted(hypotheses, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    def _score_feasibility(self, feasibility_text: str) -> float:
        """Score hypothesis feasibility (0-10)."""
        text = feasibility_text.lower()
        if "high" in text:
            return 8.0
        elif "medium" in text:
            return 6.0
        elif "low" in text:
            return 3.0
        else:
            return 5.0  # Default
    
    def _score_impact(self, impact_text: str) -> float:
        """Score potential impact (0-10)."""
        text = impact_text.lower()
        impact_keywords = ["significant", "major", "breakthrough", "revolutionary"]
        moderate_keywords = ["moderate", "improvement", "enhance"]
        
        if any(keyword in text for keyword in impact_keywords):
            return 8.0
        elif any(keyword in text for keyword in moderate_keywords):
            return 6.0
        else:
            return 5.0
    
    def _analyze_feasibility(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze feasibility distribution."""
        feasibility_counts = {"high": 0, "medium": 0, "low": 0}
        for hyp in hypotheses:
            feasibility = hyp.get("feasibility", "").lower()
            if "high" in feasibility:
                feasibility_counts["high"] += 1
            elif "medium" in feasibility:
                feasibility_counts["medium"] += 1
            elif "low" in feasibility:
                feasibility_counts["low"] += 1
        return feasibility_counts
    
    def _estimate_resources(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, str]:
        """Estimate resource requirements."""
        return {
            "timeline": "3-12 months for initial validation",
            "personnel": "2-3 researchers per hypothesis",
            "equipment": "Standard laboratory equipment + specialized instruments",
            "funding": "$50K-200K per hypothesis depending on complexity"
        }
    
    def _extract_market_insights(self, intelligence_data: Dict[str, Any]) -> List[str]:
        """Extract market insights from competitive intelligence."""
        return [
            "Market moving towards sustainable agricultural solutions",
            "Increasing regulatory pressure on emissions",
            "Growing investor interest in climate tech",
            "Partnerships between ag-tech and pharma companies"
        ]
    
    def _identify_emerging_topics(self, df: pd.DataFrame) -> List[str]:
        """Identify emerging research topics."""
        # Simplified implementation
        return ["synthetic biology approaches", "precision fermentation", "microbiome modulation"]
    
    def _calculate_research_velocity(self, df: pd.DataFrame) -> Dict[str, List]:
        """Calculate research discovery velocity."""
        # Simplified implementation
        return {
            "dates": [datetime.now().strftime("%Y-%m-%d")],
            "papers_per_day": [5]
        }
    
    def _analyze_topic_evolution(self, df: pd.DataFrame) -> Dict[str, Dict[str, List]]:
        """Analyze how topics evolve over time."""
        # Simplified implementation
        return {
            "methane inhibition": {
                "dates": [datetime.now().strftime("%Y-%m-%d")],
                "frequency": [10]
            },
            "enzyme research": {
                "dates": [datetime.now().strftime("%Y-%m-%d")],
                "frequency": [8]
            }
        }