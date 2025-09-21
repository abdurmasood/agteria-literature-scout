"""Citation tracking and source attribution utilities."""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class PaperSource:
    """Data class for paper source information."""
    id: str
    title: str
    authors: List[str]
    journal: Optional[str] = None
    published: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    database: str = "unknown"
    abstract: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "published": self.published,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pmid": self.pmid,
            "url": self.url,
            "database": self.database,
            "abstract": self.abstract
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperSource':
        """Create PaperSource from dictionary."""
        return cls(**data)
    
    def generate_citation(self, style: str = "apa") -> str:
        """Generate formatted citation."""
        if style == "apa":
            return self._apa_citation()
        elif style == "mla":
            return self._mla_citation()
        elif style == "chicago":
            return self._chicago_citation()
        else:
            return self._simple_citation()
    
    def _apa_citation(self) -> str:
        """Generate APA style citation."""
        # Format: Author, A. A. (Year). Title. Journal Name, Volume(Issue), pages. DOI or URL
        authors_str = self._format_authors_apa()
        year = self._extract_year()
        
        citation = f"{authors_str} ({year}). {self.title}."
        
        if self.journal:
            citation += f" {self.journal}."
        
        if self.doi:
            citation += f" https://doi.org/{self.doi}"
        elif self.url:
            citation += f" {self.url}"
        
        return citation
    
    def _mla_citation(self) -> str:
        """Generate MLA style citation."""
        # Format: Author(s). "Title." Journal Name, Date, URL/DOI.
        authors_str = self._format_authors_mla()
        
        citation = f"{authors_str}. \"{self.title}.\""
        
        if self.journal:
            citation += f" {self.journal},"
        
        if self.published:
            citation += f" {self.published},"
        
        if self.doi:
            citation += f" https://doi.org/{self.doi}."
        elif self.url:
            citation += f" {self.url}."
        
        return citation
    
    def _chicago_citation(self) -> str:
        """Generate Chicago style citation."""
        authors_str = self._format_authors_chicago()
        
        citation = f"{authors_str}. \"{self.title}.\""
        
        if self.journal:
            citation += f" {self.journal}"
        
        if self.published:
            citation += f" ({self.published})."
        
        if self.doi:
            citation += f" https://doi.org/{self.doi}."
        
        return citation
    
    def _simple_citation(self) -> str:
        """Generate simple citation format."""
        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += " et al."
        
        year = self._extract_year()
        
        return f"{authors} ({year}). {self.title}. {self.database.title()}."
    
    def _format_authors_apa(self) -> str:
        """Format authors for APA style."""
        if not self.authors:
            return "Unknown Author"
        
        if len(self.authors) == 1:
            return self._format_single_author_apa(self.authors[0])
        elif len(self.authors) <= 20:
            formatted = [self._format_single_author_apa(author) for author in self.authors[:-1]]
            return ", ".join(formatted) + f", & {self._format_single_author_apa(self.authors[-1])}"
        else:
            # For 21+ authors, list first 19, ellipsis, then last author
            formatted = [self._format_single_author_apa(author) for author in self.authors[:19]]
            return ", ".join(formatted) + f", ... {self._format_single_author_apa(self.authors[-1])}"
    
    def _format_single_author_apa(self, author: str) -> str:
        """Format single author for APA style (Last, F. M.)."""
        parts = author.strip().split()
        if len(parts) >= 2:
            # Handle "Last, F." format
            if ',' in author:
                return author
            
            last_name = parts[-1]
            first_names = parts[:-1]
            initials = ". ".join([name[0].upper() for name in first_names if name])
            return f"{last_name}, {initials}."
        return author
    
    def _format_authors_mla(self) -> str:
        """Format authors for MLA style."""
        if not self.authors:
            return "Unknown Author"
        
        if len(self.authors) == 1:
            return self.authors[0]
        elif len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        else:
            return f"{self.authors[0]} et al."
    
    def _format_authors_chicago(self) -> str:
        """Format authors for Chicago style."""
        if not self.authors:
            return "Unknown Author"
        
        if len(self.authors) == 1:
            return self.authors[0]
        elif len(self.authors) <= 3:
            return ", ".join(self.authors[:-1]) + f", and {self.authors[-1]}"
        else:
            return f"{self.authors[0]} et al."
    
    def _extract_year(self) -> str:
        """Extract year from publication date."""
        if self.published:
            # Try to extract year from various date formats
            import re
            year_match = re.search(r'(\d{4})', str(self.published))
            if year_match:
                return year_match.group(1)
        return "n.d."  # no date

@dataclass
class InsightWithSources:
    """Data class for insights with source attribution."""
    insight: str
    source_ids: List[str]
    confidence: str = "medium"
    quotes: List[str] = None
    insight_type: str = "general"  # "discovery", "hypothesis", "next_step", etc.
    
    def __post_init__(self):
        if self.quotes is None:
            self.quotes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "insight": self.insight,
            "source_ids": self.source_ids,
            "confidence": self.confidence,
            "quotes": self.quotes,
            "insight_type": self.insight_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightWithSources':
        """Create InsightWithSources from dictionary."""
        return cls(**data)

class CitationTracker:
    """Tracks citations and sources throughout the research process."""
    
    def __init__(self):
        self.sources: Dict[str, PaperSource] = {}
        self.insights: List[InsightWithSources] = []
        self.session_id = self._generate_session_id()
        logger.info(f"Initialized CitationTracker with session ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def add_source(self, paper_data: Dict[str, Any]) -> str:
        """
        Add a paper source and return its ID.
        
        Args:
            paper_data: Dictionary containing paper information
            
        Returns:
            String ID for the added source
        """
        # Generate unique ID based on paper content
        source_id = self._generate_source_id(paper_data)
        
        # Create PaperSource object
        source = PaperSource(
            id=source_id,
            title=paper_data.get("title", "Unknown Title"),
            authors=paper_data.get("authors", []),
            journal=paper_data.get("journal"),
            published=paper_data.get("published"),
            doi=paper_data.get("doi"),
            arxiv_id=paper_data.get("arxiv_id"),
            pmid=paper_data.get("pmid"),
            url=paper_data.get("url"),
            database=paper_data.get("database", "unknown"),
            abstract=paper_data.get("abstract")
        )
        
        self.sources[source_id] = source
        logger.info(f"Added source {source_id}: {source.title[:50]}...")
        
        return source_id
    
    def add_insight(self, insight: str, source_ids: List[str], 
                   confidence: str = "medium", quotes: List[str] = None,
                   insight_type: str = "general") -> None:
        """
        Add an insight with its source attribution.
        
        Args:
            insight: The insight text
            source_ids: List of source IDs that support this insight
            confidence: Confidence level (low, medium, high)
            quotes: Optional supporting quotes from sources
            insight_type: Type of insight (discovery, hypothesis, etc.)
        """
        insight_obj = InsightWithSources(
            insight=insight,
            source_ids=source_ids,
            confidence=confidence,
            quotes=quotes or [],
            insight_type=insight_type
        )
        
        self.insights.append(insight_obj)
        logger.info(f"Added insight with {len(source_ids)} sources: {insight[:50]}...")
    
    def get_source(self, source_id: str) -> Optional[PaperSource]:
        """Get source by ID."""
        return self.sources.get(source_id)
    
    def get_sources(self, source_ids: List[str]) -> List[PaperSource]:
        """Get multiple sources by IDs."""
        return [self.sources[sid] for sid in source_ids if sid in self.sources]
    
    def get_all_sources(self) -> List[PaperSource]:
        """Get all sources."""
        return list(self.sources.values())
    
    def get_insights_by_type(self, insight_type: str) -> List[InsightWithSources]:
        """Get insights by type."""
        return [insight for insight in self.insights if insight.insight_type == insight_type]
    
    def generate_bibliography(self, style: str = "apa") -> str:
        """
        Generate formatted bibliography of all sources.
        
        Args:
            style: Citation style (apa, mla, chicago, simple)
            
        Returns:
            Formatted bibliography string
        """
        if not self.sources:
            return "No sources available."
        
        citations = []
        for source in sorted(self.sources.values(), key=lambda x: x.title):
            citations.append(source.generate_citation(style))
        
        bibliography = f"## Bibliography ({style.upper()})\n\n"
        for i, citation in enumerate(citations, 1):
            bibliography += f"{i}. {citation}\n"
        
        return bibliography
    
    def export_sources_json(self) -> str:
        """Export all sources as JSON string."""
        sources_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "sources": {sid: source.to_dict() for sid, source in self.sources.items()},
            "insights": [insight.to_dict() for insight in self.insights]
        }
        return json.dumps(sources_data, indent=2)
    
    def import_sources_json(self, json_str: str) -> None:
        """Import sources from JSON string."""
        try:
            data = json.loads(json_str)
            
            # Import sources
            for sid, source_data in data.get("sources", {}).items():
                self.sources[sid] = PaperSource.from_dict(source_data)
            
            # Import insights
            for insight_data in data.get("insights", []):
                self.insights.append(InsightWithSources.from_dict(insight_data))
            
            logger.info(f"Imported {len(self.sources)} sources and {len(self.insights)} insights")
            
        except Exception as e:
            logger.error(f"Error importing sources from JSON: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked sources and insights."""
        database_counts = {}
        for source in self.sources.values():
            db = source.database
            database_counts[db] = database_counts.get(db, 0) + 1
        
        insight_type_counts = {}
        for insight in self.insights:
            itype = insight.insight_type
            insight_type_counts[itype] = insight_type_counts.get(itype, 0) + 1
        
        return {
            "total_sources": len(self.sources),
            "total_insights": len(self.insights),
            "sources_by_database": database_counts,
            "insights_by_type": insight_type_counts,
            "session_id": self.session_id
        }
    
    def _generate_source_id(self, paper_data: Dict[str, Any]) -> str:
        """Generate unique source ID based on paper content."""
        # Use DOI if available (most reliable)
        if paper_data.get("doi"):
            return f"doi_{hashlib.md5(paper_data['doi'].encode()).hexdigest()[:8]}"
        
        # Use ArXiv ID if available
        if paper_data.get("arxiv_id"):
            return f"arxiv_{paper_data['arxiv_id'].replace('.', '_').replace('/', '_')}"
        
        # Use PMID if available
        if paper_data.get("pmid"):
            return f"pubmed_{paper_data['pmid']}"
        
        # Fallback to title + authors hash
        title = paper_data.get("title", "")
        authors = str(paper_data.get("authors", []))
        content = f"{title}{authors}"
        hash_id = hashlib.md5(content.encode()).hexdigest()[:8]
        
        database = paper_data.get("database", "unknown")
        return f"{database}_{hash_id}"
    
    def clear(self) -> None:
        """Clear all sources and insights."""
        self.sources.clear()
        self.insights.clear()
        logger.info("Cleared all sources and insights")
    
    def validate_insight_sources(self) -> List[str]:
        """
        Validate that all insight sources exist.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for i, insight in enumerate(self.insights):
            for source_id in insight.source_ids:
                if source_id not in self.sources:
                    errors.append(f"Insight {i}: Missing source {source_id}")
        
        return errors