"""Search tools for scientific literature discovery."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import arxiv
import requests
from Bio import Entrez
from langchain_core.tools import Tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import PubmedQueryRun
from langchain_community.utilities import PubMedAPIWrapper

from ..config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivSearchTool:
    """Enhanced ArXiv search with filtering for climate/agriculture research."""
    
    def __init__(self, max_results: int = Config.MAX_ARXIV_RESULTS):
        self.max_results = max_results
        self.client = arxiv.Client()
    
    def search(self, query: str, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers related to the query.
        
        Args:
            query: Search query string
            category_filter: Optional category filter (e.g., 'q-bio', 'physics:cond-mat')
        
        Returns:
            List of paper dictionaries with metadata
        """
        try:
            # Build search query
            search_query = query
            if category_filter:
                search_query = f"cat:{category_filter} AND ({query})"
            
            # Perform search
            search = arxiv.Search(
                query=search_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = []
            for paper in self.client.results(search):
                result = {
                    'title': paper.title,
                    'authors': [str(author) for author in paper.authors],
                    'summary': paper.summary,
                    'published': paper.published.isoformat(),
                    'updated': paper.updated.isoformat(),
                    'arxiv_id': paper.entry_id.split('/')[-1],
                    'pdf_url': paper.pdf_url,
                    'categories': paper.categories,
                    'source': 'arxiv'
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} ArXiv papers for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []

class PubMedSearchTool:
    """Enhanced PubMed search for biomedical literature."""
    
    def __init__(self, max_results: int = Config.MAX_PUBMED_RESULTS):
        self.max_results = max_results
        # Set email for NCBI Entrez (required for API access)
        Entrez.email = "research@agteria.com"
    
    def search(self, query: str, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search PubMed for papers related to the query.
        
        Args:
            query: Search query string
            days_back: Limit to papers from last N days
        
        Returns:
            List of paper dictionaries with metadata
        """
        try:
            # Add date filter if specified
            search_query = query
            if days_back:
                date_filter = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
                search_query = f"({query}) AND (\"{date_filter}\"[Date - Publication] : \"3000\"[Date - Publication])"
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=self.max_results,
                sort="pub+date",
                retmode="xml"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            # Get detailed information for each paper
            if not search_results["IdList"]:
                return []
            
            ids = ",".join(search_results["IdList"])
            handle = Entrez.efetch(
                db="pubmed",
                id=ids,
                rettype="medline",
                retmode="xml"
            )
            papers = Entrez.read(handle)
            handle.close()
            
            results = []
            for paper in papers["PubmedArticle"]:
                try:
                    article = paper["MedlineCitation"]["Article"]
                    result = {
                        'title': article["ArticleTitle"],
                        'authors': self._extract_authors(article.get("AuthorList", [])),
                        'abstract': article.get("Abstract", {}).get("AbstractText", [""])[0],
                        'journal': article["Journal"]["Title"],
                        'published': self._extract_date(article.get("ArticleDate", [])),
                        'pmid': paper["MedlineCitation"]["PMID"],
                        'doi': self._extract_doi(paper["PubmedData"]),
                        'source': 'pubmed'
                    }
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error parsing PubMed paper: {e}")
                    continue
            
            logger.info(f"Found {len(results)} PubMed papers for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def _extract_authors(self, author_list: List) -> List[str]:
        """Extract author names from PubMed author list."""
        authors = []
        for author in author_list:
            if "LastName" in author and "ForeName" in author:
                authors.append(f"{author['ForeName']} {author['LastName']}")
        return authors
    
    def _extract_date(self, article_date: List) -> str:
        """Extract publication date from PubMed data."""
        if article_date:
            date_info = article_date[0]
            year = date_info.get("Year", "")
            month = date_info.get("Month", "01")
            day = date_info.get("Day", "01")
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return ""
    
    def _extract_doi(self, pubmed_data: Dict) -> str:
        """Extract DOI from PubMed data."""
        try:
            article_ids = pubmed_data.get("ArticleIdList", [])
            for id_info in article_ids:
                if id_info.attributes.get("IdType") == "doi":
                    return str(id_info)
        except:
            pass
        return ""

class WebSearchTool:
    """Web search for recent news, patents, and industry updates."""
    
    def __init__(self):
        self.serper_api_key = Config.SERPER_API_KEY
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, search_type: str = "general") -> List[Dict[str, Any]]:
        """
        Perform web search for query.
        
        Args:
            query: Search query string
            search_type: Type of search ('general', 'news', 'patents')
        
        Returns:
            List of search result dictionaries
        """
        if not self.serper_api_key:
            logger.warning("No Serper API key configured for web search")
            return []
        
        try:
            # Modify query based on search type
            if search_type == "news":
                query = f"{query} news recent"
            elif search_type == "patents":
                query = f"{query} patent"
            
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": Config.MAX_WEB_RESULTS
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("organic", []):
                result = {
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('link', ''),
                    'date': item.get('date', ''),
                    'source': 'web_search'
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} web results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

class SearchOrchestrator:
    """Orchestrates searches across multiple sources."""
    
    def __init__(self):
        self.arxiv_tool = ArxivSearchTool()
        self.pubmed_tool = PubMedSearchTool()
        self.web_tool = WebSearchTool()
    
    def search_all(self, query: str, include_web: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all available sources.
        
        Args:
            query: Search query string
            include_web: Whether to include web search results
        
        Returns:
            Dictionary with results from each source
        """
        results = {
            'arxiv': self.arxiv_tool.search(query),
            'pubmed': self.pubmed_tool.search(query)
        }
        
        if include_web:
            results['web'] = self.web_tool.search(query)
            results['news'] = self.web_tool.search(query, search_type="news")
            results['patents'] = self.web_tool.search(query, search_type="patents")
        
        return results
    
    def search_cross_domain(self, base_query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform cross-domain searches to find unexpected connections.
        
        Args:
            base_query: Base research query
        
        Returns:
            Dictionary with cross-domain search results
        """
        from ..config import CROSS_DOMAIN_KEYWORDS
        
        all_results = {}
        
        for domain_keyword in CROSS_DOMAIN_KEYWORDS:
            combined_query = f"{base_query} {domain_keyword}"
            domain_results = self.search_all(combined_query, include_web=False)
            
            # Filter and combine results
            all_results[domain_keyword] = {
                'arxiv': domain_results['arxiv'][:3],  # Top 3 from each
                'pubmed': domain_results['pubmed'][:3]
            }
        
        return all_results

# Create tool instances for LangChain
def create_langchain_search_tools(citation_tracker=None) -> List[Tool]:
    """Create LangChain-compatible tool instances."""
    
    from ..utils.citation_tracker import CitationTracker
    
    orchestrator = SearchOrchestrator()
    
    # Use provided citation tracker or fall back to global one
    if citation_tracker is None:
        global _global_citation_tracker
        if '_global_citation_tracker' not in globals():
            _global_citation_tracker = CitationTracker()
        citation_tracker = _global_citation_tracker
    
    def arxiv_search_func(query: str) -> str:
        """Search ArXiv, store papers, and return results with paper IDs."""
        results = orchestrator.arxiv_tool.search(query)
        if not results:
            return "No ArXiv papers found for this query."
        
        # Store papers and get IDs
        paper_ids = []
        for paper in results[:5]:
            # Standardize paper data format
            paper_data = {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("summary", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "url": paper.get("url", ""),
                "published": paper.get("published", ""),
                "database": "arxiv"
            }
            
            # Add to citation tracker and get ID
            paper_id = citation_tracker.add_source(paper_data)
            paper_ids.append(paper_id)
        
        # Format results with paper IDs
        formatted = f"ArXiv Papers Found (Query: {query}):\n\n"
        for i, (paper, paper_id) in enumerate(zip(results[:5], paper_ids), 1):
            formatted += f"{i}. [ID: {paper_id}] {paper['title']}\n"
            formatted += f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
            formatted += f"   Summary: {paper['summary'][:200]}...\n"
            formatted += f"   ArXiv ID: {paper['arxiv_id']}\n\n"
        
        # Add paper IDs for agent tracking
        formatted += f"PAPER_IDS: {', '.join(paper_ids)}\n"
        
        return formatted
    
    def pubmed_search_func(query: str) -> str:
        """Search PubMed, store papers, and return results with paper IDs."""
        results = orchestrator.pubmed_tool.search(query)
        if not results:
            return "No PubMed papers found for this query."
        
        # Store papers and get IDs
        paper_ids = []
        for paper in results[:5]:
            # Standardize paper data format
            paper_data = {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "journal": paper.get("journal", ""),
                "pmid": paper.get("pmid", ""),
                "doi": paper.get("doi", ""),
                "published": paper.get("published", ""),
                "database": "pubmed"
            }
            
            # Add to citation tracker and get ID
            paper_id = citation_tracker.add_source(paper_data)
            paper_ids.append(paper_id)
        
        # Format results with paper IDs
        formatted = f"PubMed Papers Found (Query: {query}):\n\n"
        for i, (paper, paper_id) in enumerate(zip(results[:5], paper_ids), 1):
            formatted += f"{i}. [ID: {paper_id}] {paper['title']}\n"
            formatted += f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
            formatted += f"   Journal: {paper['journal']}\n"
            formatted += f"   Abstract: {paper['abstract'][:200]}...\n"
            formatted += f"   PMID: {paper['pmid']}\n\n"
        
        # Add paper IDs for agent tracking
        formatted += f"PAPER_IDS: {', '.join(paper_ids)}\n"
        
        return formatted
    
    def comprehensive_search_func(query: str) -> str:
        """Search all sources, store papers, and return comprehensive results."""
        results = orchestrator.search_all(query)
        
        all_paper_ids = []
        formatted = f"Comprehensive Search Results for: {query}\n\n"
        
        # ArXiv results
        if results['arxiv']:
            formatted += "=== ArXiv Papers ===\n"
            for paper in results['arxiv'][:3]:
                # Store paper and get ID
                paper_data = {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("summary", ""),
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "url": paper.get("url", ""),
                    "published": paper.get("published", ""),
                    "database": "arxiv"
                }
                paper_id = citation_tracker.add_source(paper_data)
                all_paper_ids.append(paper_id)
                
                formatted += f"• [ID: {paper_id}] {paper['title']}\n  {paper['summary'][:100]}...\n\n"
        
        # PubMed results  
        if results['pubmed']:
            formatted += "=== PubMed Papers ===\n"
            for paper in results['pubmed'][:3]:
                # Store paper and get ID
                paper_data = {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "journal": paper.get("journal", ""),
                    "pmid": paper.get("pmid", ""),
                    "doi": paper.get("doi", ""),
                    "published": paper.get("published", ""),
                    "database": "pubmed"
                }
                paper_id = citation_tracker.add_source(paper_data)
                all_paper_ids.append(paper_id)
                
                formatted += f"• [ID: {paper_id}] {paper['title']}\n  {paper['abstract'][:100]}...\n\n"
        
        # Web results
        if results.get('web'):
            formatted += "=== Web Results ===\n"
            for item in results['web'][:2]:
                # Store web result and get ID
                paper_data = {
                    "title": item.get("title", ""),
                    "authors": [],
                    "abstract": item.get("snippet", ""),
                    "url": item.get("url", ""),
                    "database": "web"
                }
                paper_id = citation_tracker.add_source(paper_data)
                all_paper_ids.append(paper_id)
                
                formatted += f"• [ID: {paper_id}] {item['title']}\n  {item['snippet']}\n\n"
        
        # Add all paper IDs for agent tracking
        if all_paper_ids:
            formatted += f"ALL_PAPER_IDS: {', '.join(all_paper_ids)}\n"
        
        return formatted
    
    tools = [
        Tool(
            name="arxiv_search",
            description="Search ArXiv for scientific papers and preprints. Use for recent research in physics, chemistry, biology, and computer science.",
            func=arxiv_search_func
        ),
        Tool(
            name="pubmed_search", 
            description="Search PubMed for biomedical and life science literature. Use for medical, biological, and agricultural research.",
            func=pubmed_search_func
        ),
        Tool(
            name="comprehensive_search",
            description="Search across multiple scientific databases (ArXiv, PubMed, Web) for comprehensive results. Use when you need broad coverage.",
            func=comprehensive_search_func
        )
    ]
    
    # Add web search if API key is available
    if Config.SERPER_API_KEY:
        def web_search_func(query: str) -> str:
            """Search web for recent news and information, store results with IDs."""
            results = orchestrator.web_tool.search(query)
            if not results:
                return "No web results found for this query."
            
            # Store web results and get IDs
            paper_ids = []
            for item in results[:5]:
                paper_data = {
                    "title": item.get("title", ""),
                    "authors": [],
                    "abstract": item.get("snippet", ""),
                    "url": item.get("url", ""),
                    "database": "web"
                }
                paper_id = citation_tracker.add_source(paper_data)
                paper_ids.append(paper_id)
            
            formatted = f"Web Search Results (Query: {query}):\n\n"
            for i, (item, paper_id) in enumerate(zip(results[:5], paper_ids), 1):
                formatted += f"{i}. [ID: {paper_id}] {item['title']}\n"
                formatted += f"   {item['snippet']}\n"
                formatted += f"   URL: {item['url']}\n\n"
            
            # Add paper IDs for agent tracking
            formatted += f"WEB_PAPER_IDS: {', '.join(paper_ids)}\n"
            
            return formatted
        
        tools.append(Tool(
            name="web_search",
            description="Search the web for recent news, industry updates, and general information. Use for current events and non-academic sources.",
            func=web_search_func
        ))
    
    return tools

def get_global_citation_tracker():
    """Get the global citation tracker instance."""
    global _global_citation_tracker
    if '_global_citation_tracker' not in globals():
        from ..utils.citation_tracker import CitationTracker
        _global_citation_tracker = CitationTracker()
    return _global_citation_tracker