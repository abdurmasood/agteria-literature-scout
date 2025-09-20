"""Configuration module for Agteria Literature Scout."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for API keys and settings."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")  # For Google Search
    
    # Model Configuration
    DEFAULT_MODEL: str = "gpt-4-turbo-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.7
    
    # Search Configuration
    MAX_ARXIV_RESULTS: int = 10
    MAX_PUBMED_RESULTS: int = 10
    MAX_WEB_RESULTS: int = 5
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    MAX_DOC_LENGTH: int = 50000
    
    # Vector Store Configuration
    VECTOR_DB_PATH: str = "./data/research_db"
    COLLECTION_NAME: str = "research_papers"
    
    # Report Configuration
    REPORTS_DIR: str = "./reports"
    MAX_REPORT_DAYS: int = 30
    
    @classmethod
    def validate_keys(cls) -> bool:
        """Validate that required API keys are set."""
        missing_keys = []
        
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        
        if missing_keys:
            print(f"Missing required API keys: {', '.join(missing_keys)}")
            print("Please set these in your .env file")
            return False
        
        return True
    
    @classmethod
    def create_env_template(cls) -> str:
        """Create a template .env file content."""
        return """# Agteria Literature Scout - Environment Variables

# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional API Keys (for enhanced functionality)
SERPER_API_KEY=your_serper_api_key_here

# Database URLs (if using remote services)
# CHROMA_DB_URL=your_chroma_db_url_here
"""

# Research domain-specific keywords for Agteria
AGTERIA_KEYWORDS = [
    "methane inhibition",
    "cattle emissions",
    "livestock greenhouse gas",
    "ruminant fermentation", 
    "methanogenesis inhibitor",
    "feed additive",
    "enzyme inhibitor",
    "methane reduction",
    "bovine methane",
    "agricultural emissions"
]

# Cross-domain search terms for novel discoveries
CROSS_DOMAIN_KEYWORDS = [
    "algae enzyme inhibition",
    "marine bacteria methane",
    "pharmaceutical enzyme blocker",
    "industrial catalyst methane",
    "plant secondary metabolite",
    "microbial fermentation control",
    "biotechnology methane capture",
    "synthetic biology methane"
]

# Target journals and databases for focused searches
TARGET_SOURCES = [
    "Animal Feed Science and Technology",
    "Journal of Dairy Science", 
    "Applied and Environmental Microbiology",
    "Environmental Science & Technology",
    "Nature Climate Change",
    "Agriculture, Ecosystems & Environment",
    "Bioresource Technology",
    "Animal Production Science"
]