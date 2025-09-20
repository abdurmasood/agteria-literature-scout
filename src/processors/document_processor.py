"""Document processing for scientific papers and research literature."""

import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import requests

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import ArxivLoader, UnstructuredPDFLoader

from ..config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process scientific documents for analysis and storage."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Scientific text splitter with specialized separators
        self.scientific_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=[
                "\n\n## ",  # Section headers
                "\n\n### ",  # Subsection headers
                "\nAbstract\n", "\nIntroduction\n", "\nMethods\n", 
                "\nResults\n", "\nDiscussion\n", "\nConclusion\n",
                "\n\n", "\n", ". ", " ", ""
            ]
        )
    
    def process_arxiv_paper(self, arxiv_id: str) -> List[Document]:
        """
        Process an ArXiv paper by ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.08727")
        
        Returns:
            List of processed document chunks
        """
        try:
            # Use ArXiv loader
            loader = ArxivLoader(
                query=arxiv_id,
                load_max_docs=1,
                doc_content_chars_max=Config.MAX_DOC_LENGTH
            )
            
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No documents found for ArXiv ID: {arxiv_id}")
                return []
            
            # Process the document
            processed_docs = []
            for doc in documents:
                # Extract metadata
                metadata = self._extract_arxiv_metadata(doc)
                
                # Split into chunks
                chunks = self.scientific_splitter.split_documents([doc])
                
                # Add metadata to each chunk
                for chunk in chunks:
                    chunk.metadata.update(metadata)
                    chunk.metadata["chunk_id"] = self._generate_chunk_id(chunk.page_content)
                    chunk.metadata["processed_at"] = datetime.now().isoformat()
                
                processed_docs.extend(chunks)
            
            logger.info(f"Processed ArXiv paper {arxiv_id} into {len(processed_docs)} chunks")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing ArXiv paper {arxiv_id}: {e}")
            return []
    
    def process_pdf_url(self, pdf_url: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a PDF from URL.
        
        Args:
            pdf_url: URL to the PDF file
            metadata: Optional metadata to attach
        
        Returns:
            List of processed document chunks
        """
        try:
            # Download and process PDF
            loader = PyPDFLoader(pdf_url)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content extracted from PDF: {pdf_url}")
                return []
            
            # Combine all pages
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Create single document
            combined_doc = Document(
                page_content=full_text,
                metadata={
                    "source": pdf_url,
                    "document_type": "pdf",
                    "total_pages": len(documents),
                    **(metadata or {})
                }
            )
            
            # Split into chunks
            chunks = self.scientific_splitter.split_documents([combined_doc])
            
            # Process each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "chunk_id": self._generate_chunk_id(chunk.page_content),
                    "processed_at": datetime.now().isoformat()
                })
                processed_chunks.append(chunk)
            
            logger.info(f"Processed PDF from {pdf_url} into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF from {pdf_url}: {e}")
            return []
    
    def process_text_content(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Process raw text content (e.g., from abstracts, web scraping).
        
        Args:
            text: Raw text content
            metadata: Metadata about the content
        
        Returns:
            List of processed document chunks
        """
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Create document
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "document_type": "text",
                    "content_length": len(cleaned_text),
                    **metadata
                }
            )
            
            # Split into chunks if text is long
            if len(cleaned_text) > Config.CHUNK_SIZE:
                chunks = self.scientific_splitter.split_documents([doc])
            else:
                chunks = [doc]
            
            # Process chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "chunk_id": self._generate_chunk_id(chunk.page_content),
                    "processed_at": datetime.now().isoformat()
                })
                processed_chunks.append(chunk)
            
            logger.info(f"Processed text content into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return []
    
    def extract_paper_metadata(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize metadata from paper data.
        
        Args:
            paper_data: Raw paper data from search results
        
        Returns:
            Normalized metadata dictionary
        """
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "document_type": "research_paper"
        }
        
        # Common fields
        field_mappings = {
            "title": ["title"],
            "authors": ["authors", "author"],
            "abstract": ["abstract", "summary"],
            "published_date": ["published", "publication_date", "date"],
            "journal": ["journal", "venue"],
            "doi": ["doi"],
            "source_database": ["source"],
            "arxiv_id": ["arxiv_id", "id"],
            "pmid": ["pmid"],
            "url": ["url", "pdf_url", "link"],
            "categories": ["categories", "keywords", "subject_areas"]
        }
        
        for standard_field, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in paper_data:
                    metadata[standard_field] = paper_data[field]
                    break
        
        # Generate unique document ID
        content_for_id = f"{metadata.get('title', '')}{metadata.get('authors', '')}"
        metadata["document_id"] = hashlib.md5(content_for_id.encode()).hexdigest()
        
        # Extract year from publication date
        if "published_date" in metadata:
            try:
                date_str = str(metadata["published_date"])
                year_match = re.search(r"(\d{4})", date_str)
                if year_match:
                    metadata["publication_year"] = int(year_match.group(1))
            except:
                pass
        
        return metadata
    
    def identify_document_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and extract major sections from a scientific paper.
        
        Args:
            text: Full text of the paper
        
        Returns:
            Dictionary with section content
        """
        sections = {}
        
        # Common section patterns
        section_patterns = {
            "abstract": r"(?i)abstract\s*:?\s*(.*?)(?=\n\s*(?:introduction|keywords|1\.|background))",
            "introduction": r"(?i)(?:introduction|1\.\s*introduction)\s*:?\s*(.*?)(?=\n\s*(?:methods|methodology|2\.))",
            "methods": r"(?i)(?:methods|methodology|materials\s+and\s+methods|2\.\s*(?:methods|methodology))\s*:?\s*(.*?)(?=\n\s*(?:results|3\.))",
            "results": r"(?i)(?:results|3\.\s*results)\s*:?\s*(.*?)(?=\n\s*(?:discussion|conclusion|4\.))",
            "discussion": r"(?i)(?:discussion|4\.\s*discussion)\s*:?\s*(.*?)(?=\n\s*(?:conclusion|references|5\.))",
            "conclusion": r"(?i)(?:conclusion|conclusions)\s*:?\s*(.*?)(?=\n\s*(?:references|acknowledgments))"
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                section_content = match.group(1).strip()
                # Clean up the content
                section_content = re.sub(r'\n+', ' ', section_content)
                section_content = re.sub(r'\s+', ' ', section_content)
                sections[section_name] = section_content
        
        return sections
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract citations from paper text.
        
        Args:
            text: Paper text
        
        Returns:
            List of extracted citations
        """
        citations = []
        
        # Pattern for citations in various formats
        citation_patterns = [
            r"\(([^)]+\d{4}[^)]*)\)",  # (Author, 2023)
            r"\[(\d+(?:[-,]\s*\d+)*)\]",  # [1], [1-3], [1,2,3]
            r"(?:doi|DOI):\s*(10\.\d+/[^\s]+)"  # DOI patterns
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        # Remove duplicates and empty strings
        citations = list(set([c.strip() for c in citations if c.strip()]))
        
        return citations[:50]  # Limit to prevent overflow
    
    def _extract_arxiv_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract metadata from ArXiv document."""
        metadata = doc.metadata.copy()
        
        # Normalize ArXiv metadata
        normalized = {
            "document_type": "arxiv_paper",
            "source_database": "arxiv"
        }
        
        # Map ArXiv fields to standard fields
        field_mappings = {
            "Entry ID": "arxiv_url",
            "Published": "published_date",
            "Title": "title",
            "Authors": "authors",
            "Summary": "abstract"
        }
        
        for arxiv_field, standard_field in field_mappings.items():
            if arxiv_field in metadata:
                normalized[standard_field] = metadata[arxiv_field]
        
        # Extract ArXiv ID from URL
        if "arxiv_url" in normalized:
            arxiv_id_match = re.search(r"arxiv\.org/abs/(.+)$", normalized["arxiv_url"])
            if arxiv_id_match:
                normalized["arxiv_id"] = arxiv_id_match.group(1)
        
        return normalized
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove special characters that might interfere (simplified)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\/\@\#\$\%\&\*\+\=\<\>]', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        return text.strip()
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate unique ID for a chunk of content."""
        # Use first 100 characters + hash for uniqueness
        prefix = content[:100]
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{hash_part}_{len(content)}"

class DuplicateDetector:
    """Detect duplicate papers and content."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def is_duplicate(self, doc1: Document, doc2: Document) -> bool:
        """
        Check if two documents are duplicates.
        
        Args:
            doc1: First document
            doc2: Second document
        
        Returns:
            True if documents are likely duplicates
        """
        # Check metadata first (faster)
        if self._metadata_match(doc1.metadata, doc2.metadata):
            return True
        
        # Check content similarity
        similarity = self._calculate_similarity(doc1.page_content, doc2.page_content)
        return similarity >= self.similarity_threshold
    
    def _metadata_match(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> bool:
        """Check if metadata indicates duplicate papers."""
        # Same DOI
        if meta1.get("doi") and meta2.get("doi"):
            return meta1["doi"] == meta2["doi"]
        
        # Same ArXiv ID
        if meta1.get("arxiv_id") and meta2.get("arxiv_id"):
            return meta1["arxiv_id"] == meta2["arxiv_id"]
        
        # Same PMID
        if meta1.get("pmid") and meta2.get("pmid"):
            return meta1["pmid"] == meta2["pmid"]
        
        # Same title and authors
        if (meta1.get("title") and meta2.get("title") and 
            meta1.get("authors") and meta2.get("authors")):
            title_match = meta1["title"].lower() == meta2["title"].lower()
            # Simple author comparison (could be enhanced)
            authors1 = str(meta1["authors"]).lower()
            authors2 = str(meta2["authors"]).lower()
            author_match = authors1 == authors2
            return title_match and author_match
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts."""
        # Simple word-based similarity (could use more sophisticated methods)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class ContentValidator:
    """Validate content quality and relevance."""
    
    def __init__(self):
        self.min_content_length = 100
        self.required_fields = ["title"]
    
    def is_valid_document(self, doc: Document) -> Tuple[bool, List[str]]:
        """
        Validate if document meets quality standards.
        
        Args:
            doc: Document to validate
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check content length
        if len(doc.page_content) < self.min_content_length:
            issues.append(f"Content too short: {len(doc.page_content)} characters")
        
        # Check required metadata fields
        for field in self.required_fields:
            if field not in doc.metadata or not doc.metadata[field]:
                issues.append(f"Missing required field: {field}")
        
        # Check for garbled text (too many special characters)
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', doc.page_content)) / len(doc.page_content)
        if special_char_ratio > 0.3:
            issues.append(f"High special character ratio: {special_char_ratio:.2f}")
        
        # Check for reasonable word count
        words = doc.page_content.split()
        if len(words) < 20:
            issues.append(f"Too few words: {len(words)}")
        
        return len(issues) == 0, issues
    
    def get_quality_score(self, doc: Document) -> float:
        """
        Calculate quality score (0-1) for document.
        
        Args:
            doc: Document to score
        
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Content length score
        length = len(doc.page_content)
        if length < self.min_content_length:
            score *= 0.5
        elif length < 500:
            score *= 0.8
        
        # Metadata completeness score
        important_fields = ["title", "authors", "abstract", "published_date"]
        present_fields = sum(1 for field in important_fields if doc.metadata.get(field))
        score *= present_fields / len(important_fields)
        
        # Text quality score
        words = doc.page_content.split()
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 3:  # Too many short words
                score *= 0.8
            elif avg_word_length > 8:  # Too many long words (might be garbled)
                score *= 0.9
        
        return max(0.0, min(1.0, score))