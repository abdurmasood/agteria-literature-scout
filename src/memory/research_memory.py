"""Research memory system using ChromaDB for persistent knowledge storage."""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from ..config import Config
from ..processors.document_processor import DuplicateDetector, ContentValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchMemory:
    """Persistent memory system for research papers and analysis."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or Config.VECTOR_DB_PATH
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize collections
        self._initialize_collections()
        
        # Initialize LangChain vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings
        )
        
        # Initialize utilities
        self.duplicate_detector = DuplicateDetector()
        self.content_validator = ContentValidator()
        
        logger.info(f"Initialized ResearchMemory with {self.get_document_count()} documents")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections."""
        try:
            # Main papers collection
            self.papers_collection = self.chroma_client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"description": "Research papers and abstracts"}
            )
            
            # Analysis results collection
            self.analysis_collection = self.chroma_client.get_or_create_collection(
                name="analysis_results",
                metadata={"description": "Paper analysis and hypothesis results"}
            )
            
            # Search history collection
            self.search_history_collection = self.chroma_client.get_or_create_collection(
                name="search_history",
                metadata={"description": "Search queries and results"}
            )
            
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise
    
    def add_papers(self, documents: List[Document], analysis_results: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add research papers to memory with duplicate detection.
        
        Args:
            documents: List of processed documents
            analysis_results: Optional analysis results for each document
        
        Returns:
            List of document IDs that were added (excluding duplicates)
        """
        added_ids = []
        
        for i, doc in enumerate(documents):
            try:
                # Validate document quality
                is_valid, issues = self.content_validator.is_valid_document(doc)
                if not is_valid:
                    logger.warning(f"Skipping invalid document: {issues}")
                    continue
                
                # Check for duplicates
                if self._is_duplicate_in_memory(doc):
                    logger.info(f"Skipping duplicate document: {doc.metadata.get('title', 'Unknown')}")
                    continue
                
                # Generate unique ID
                doc_id = doc.metadata.get("document_id") or str(uuid.uuid4())
                doc.metadata["doc_id"] = doc_id
                doc.metadata["added_to_memory"] = datetime.now().isoformat()
                
                # Add quality score
                quality_score = self.content_validator.get_quality_score(doc)
                doc.metadata["quality_score"] = quality_score
                
                # Add to vector store
                self.vector_store.add_documents([doc], ids=[doc_id])
                
                # Add analysis results if provided
                if analysis_results and i < len(analysis_results):
                    self._store_analysis_result(doc_id, analysis_results[i])
                
                added_ids.append(doc_id)
                logger.info(f"Added document {doc_id} to memory")
                
            except Exception as e:
                logger.error(f"Error adding document to memory: {e}")
                continue
        
        logger.info(f"Added {len(added_ids)} new documents to memory")
        return added_ids
    
    def search_papers(self, query: str, k: int = 10, filter_dict: Optional[Dict[str, Any]] = None,
                     min_quality_score: float = 0.0) -> List[Document]:
        """
        Search for papers using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            min_quality_score: Minimum quality score threshold
        
        Returns:
            List of relevant documents
        """
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=query,
                k=k * 2,  # Get more results for filtering
                filter=filter_dict
            )
            
            # Filter by quality score
            filtered_results = [
                doc for doc in results 
                if doc.metadata.get("quality_score", 1.0) >= min_quality_score
            ]
            
            # Limit to requested number
            final_results = filtered_results[:k]
            
            # Log search
            self._log_search(query, len(final_results))
            
            logger.info(f"Found {len(final_results)} papers for query: {query}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for papers with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Found {len(results)} papers with scores for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching papers with scores: {e}")
            return []
    
    def get_paper_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific paper by its ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document if found, None otherwise
        """
        try:
            results = self.papers_collection.get(ids=[doc_id], include=["documents", "metadatas"])
            
            if results["documents"]:
                doc = Document(
                    page_content=results["documents"][0],
                    metadata=results["metadatas"][0]
                )
                return doc
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving paper {doc_id}: {e}")
            return None
    
    def get_recent_papers(self, days: int = 7, limit: int = 50) -> List[Document]:
        """
        Get recently added papers.
        
        Args:
            days: Number of days back to look
            limit: Maximum number of papers to return
        
        Returns:
            List of recent documents
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Search for papers added after cutoff date
            results = self.vector_store.similarity_search(
                query="recent research",  # Generic query
                k=limit,
                filter={"added_to_memory": {"$gte": cutoff_date}}
            )
            
            logger.info(f"Found {len(results)} papers from last {days} days")
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent papers: {e}")
            return []
    
    def get_papers_by_topic(self, topic: str, limit: int = 20) -> List[Document]:
        """
        Get papers related to a specific topic.
        
        Args:
            topic: Research topic
            limit: Maximum number of papers
        
        Returns:
            List of topic-related documents
        """
        # Use semantic search for topic
        return self.search_papers(query=topic, k=limit)
    
    def get_analysis_result(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis results for a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Analysis results if found
        """
        try:
            results = self.analysis_collection.get(
                ids=[f"analysis_{doc_id}"],
                include=["documents", "metadatas"]
            )
            
            if results["documents"]:
                return json.loads(results["documents"][0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting analysis for {doc_id}: {e}")
            return None
    
    def get_search_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent search history.
        
        Args:
            limit: Maximum number of searches to return
        
        Returns:
            List of search history entries
        """
        try:
            results = self.search_history_collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            history = []
            for i, doc in enumerate(results["documents"]):
                entry = json.loads(doc)
                entry.update(results["metadatas"][i])
                history.append(entry)
            
            # Sort by timestamp (most recent first)
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting search history: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in memory."""
        try:
            return self.papers_collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            stats = {
                "total_documents": self.get_document_count(),
                "total_analyses": self.analysis_collection.count(),
                "total_searches": self.search_history_collection.count(),
                "last_updated": datetime.now().isoformat()
            }
            
            # Get document breakdown by source
            results = self.papers_collection.get(include=["metadatas"])
            source_counts = {}
            quality_scores = []
            
            for metadata in results["metadatas"]:
                source = metadata.get("source_database", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
                
                quality = metadata.get("quality_score", 1.0)
                quality_scores.append(quality)
            
            stats["documents_by_source"] = source_counts
            
            if quality_scores:
                stats["average_quality_score"] = sum(quality_scores) / len(quality_scores)
                stats["min_quality_score"] = min(quality_scores)
                stats["max_quality_score"] = max(quality_scores)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def clear_old_documents(self, days: int = 90):
        """
        Clear old documents to manage memory size.
        
        Args:
            days: Remove documents older than this many days
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get old documents
            results = self.papers_collection.get(
                where={"added_to_memory": {"$lt": cutoff_date}},
                include=["ids"]
            )
            
            if results["ids"]:
                # Delete old documents
                self.papers_collection.delete(ids=results["ids"])
                logger.info(f"Removed {len(results['ids'])} old documents")
            
        except Exception as e:
            logger.error(f"Error clearing old documents: {e}")
    
    def backup_memory(self, backup_path: str):
        """
        Create a backup of the memory database.
        
        Args:
            backup_path: Path to save backup
        """
        try:
            import shutil
            shutil.copytree(self.persist_directory, backup_path)
            logger.info(f"Memory backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up memory: {e}")
    
    def _is_duplicate_in_memory(self, doc: Document) -> bool:
        """Check if document is already in memory."""
        try:
            # Check by document ID first
            doc_id = doc.metadata.get("document_id")
            if doc_id:
                existing = self.get_paper_by_id(doc_id)
                if existing:
                    return True
            
            # Check by title similarity for papers without IDs
            title = doc.metadata.get("title")
            if title:
                similar_docs = self.search_papers(
                    query=title,
                    k=5,
                    min_quality_score=0.0
                )
                
                for similar_doc in similar_docs:
                    if self.duplicate_detector.is_duplicate(doc, similar_doc):
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
            return False
    
    def _store_analysis_result(self, doc_id: str, analysis: Dict[str, Any]):
        """Store analysis results for a document."""
        try:
            analysis_id = f"analysis_{doc_id}"
            
            self.analysis_collection.add(
                documents=[json.dumps(analysis)],
                ids=[analysis_id],
                metadatas=[{
                    "doc_id": doc_id,
                    "analysis_type": analysis.get("analysis_type", "general"),
                    "created_at": datetime.now().isoformat()
                }]
            )
            
        except Exception as e:
            logger.error(f"Error storing analysis for {doc_id}: {e}")
    
    def _log_search(self, query: str, result_count: int):
        """Log search query and results."""
        try:
            search_id = str(uuid.uuid4())
            search_data = {
                "query": query,
                "result_count": result_count,
                "timestamp": datetime.now().isoformat()
            }
            
            self.search_history_collection.add(
                documents=[json.dumps(search_data)],
                ids=[search_id],
                metadatas=[{
                    "query": query,
                    "result_count": result_count,
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
        except Exception as e:
            logger.warning(f"Error logging search: {e}")

class KnowledgeGraph:
    """Simple knowledge graph for tracking research concepts and relationships."""
    
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.concepts = {}  # concept -> {papers: [], relations: []}
        self.relations = {}  # (concept1, concept2) -> relation_type
    
    def extract_concepts(self, doc: Document) -> List[str]:
        """
        Extract key concepts from a document.
        
        Args:
            doc: Document to analyze
        
        Returns:
            List of extracted concepts
        """
        # Simple keyword extraction (could be enhanced with NER)
        text = doc.page_content.lower()
        
        # Scientific concepts
        concepts = []
        
        # Chemical/molecular terms
        chemical_patterns = [
            r"methane", r"enzyme", r"inhibitor", r"catalyst", r"molecule",
            r"protein", r"fermentation", r"ruminant", r"cattle", r"livestock"
        ]
        
        for pattern in chemical_patterns:
            if pattern in text:
                concepts.append(pattern)
        
        # Extract specific molecules/compounds mentioned
        import re
        compound_matches = re.findall(r"[A-Z][a-z]*(?:-[A-Z][a-z]*)*", doc.page_content)
        concepts.extend([m.lower() for m in compound_matches if len(m) > 3])
        
        return list(set(concepts))
    
    def add_document_concepts(self, doc: Document):
        """Add concepts from a document to the knowledge graph."""
        concepts = self.extract_concepts(doc)
        doc_id = doc.metadata.get("doc_id")
        
        for concept in concepts:
            if concept not in self.concepts:
                self.concepts[concept] = {"papers": [], "relations": []}
            
            if doc_id not in self.concepts[concept]["papers"]:
                self.concepts[concept]["papers"].append(doc_id)
    
    def find_related_concepts(self, concept: str, max_related: int = 10) -> List[Tuple[str, int]]:
        """
        Find concepts related to the given concept.
        
        Args:
            concept: Concept to find relations for
            max_related: Maximum number of related concepts
        
        Returns:
            List of (related_concept, co_occurrence_count) tuples
        """
        if concept not in self.concepts:
            return []
        
        concept_papers = set(self.concepts[concept]["papers"])
        related_counts = {}
        
        for other_concept, data in self.concepts.items():
            if other_concept == concept:
                continue
            
            other_papers = set(data["papers"])
            overlap = len(concept_papers.intersection(other_papers))
            
            if overlap > 0:
                related_counts[other_concept] = overlap
        
        # Sort by co-occurrence count
        sorted_related = sorted(related_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_related[:max_related]
    
    def get_concept_network(self, core_concepts: List[str]) -> Dict[str, Any]:
        """
        Get a network view of concepts and their relationships.
        
        Args:
            core_concepts: List of core concepts to include
        
        Returns:
            Dictionary representing the concept network
        """
        network = {
            "nodes": [],
            "edges": []
        }
        
        # Add core concept nodes
        for concept in core_concepts:
            if concept in self.concepts:
                network["nodes"].append({
                    "id": concept,
                    "label": concept,
                    "paper_count": len(self.concepts[concept]["papers"]),
                    "type": "core"
                })
        
        # Add related concepts and edges
        added_concepts = set(core_concepts)
        
        for concept in core_concepts:
            if concept in self.concepts:
                related = self.find_related_concepts(concept, max_related=5)
                
                for related_concept, co_occurrence in related:
                    if related_concept not in added_concepts:
                        network["nodes"].append({
                            "id": related_concept,
                            "label": related_concept,
                            "paper_count": len(self.concepts[related_concept]["papers"]),
                            "type": "related"
                        })
                        added_concepts.add(related_concept)
                    
                    # Add edge
                    network["edges"].append({
                        "source": concept,
                        "target": related_concept,
                        "weight": co_occurrence,
                        "type": "co_occurrence"
                    })
        
        return network