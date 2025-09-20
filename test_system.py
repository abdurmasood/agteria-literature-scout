#!/usr/bin/env python3
"""
Test script for Agteria Literature Scout

This script validates the core functionality of the Literature Scout system.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration and API keys."""
    print("🔧 Testing Configuration...")
    
    # Test API key validation
    keys_valid = Config.validate_keys()
    if keys_valid:
        print("✅ API keys configured correctly")
    else:
        print("❌ Missing required API keys")
        return False
    
    # Test configuration values
    assert Config.DEFAULT_MODEL, "Default model not set"
    assert Config.CHUNK_SIZE > 0, "Invalid chunk size"
    assert Config.MAX_ARXIV_RESULTS > 0, "Invalid ArXiv results limit"
    
    print("✅ Configuration tests passed")
    return True

def test_search_tools():
    """Test search tools functionality."""
    print("\n🔍 Testing Search Tools...")
    
    try:
        from src.tools.search_tools import create_langchain_search_tools, ArxivSearchTool, PubMedSearchTool
        
        # Test tool creation
        tools = create_langchain_search_tools()
        assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}"
        print(f"✅ Created {len(tools)} search tools")
        
        # Test ArXiv search
        arxiv_tool = ArxivSearchTool()
        arxiv_results = arxiv_tool.search("methane cattle", category_filter=None)
        print(f"✅ ArXiv search returned {len(arxiv_results)} results")
        
        # Test PubMed search (basic functionality)
        pubmed_tool = PubMedSearchTool()
        # Note: PubMed requires network connection and proper setup
        print("✅ PubMed tool initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Search tools test failed: {e}")
        return False

def test_analysis_tools():
    """Test analysis tools functionality."""
    print("\n🧪 Testing Analysis Tools...")
    
    try:
        from src.tools.analysis_tools import create_langchain_analysis_tools, PaperAnalyzer, RelevanceScorer
        
        # Test tool creation
        tools = create_langchain_analysis_tools()
        assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}"
        print(f"✅ Created {len(tools)} analysis tools")
        
        # Test paper analyzer
        analyzer = PaperAnalyzer()
        test_content = "This paper discusses novel enzyme inhibitors for methane reduction in cattle."
        analysis = analyzer.analyze_paper(test_content, "methane reduction")
        
        assert "analyzed_at" in analysis, "Analysis missing timestamp"
        print("✅ Paper analysis completed")
        
        # Test relevance scorer
        scorer = RelevanceScorer()
        score = scorer.score_paper(
            "Novel Methane Inhibitors for Livestock",
            "This study presents new compounds that reduce methane emissions in cattle."
        )
        
        assert "relevance_score" in score, "Score missing relevance_score"
        assert 0 <= score["relevance_score"] <= 10, "Invalid relevance score"
        print(f"✅ Relevance scoring completed (score: {score['relevance_score']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis tools test failed: {e}")
        return False

def test_hypothesis_tools():
    """Test hypothesis generation tools."""
    print("\n💡 Testing Hypothesis Tools...")
    
    try:
        from src.tools.hypothesis_tools import create_langchain_hypothesis_tools, HypothesisGenerator
        
        # Test tool creation
        tools = create_langchain_hypothesis_tools()
        assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}"
        print(f"✅ Created {len(tools)} hypothesis tools")
        
        # Test hypothesis generator
        generator = HypothesisGenerator()
        hypotheses = generator.generate_cross_domain_hypotheses(
            "marine biology", "algae produce enzyme inhibitors",
            "agriculture", "cattle produce methane",
            "methane reduction"
        )
        
        assert isinstance(hypotheses, list), "Hypotheses should be a list"
        print(f"✅ Generated {len(hypotheses)} cross-domain hypotheses")
        
        return True
        
    except Exception as e:
        print(f"❌ Hypothesis tools test failed: {e}")
        return False

def test_document_processor():
    """Test document processing functionality."""
    print("\n📄 Testing Document Processor...")
    
    try:
        from src.processors.document_processor import DocumentProcessor, DuplicateDetector, ContentValidator
        
        # Test processor creation
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Test text processing
        test_metadata = {
            "title": "Test Paper",
            "authors": ["Test Author"],
            "source": "test"
        }
        
        chunks = processor.process_text_content("This is a test document for processing.", test_metadata)
        assert len(chunks) >= 1, "Should produce at least one chunk"
        print(f"✅ Text processing produced {len(chunks)} chunks")
        
        # Test duplicate detector
        detector = DuplicateDetector()
        print("✅ Duplicate detector initialized")
        
        # Test content validator
        validator = ContentValidator()
        is_valid, issues = validator.is_valid_document(chunks[0])
        print(f"✅ Content validation: valid={is_valid}, issues={len(issues)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_memory_system():
    """Test memory system functionality."""
    print("\n💾 Testing Memory System...")
    
    try:
        from src.memory.research_memory import ResearchMemory, KnowledgeGraph
        from langchain.schema import Document
        
        # Test memory initialization
        memory = ResearchMemory()
        print("✅ Research memory initialized")
        
        # Test document addition
        test_doc = Document(
            page_content="Test paper about methane reduction in cattle.",
            metadata={
                "title": "Test Paper",
                "document_id": "test_001",
                "source": "test"
            }
        )
        
        # Note: This would require valid API keys for embeddings
        # doc_ids = memory.add_papers([test_doc])
        # print(f"✅ Added {len(doc_ids)} documents to memory")
        
        # Test knowledge graph
        kg = KnowledgeGraph(memory)
        concepts = kg.extract_concepts(test_doc)
        print(f"✅ Extracted {len(concepts)} concepts: {concepts}")
        
        # Test memory stats (without adding documents)
        stats = memory.get_memory_stats()
        assert "total_documents" in stats, "Stats missing document count"
        print(f"✅ Memory stats: {stats['total_documents']} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False

def test_report_generator():
    """Test report generation functionality."""
    print("\n📊 Testing Report Generator...")
    
    try:
        from src.utils.report_generator import ReportGenerator
        
        # Test generator initialization
        generator = ReportGenerator()
        print("✅ Report generator initialized")
        
        # Test data preparation
        test_scan_results = {
            "scan_date": "2024-01-01",
            "queries_processed": 5,
            "novel_discoveries": ["Discovery 1", "Discovery 2"],
            "generated_hypotheses": ["Hypothesis 1", "Hypothesis 2"],
            "results": []
        }
        
        # Test digest data preparation
        digest_data = generator._prepare_daily_digest_data(test_scan_results)
        assert "date" in digest_data, "Digest data missing date"
        assert "statistics" in digest_data, "Digest data missing statistics"
        print("✅ Daily digest data preparation successful")
        
        # Test markdown generation
        markdown_content = generator._generate_markdown_digest(digest_data)
        assert len(markdown_content) > 100, "Generated markdown too short"
        print("✅ Markdown report generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generator test failed: {e}")
        return False

def test_literature_scout():
    """Test the main Literature Scout agent."""
    print("\n🤖 Testing Literature Scout Agent...")
    
    try:
        from src.agents.literature_scout import LiteratureScout
        
        # Test agent initialization
        scout = LiteratureScout(verbose=False)
        print("✅ Literature Scout agent initialized")
        
        # Test status
        status = scout.get_agent_status()
        assert status["agent_initialized"], "Agent not properly initialized"
        assert status["available_tools"] > 0, "No tools available"
        print(f"✅ Agent status: {status['available_tools']} tools available")
        
        # Test simple research (would require API keys for full functionality)
        # result = scout.conduct_research("test query", ["novel mechanisms"])
        # print("✅ Research functionality accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Literature Scout test failed: {e}")
        return False

def run_all_tests():
    """Run all system tests."""
    print("🧪 Starting Agteria Literature Scout System Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Search Tools", test_search_tools),
        ("Analysis Tools", test_analysis_tools),
        ("Hypothesis Tools", test_hypothesis_tools),
        ("Document Processor", test_document_processor),
        ("Memory System", test_memory_system),
        ("Report Generator", test_report_generator),
        ("Literature Scout Agent", test_literature_scout),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Literature Scout is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

def quick_functionality_test():
    """Quick test of core functionality without requiring API keys."""
    print("⚡ Quick Functionality Test (No API keys required)")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.config import Config
        from src.tools.search_tools import ArxivSearchTool
        from src.processors.document_processor import DocumentProcessor
        from src.memory.research_memory import ResearchMemory
        print("✅ All imports successful")
        
        # Test basic initialization
        print("🔧 Testing basic initialization...")
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Test text processing
        print("📄 Testing text processing...")
        test_doc = processor.process_text_content(
            "This is a test document about methane reduction.",
            {"title": "Test", "source": "test"}
        )
        assert len(test_doc) > 0, "Document processing failed"
        print(f"✅ Processed document into {len(test_doc)} chunks")
        
        print("\n🎉 Quick test passed! Core functionality is working.")
        print("💡 Run with valid API keys for full functionality testing.")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        success = quick_functionality_test()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)