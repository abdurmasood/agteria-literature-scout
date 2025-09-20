#!/usr/bin/env python3
"""Simple test of ArXiv functionality."""

import arxiv

def test_arxiv():
    print("Testing ArXiv search...")
    
    try:
        # Create client
        client = arxiv.Client()
        
        # Search for papers
        search = arxiv.Search(
            query="methane cattle livestock",
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = list(client.results(search))
        
        print(f"Found {len(results)} papers:")
        for i, paper in enumerate(results, 1):
            print(f"{i}. {paper.title}")
            print(f"   Authors: {', '.join(str(author) for author in paper.authors[:2])}")
            print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
            print()
        
        print("✅ ArXiv search working!")
        return True
        
    except Exception as e:
        print(f"❌ ArXiv search failed: {e}")
        return False

if __name__ == "__main__":
    test_arxiv()