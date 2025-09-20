#!/usr/bin/env python3
"""
Agteria Literature Scout - CLI Interface

A command-line interface for the intelligent literature discovery system.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.agents.literature_scout import LiteratureScout
from src.utils.report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('literature_scout.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiteratureScoutCLI:
    """Command-line interface for Literature Scout."""
    
    def __init__(self):
        self.scout = None
        self.report_generator = None
    
    def initialize(self) -> bool:
        """Initialize the Literature Scout agent."""
        try:
            # Validate API keys
            if not Config.validate_keys():
                logger.error("Missing required API keys. Please check your .env file.")
                return False
            
            # Initialize components
            self.scout = LiteratureScout(verbose=True)
            self.report_generator = ReportGenerator()
            
            logger.info("Literature Scout initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Literature Scout: {e}")
            return False
    
    def search_research(self, query: str, focus_areas: Optional[List[str]] = None) -> None:
        """Conduct research on a query."""
        if not self.scout:
            logger.error("Literature Scout not initialized")
            return
        
        try:
            print(f"\n🔍 Researching: {query}")
            print("=" * 50)
            
            result = self.scout.conduct_research(query, focus_areas)
            
            # Display results
            print(f"\n📊 Research Results:")
            print(f"Query: {result['query']}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Papers Found: {result.get('papers_found', 0)}")
            
            if result.get('novel_insights'):
                print(f"\n💡 Novel Insights ({len(result['novel_insights'])}):")
                for i, insight in enumerate(result['novel_insights'], 1):
                    print(f"{i}. {insight}")
            
            if result.get('hypotheses'):
                print(f"\n🧪 Generated Hypotheses ({len(result['hypotheses'])}):")
                for i, hypothesis in enumerate(result['hypotheses'], 1):
                    print(f"{i}. {hypothesis}")
            
            if result.get('next_steps'):
                print(f"\n📋 Next Steps ({len(result['next_steps'])}):")
                for i, step in enumerate(result['next_steps'], 1):
                    print(f"{i}. {step}")
            
            print(f"\n📄 Full Response:")
            print(result['response'])
            
            # Save detailed results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"research_result_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            print(f"❌ Research failed: {e}")
    
    def daily_scan(self, custom_queries: Optional[List[str]] = None, 
                   generate_report: bool = True) -> None:
        """Perform daily research scan."""
        if not self.scout:
            logger.error("Literature Scout not initialized")
            return
        
        try:
            print("\n🌅 Starting Daily Research Scan...")
            print("=" * 40)
            
            scan_results = self.scout.daily_research_scan(custom_queries)
            
            # Display scan summary
            print(f"\n📊 Scan Summary:")
            print(f"Date: {scan_results['scan_date']}")
            print(f"Queries Processed: {scan_results['queries_processed']}")
            print(f"Novel Discoveries: {len(scan_results.get('novel_discoveries', []))}")
            print(f"Generated Hypotheses: {len(scan_results.get('generated_hypotheses', []))}")
            
            # Show top discoveries
            discoveries = scan_results.get('novel_discoveries', [])
            if discoveries:
                print(f"\n🔬 Top Novel Discoveries:")
                for i, discovery in enumerate(discoveries[:3], 1):
                    print(f"{i}. {discovery}")
            
            # Show top hypotheses
            hypotheses = scan_results.get('generated_hypotheses', [])
            if hypotheses:
                print(f"\n💡 Top Generated Hypotheses:")
                for i, hypothesis in enumerate(hypotheses[:3], 1):
                    print(f"{i}. {hypothesis}")
            
            # Generate report if requested
            if generate_report and self.report_generator:
                report_path = self.report_generator.generate_daily_digest(scan_results)
                if report_path:
                    print(f"\n📄 Daily digest report generated: {report_path}")
            
            # Save raw results
            timestamp = datetime.now().strftime("%Y%m%d")
            output_file = f"daily_scan_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(scan_results, f, indent=2)
            
            print(f"\n💾 Scan results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error during daily scan: {e}")
            print(f"❌ Daily scan failed: {e}")
    
    def analyze_breakthrough(self, findings: str) -> None:
        """Analyze breakthrough potential of findings."""
        if not self.scout:
            logger.error("Literature Scout not initialized")
            return
        
        try:
            print(f"\n🚀 Analyzing Breakthrough Potential...")
            print("=" * 40)
            
            result = self.scout.analyze_breakthrough_potential(findings)
            
            print(f"\n📊 Breakthrough Analysis:")
            print(result.get('response', 'No analysis available'))
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"breakthrough_analysis_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error during breakthrough analysis: {e}")
            print(f"❌ Breakthrough analysis failed: {e}")
    
    def explore_gaps(self, research_area: str) -> None:
        """Explore research gaps in an area."""
        if not self.scout:
            logger.error("Literature Scout not initialized")
            return
        
        try:
            print(f"\n🔍 Exploring Research Gaps in: {research_area}")
            print("=" * 50)
            
            result = self.scout.explore_research_gaps(research_area)
            
            print(f"\n📊 Gap Analysis:")
            print(result.get('response', 'No analysis available'))
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_area = "".join(c for c in research_area if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_area = safe_area.replace(' ', '_')[:30]
            output_file = f"gap_analysis_{safe_area}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Gap analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error during gap exploration: {e}")
            print(f"❌ Gap exploration failed: {e}")
    
    def track_competitors(self, competitors: List[str]) -> None:
        """Track competitor research."""
        if not self.scout:
            logger.error("Literature Scout not initialized")
            return
        
        try:
            print(f"\n👥 Tracking Competitor Research...")
            print(f"Competitors: {', '.join(competitors)}")
            print("=" * 40)
            
            intelligence = self.scout.track_competitor_research(competitors)
            
            print(f"\n📊 Competitive Intelligence Summary:")
            print(f"Competitors Analyzed: {len(intelligence.get('competitors_analyzed', []))}")
            print(f"Findings: {len(intelligence.get('findings', []))}")
            
            # Show key findings
            findings = intelligence.get('findings', [])
            if findings:
                print(f"\n🔍 Key Findings:")
                for i, finding in enumerate(findings[:3], 1):
                    query = finding.get('query', 'Unknown')
                    response = finding.get('response', 'No response')[:200] + "..."
                    print(f"{i}. Query: {query}")
                    print(f"   Finding: {response}\n")
            
            # Generate report
            if self.report_generator:
                report_path = self.report_generator.generate_competitive_intelligence(intelligence)
                if report_path:
                    print(f"\n📄 Competitive intelligence report: {report_path}")
            
            # Save raw results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"competitive_intelligence_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(intelligence, f, indent=2)
            
            print(f"\n💾 Intelligence data saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error tracking competitors: {e}")
            print(f"❌ Competitor tracking failed: {e}")
    
    def show_status(self) -> None:
        """Show current status of the Literature Scout."""
        if not self.scout:
            print("❌ Literature Scout not initialized")
            return
        
        try:
            status = self.scout.get_agent_status()
            
            print("\n📊 Literature Scout Status")
            print("=" * 30)
            print(f"Agent Initialized: ✅" if status['agent_initialized'] else "❌")
            print(f"Current Focus: {status.get('current_focus', 'None')}")
            print(f"Papers in Memory: {status.get('papers_in_memory', 0)}")
            print(f"Recent Discoveries: {status.get('recent_discoveries', 0)}")
            print(f"Available Tools: {status.get('available_tools', 0)}")
            print(f"Memory Health: {status.get('memory_health', 'unknown')}")
            
            # Show session stats
            session_stats = status.get('session_stats', {})
            if session_stats:
                print(f"\n📈 Session Statistics:")
                for key, value in session_stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            print(f"❌ Failed to get status: {e}")
    
    def interactive_mode(self) -> None:
        """Start interactive mode."""
        print("\n🤖 Literature Scout - Interactive Mode")
        print("=" * 40)
        print("Commands:")
        print("  research <query> - Conduct research")
        print("  scan - Perform daily scan")
        print("  breakthrough <findings> - Analyze breakthrough potential")
        print("  gaps <area> - Explore research gaps")
        print("  competitors <company1,company2> - Track competitors")
        print("  status - Show current status")
        print("  help - Show this help")
        print("  quit - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input("📝 Enter command: ").strip()
                
                if not command:
                    continue
                
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd == 'quit' or cmd == 'exit':
                    print("👋 Goodbye!")
                    break
                elif cmd == 'help':
                    print("\nCommands:")
                    print("  research <query> - Conduct research")
                    print("  scan - Perform daily scan")
                    print("  breakthrough <findings> - Analyze breakthrough potential")
                    print("  gaps <area> - Explore research gaps")
                    print("  competitors <company1,company2> - Track competitors")
                    print("  status - Show current status")
                    print("  help - Show this help")
                    print("  quit - Exit interactive mode")
                elif cmd == 'research':
                    if args:
                        self.search_research(args)
                    else:
                        print("❌ Please provide a research query")
                elif cmd == 'scan':
                    self.daily_scan()
                elif cmd == 'breakthrough':
                    if args:
                        self.analyze_breakthrough(args)
                    else:
                        print("❌ Please provide research findings to analyze")
                elif cmd == 'gaps':
                    if args:
                        self.explore_gaps(args)
                    else:
                        print("❌ Please provide a research area")
                elif cmd == 'competitors':
                    if args:
                        competitors = [c.strip() for c in args.split(',')]
                        self.track_competitors(competitors)
                    else:
                        print("❌ Please provide competitor names (comma-separated)")
                elif cmd == 'status':
                    self.show_status()
                else:
                    print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                
                print()  # Add spacing between commands
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agteria Literature Scout - Intelligent Research Discovery System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Research command
    research_parser = subparsers.add_parser('research', help='Conduct research on a query')
    research_parser.add_argument('query', help='Research query')
    research_parser.add_argument('--focus', nargs='+', help='Focus areas')
    
    # Daily scan command
    scan_parser = subparsers.add_parser('scan', help='Perform daily research scan')
    scan_parser.add_argument('--queries', nargs='+', help='Custom queries to include')
    scan_parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    
    # Breakthrough analysis command
    breakthrough_parser = subparsers.add_parser('breakthrough', help='Analyze breakthrough potential')
    breakthrough_parser.add_argument('findings', help='Research findings to analyze')
    
    # Gap exploration command
    gaps_parser = subparsers.add_parser('gaps', help='Explore research gaps')
    gaps_parser.add_argument('area', help='Research area to explore')
    
    # Competitor tracking command
    competitors_parser = subparsers.add_parser('competitors', help='Track competitor research')
    competitors_parser.add_argument('companies', nargs='+', help='Competitor company names')
    
    # Status command
    subparsers.add_parser('status', help='Show Literature Scout status')
    
    # Interactive mode command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup Literature Scout environment')
    
    args = parser.parse_args()
    
    # Handle setup command
    if args.command == 'setup':
        print("🛠️  Setting up Literature Scout...")
        
        # Check if .env file exists
        env_file = Path('.env')
        if not env_file.exists():
            print("📝 Creating .env file template...")
            template_content = Config.create_env_template()
            with open('.env', 'w') as f:
                f.write(template_content)
            print("✅ .env template created. Please fill in your API keys.")
        else:
            print("✅ .env file already exists.")
        
        # Create reports directory
        reports_dir = Path(Config.REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Reports directory created: {reports_dir}")
        
        # Create data directory
        data_dir = Path(Config.VECTOR_DB_PATH).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Data directory created: {data_dir}")
        
        print("\n🎉 Setup complete! You can now run the Literature Scout.")
        print("💡 Don't forget to add your API keys to the .env file.")
        return
    
    # Initialize CLI
    cli = LiteratureScoutCLI()
    
    if not cli.initialize():
        print("❌ Failed to initialize Literature Scout. Check logs for details.")
        print("💡 Try running 'python main.py setup' first.")
        return
    
    # Handle commands
    if args.command == 'research':
        cli.search_research(args.query, args.focus)
    
    elif args.command == 'scan':
        cli.daily_scan(args.queries, not args.no_report)
    
    elif args.command == 'breakthrough':
        cli.analyze_breakthrough(args.findings)
    
    elif args.command == 'gaps':
        cli.explore_gaps(args.area)
    
    elif args.command == 'competitors':
        cli.track_competitors(args.companies)
    
    elif args.command == 'status':
        cli.show_status()
    
    elif args.command == 'interactive':
        cli.interactive_mode()
    
    else:
        # No command provided, show help
        parser.print_help()
        print("\n💡 Try 'python main.py interactive' for interactive mode.")
        print("💡 Try 'python main.py setup' if this is your first time.")

if __name__ == "__main__":
    main()