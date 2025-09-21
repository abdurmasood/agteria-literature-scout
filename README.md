  <picture>
    <!-- <source srcset="./banner-dark.png" media="(prefers-color-scheme: dark)"> -->
    <source srcset="./banner.png" media="(prefers-color-scheme: light)">
    <img src="./banner.png" alt="Agteria Background">
  </picture>
  <h2 align="center">
    Agteria Literature Scout
  </h2>
<p align="center">
  <strong>Intelligent Literature Discovery for Climate Technology</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Beta-orange.svg" alt="Status">
</p>

## About the Project
An AI-powered research assistant designed to accelerate the discovery of novel methane reduction solutions through automated literature analysis, cross-domain hypothesis generation, and breakthrough identification.

## Features

### Intelligent Literature Search
- **Multi-Database Integration**: Searches ArXiv, PubMed, and web sources simultaneously
- **Semantic Search**: Uses AI embeddings for context-aware paper discovery
- **Cross-Domain Discovery**: Finds unexpected connections across research fields
- **Real-time Monitoring**: Automated daily scans for new publications

### Advanced Analysis
- **Paper Analysis**: Extracts key findings, methodologies, and molecular targets
- **Hypothesis Generation**: Creates novel, testable research hypotheses
- **Breakthrough Assessment**: Evaluates commercial and technical potential
- **Gap Identification**: Discovers unexplored research opportunities

### Persistent Memory
- **Vector Database**: ChromaDB for semantic paper storage and retrieval
- **Knowledge Graph**: Tracks concepts and relationships across papers
- **Duplicate Detection**: Prevents redundant analysis
- **Quality Scoring**: Ranks papers by relevance and reliability

### Comprehensive Reporting
- **Daily Digests**: Automated research summaries
- **Hypothesis Reports**: Prioritized research directions
- **Competitive Intelligence**: Competitor research tracking
- **Trend Analysis**: Research momentum and topic evolution

### Multiple Interfaces
- **Command Line Interface**: Full-featured CLI for power users
- **Web Interface**: Streamlit-based dashboard with visualizations
- **Interactive Mode**: Conversational research assistant

## Expected Impact

This system should:
- **Reduce literature review time by 70-80%**
- **Increase research coverage by 10x**
- **Discover novel research directions** you might miss manually
- **Monitor competitors** and market developments automatically
- **Generate hypotheses** for breakthrough innovations
- **Identify collaboration opportunities** across domains

## Research Capabilities

### Daily Research Workflow
1. **Morning Scan**: Automated discovery of new papers
2. **Analysis**: AI extracts key findings and methodologies
3. **Hypothesis Generation**: Creates novel research directions
4. **Report**: Summarized insights delivered via dashboard
5. **Memory Update**: Papers stored for future reference

### Cross-Domain Innovation Examples
- **Marine Biology → Agriculture**: "How do algae enzyme systems work in ruminants?"
- **Pharmaceuticals → Feed Additives**: "Can drug delivery methods improve feed additive efficacy?"
- **Materials Science → Livestock**: "How can nanotechnology enhance methane inhibitors?"

## Example Results

### Recent Discoveries
- Found 15 novel methane inhibitor papers in last scan
- Identified 3 cross-domain opportunities from marine biology
- Generated 8 testable hypotheses for enzyme targets
- Tracked 12 competitor research developments

### Sample Hypothesis
*"Based on deep-sea bacterial methane processing mechanisms, we hypothesize that targeting specific archaeal enzymes with marine-derived inhibitors could reduce livestock methane emissions by 30-50% while maintaining feed efficiency."*

## Usage Examples

### Command Line Interface

```bash
# Conduct research on a topic
python main.py research "novel methane inhibitors for cattle"

# Run daily research scan
python main.py scan

# Interactive mode
python main.py interactive

# Analyze breakthrough potential
python main.py breakthrough "New enzyme inhibitor reduces methane by 40%"

# Explore research gaps
python main.py gaps "ruminant fermentation control"

# Track competitors
python main.py competitors DSM Cargill Alltech
```

### Web Interface

```bash
# Launch the dashboard
streamlit run app.py
# Open browser to http://localhost:8501
```

### Python API

```python
from src.agents.literature_scout import LiteratureScout

# Initialize the scout
scout = LiteratureScout()

# Conduct research
result = scout.conduct_research(
    "methane reduction in livestock",
    focus_areas=["novel mechanisms", "practical applications"]
)

# Daily scan
scan_results = scout.daily_research_scan()

# Analyze breakthrough potential
analysis = scout.analyze_breakthrough_potential(
    "Novel enzyme inhibitor shows 50% methane reduction"
)
```

## Architecture

```
agteria-literature-scout/
├── src/
│   ├── agents/              # Main ReAct agent
│   ├── tools/               # Search, analysis, and hypothesis tools
│   ├── processors/          # Document processing
│   ├── memory/              # ChromaDB knowledge storage
│   ├── utils/               # Report generation utilities
│   └── config.py            # Configuration management
├── main.py                  # CLI interface
├── app.py                   # Streamlit web interface
└── README.md               # Documentation
```

### Core Components

- **Literature Scout Agent**: ReAct framework for reasoning and tool orchestration
- **Tool Ecosystem**: Search, analysis, and hypothesis generation tools
- **Memory System**: ChromaDB vector storage with semantic search
- **Report Generation**: Automated insights and trend analysis

---

## Configuration

### Environment Variables (.env)

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Enhanced web search
SERPER_API_KEY=your_serper_api_key_here

# Model Configuration (optional)
DEFAULT_MODEL=gpt-4-turbo-preview
TEMPERATURE=0.7
MAX_ARXIV_RESULTS=10
MAX_PUBMED_RESULTS=10
```

### Research Focus

The system is pre-configured for climate technology research:

```python
AGTERIA_KEYWORDS = [
    "methane inhibition",
    "cattle emissions", 
    "livestock greenhouse gas",
    "ruminant fermentation",
    "feed additive",
    "enzyme inhibitor"
]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Agteria Biotech** for the vision and mission
- **LangChain** for the agent framework
- **OpenAI** for the language model capabilities
- **ArXiv & PubMed** for scientific literature access
- **Open Source Community** for the foundational tools