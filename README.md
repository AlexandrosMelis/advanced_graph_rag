# Advanced Graph Rag 
AUTH Diploma Thesis Project

## Project's folder structure
advanced_graph_rag/
├── README.md                  # High-level overview of the project
├── LICENSE                    # (Optional) Add a license for your code
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies (alternatively environment.yml for conda)
│
├── data/
│   ├── raw/                   # Original/untouched data, e.g. PubMed articles
│   ├── processed/             # Preprocessed data ready for modeling
│   ├── intermediate/               # Intermediate data outputs
│   └── external/              # Any external datasets or QA benchmark data
│
├── notebooks/
│   ├── exploratory/           # Jupyter notebooks for initial data exploration
│   ├── experiments/           # Jupyter notebooks used for testing ideas, small-scale experiments
│   └── reports/               # Analysis and visualizations that may feed into your thesis or presentations
│
├── src/
│   ├── data_collection/       
│   │   ├── pubmed_scraper.py      # Scripts for downloading PubMed publications
│   │   └── ...
│   │
│   ├── data_preprocessing/
│   │   ├── text_cleaning.py       # Tokenization, cleaning scripts, etc.
│   │   └── ...
│   │
│   ├── knowledge_graph/           # Scripts for KG creation and publication ingestion
│   │   ├── build_graph.py
│   │   ├── ingestion_utils.py
│   │   └── ...
│   │
│   ├── embeddings/                # Scripts / modules for experimenting with graph embeddings
│   │   ├── train_embeddings.py
│   │   ├── embedding_models/
│   │   └── ...
│   │
│   ├── llm/                       # Scripts for experimenting with different LLMs
│   │   ├── model_integration.py   # Integrating your KG-based retrieval with LLM input
│   │   ├── llm_utils.py
│   │   └── ...
│   │
│   └── evaluation/                # Evaluation pipeline on QA benchmark datasets
│       ├── metrics.py
│       ├── run_evaluation.py
│       └── ...
│
├── scripts/
│   ├── run_data_pipeline.sh       # Or .py script orchestrating data collection + preprocessing
│   ├── evaluate.sh                # Script to run full evaluation on QA datasets
│   └── ...
│
└── docs/
    ├── thesis/                    # Drafts, diagrams, or LaTeX files for your thesis
    ├── system_design.md           # System architecture documentation
    └── references/                # Reference papers, PDFs, or other resources
