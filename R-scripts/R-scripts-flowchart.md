# R Scripts Process Flow Chart

- View flowchart: go to https://mermaid.live and paste mermaid code below:

```mermaid
flowchart TD
    %% Input Data
    A[MeSH XML Data<br/>desc2025.xml] --> B[mesh_term_selector.R]
    
    %% Script 1: MeSH Term Processing
    B --> B1{Parse XML}
    B1 --> B2[Extract DescriptorName<br/>& TreeNumberList]
    B2 --> B3[Create Binary Flags:<br/>- Neoplasm C04<br/>- Nervous System C10]
    B3 --> B4[Calculate Tree Depth<br/>count decimal points]
    B4 --> B5[Save mesh_terms.Rdata]
    
    %% Script 2: Swiss Publication Analysis
    C[PubMed API] --> D[pubmed_affiliation_extract.R]
    D --> D1[Query Specific Diseases:<br/>Alzheimer, MS, Cancer, Stroke]
    D1 --> D2[Filter Switzerland Affiliation]
    D2 --> D3[Parse Author Affiliations]
    D3 --> D4[Flag Swiss Authors:<br/>- Any Swiss coauthor<br/>- First author Swiss<br/>- Last author Swiss]
    D4 --> D5[Generate Yearly Counts]
    
    %% Script 3: Comparative Analysis
    B5 --> E[swiss_publication_mesh_scanner.R]
    E --> E1[Filter MeSH terms:<br/>tree_step < 10<br/>neoplasm OR nervous]
    E1 --> E2[Query Each Term:<br/>2010-2015 vs 2020-2025]
    E2 --> E3[Count Swiss Publications<br/>for each period]
    E3 --> E4[Save publication counts<br/>CSV & Rdata]
    
    %% Script 4: Global Share Analysis
    B5 --> F[swiss_publication_mesh_shares.R]
    F --> F1[Filter MeSH terms:<br/>tree_step < mesh_depth<br/>neoplasm OR nervous]
    F1 --> F2[Loop Through:<br/>Each term × Each year 2000-2025]
    F2 --> F3[Query for Each Term/Year:<br/>- Swiss publications<br/>- Total global publications]
    F3 --> F4[Calculate Swiss Share<br/>swiss/total]
    F4 --> F5[Apply 5-year<br/>running average]
    F5 --> F6[Identify Top 5<br/>Increasing Terms]
    F6 --> F7[Generate Interactive<br/>Visualization]
    
    %% API Dependencies
    API[PubMed API Key<br/>Rate Limited] --> D
    API --> E2
    API --> F3
    
    %% Outputs
    B5 --> G[mesh_terms.Rdata]
    D5 --> H[Swiss Publication<br/>Analysis Results]
    E4 --> I[publication_counts_meshterm.csv<br/>publication_counts_meshterm.Rdata]
    F7 --> J[Interactive Plot:<br/>Swiss Research Trends<br/>Over Time]
    
    %% Data Flow Dependencies
    G -.-> E
    G -.-> F
    
    %% Styling
    classDef script fill:#e1f5fe
    classDef data fill:#f3e5f5
    classDef api fill:#fff3e0
    classDef output fill:#e8f5e8
    
    class B,D,E,F script
    class A,G,I,J data
    class C,API api
    class H output
```

## Key Process Components:

### 1. **Data Preparation Phase** (`mesh_term_selector.R`)
- **Input**: MeSH XML descriptor file
- **Process**: Parse → Extract → Categorize → Calculate hierarchy
- **Output**: Structured MeSH term database

### 2. **Swiss Research Analysis** (`pubmed_affiliation_extract.R`)
- **Input**: PubMed API + specific disease queries
- **Process**: Query → Filter affiliations → Parse authors → Flag Swiss involvement
- **Output**: Swiss authorship patterns by year

### 3. **Comparative Time Analysis** (`swiss_publication_mesh_scanner.R`)
- **Dependencies**: Uses MeSH terms from Script 1
- **Process**: Filter terms → Query two time periods → Count publications
- **Output**: Before/after comparison data

### 4. **Global Share Tracking** (`swiss_publication_mesh_shares.R`)
- **Dependencies**: Uses MeSH terms from Script 1
- **Process**: Year-by-year queries → Calculate shares → Smooth trends → Visualize
- **Output**: Interactive trend visualization

## Critical Dependencies:
- **Sequential Dependency**: Script 1 must run first (creates MeSH term database)
- **API Dependency**: Scripts 2-4 depend on PubMed API access
- **Rate Limiting**: API calls are sequential to avoid rate limits

## Data Flow Pattern:
1. **Prepare** → Process MeSH taxonomy
2. **Query** → Extract publication data from PubMed
3. **Analyze** → Calculate metrics and trends
4. **Visualize** → Generate insights and plots