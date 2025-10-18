# CRISP-T Framework User Instructions

## Overview

CRISP-T (**CRoss** **I**ndustry **S**tandard **P**rocess for **T**riangulation) is a framework that integrates textual data (as a list of documents) and numeric data (as Pandas DataFrames) into structured classes that retain metadata from various analytical processes. Further, if the numeric and textual datasets share same id, or if the textual metadata contains keywords that match numeric column names; both datasets are filtered simultaneously, ensuring alignment and facilitating triangulation. 👉 [See Demo](/DEMO.md). This framework enables researchers to analyze qualitative and quantitative data using advanced NLP, machine learning, and statistical techniques. This is under active development; please [report any issues or feature requests on GitHub](https://github.com/dermatologist/crisp-t/issues).

## Recommended Sequence of Analysis

### Phase 1: Data Preparation and Exploration
Prepare your data by loading textual and numeric datasets. Clean the data to remove duplicates and handle missing values. Inspect the structure and metadata to ensure readiness for analysis. Most of the data cleaning can be done externally using your preferred tools.

### Phase 2: Descriptive Analysis
Perform initial exploration of the data. Generate coding dictionaries, analyze sentiment, and extract categories to understand the basic patterns and themes in the textual data.

### Phase 3: Advanced Pattern Discovery
Use topic modeling and clustering to uncover latent themes and groupings in the data. Apply association rule mining to identify relationships between variables and discover unexpected patterns.

### Phase 4: Predictive Modeling
Build predictive models using machine learning techniques such as regression, classification, and dimensionality reduction. Use these models to test hypotheses and understand the impact of different factors on outcomes. Note that you can export metadata from textual analyses to DataFrames for integration with numeric data during this phase.

### Phase 5: Validation and Triangulation
Validate findings by comparing results across different analytical methods. Use triangulation to link textual insights with numeric patterns and ensure theoretical coherence. Document relationships and generate reports to summarize the analysis.


## Quality Assurance Checklist
- [ ] Data cleaning (missing values, duplicates handled)
- [ ] Multiple analytical approaches applied to the same research question
- [ ] Model performance metrics documented
- [ ] Statistical significance of findings verified
- [ ] Theoretical coherence of results evaluated
- [ ] Findings triangulated across textual and numerical analyses
- [ ] Metadata preserved for reproducibility
- [ ] Results validated with domain expertise

This systematic approach ensures comprehensive analysis while maintaining theoretical rigor and methodological transparency.
