# CRISP-T Linkage Mechanisms: Deep Analysis

This document provides a detailed analysis of how various linkage mechanisms work in the CRISP-T framework, focusing on the logic in `src/crisp_t/helpers/analyzer.py`.

## 1. Implicit Linkages

### a. ID Linkage (Explicit Implementation)
- **Mechanism:**
  - When filtering using the special `id` filter (e.g., `id=123` or `id:123`), both documents and DataFrame rows are synchronized.
  - Documents are filtered by their `id` attribute.
  - DataFrame rows are filtered by the `id` column (if it exists).
  - This creates **synchronized filtering** where text and numeric data remain aligned.
- **Implementation:**
  - `_apply_regular_filters()` detects when `key.lower() == "id"`.
  - It filters documents via the text analyzer.
  - It also filters the DataFrame directly using `df[df["id"].astype(str) == str(value)]`.
  - Both are updated in-place to maintain alignment.
- **Example:**
  - `filters=["id=12345"]` will keep only documents with `id="12345"` AND DataFrame rows where `id_column == "12345"`.
- **Use Case:**
  - Ensures that text and numeric data remain synchronized when subsetting by document/row ID.
  - Enables implicit linkage without explicit link metadata.

### b. Metadata Linkage (Implicit)
- **Mechanism:**
  - If a document's metadata key matches a DataFrame column name, filtering on that key will affect both the documents and the DataFrame.
  - For example, filtering on `keywords=health` will keep only documents with `metadata['keywords'] == 'health'` and DataFrame rows where the `keywords` column matches.
- **Example:**
  - `filters=["keywords=health"]` will apply to both analyzers if both have the `keywords` field/column.
- **Use Case:**
  - Enables seamless triangulation between qualitative codes and quantitative variables.

## 2. Explicit Linkages

### a. Embedding Linkage (Explicit)
- **Mechanism:**
  - Uses semantic similarity (e.g., via sentence embeddings) to link documents to DataFrame rows.
  - Links are stored in `Document.metadata['embedding_links']` as a list of dicts, each with a `df_index` and similarity score.
- **Filter Directions:**
  - `embedding:text` (or legacy `=embedding`/`:embedding`): Filters DataFrame rows to those linked from the current set of documents.
  - `embedding:df`: Filters documents to those linked from the current set of DataFrame rows.
- **Implementation:**
  - `_apply_text_to_df_filter` collects all `df_index` values from the current documents' `embedding_links` and filters the DataFrame.
  - `_apply_df_to_text_filter` builds a reverse map from DataFrame indices to document IDs and filters documents accordingly.
- **Use Case:**
  - Enables semantic triangulation between qualitative and quantitative data, even when no direct ID or metadata match exists.

### b. Temporal Linkage (Explicit)
- **Mechanism:**
  - Uses temporal proximity (e.g., timestamps) to link documents to DataFrame rows.
  - Links are stored in `Document.metadata['temporal_links']` as a list of dicts, each with a `df_index` and time gap.
- **Filter Directions:**
  - `temporal:text` (or legacy `=temporal`/`:temporal`): Filters DataFrame rows to those linked from the current set of documents.
  - `temporal:df`: Filters documents to those linked from the current set of DataFrame rows.
- **Implementation:**
  - Same as embedding, but uses `temporal_links` instead of `embedding_links`.
- **Use Case:**
  - Enables time-based triangulation, e.g., linking interview responses to sensor data by timestamp.

## 3. Internal Logic in analyzer.py
- The `get_analyzers` function separates regular filters (key=value or key:value) from link filters (embedding/temporal, both directions).
- Regular filters are applied via `_apply_regular_filters()`:
  - **ID Filter**: When key is "id", special synchronized filtering is applied to both documents and DataFrame by ID.
  - **Other Filters**: Applied independently to each analyzer (documents and DataFrame).
- Link filters are routed to `_apply_link_filters`, which dispatches to the correct direction handler.
- The DataFrame and document list are updated in-place to reflect the filtered state.

## 4. Suggestions for Improvement
- **Unified Linkage API:** Consider a single function to handle all linkage types (ID, metadata, embedding, temporal) with a consistent interface.
- **Linkage Metadata:** Store linkage provenance (how/why a link was made) for better auditability.
- **Chained Filtering:** Allow chaining of multiple linkage types in a single call for advanced triangulation workflows.
- **Performance:** For large datasets, optimize reverse lookups and index-based filtering.
- **User Feedback:** Provide more detailed feedback on which documents/rows were filtered and why.

---

This document reflects the current state of linkage logic in CRISP-T as of January 2026. For further details, see the code in `src/crisp_t/helpers/analyzer.py` and related modules.
