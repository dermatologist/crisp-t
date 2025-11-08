**Example: Graph Generation Workflow**
```bash
# Step 1: Load corpus with documents that have keywords
crispt --inp crisp_input --graph --out crisp_input

# Step 2: Visualize the graph (all node types)
crispviz --inp crisp_input --out visualizations --graph

# Step 3: Visualize only documents and keywords
crispviz --inp crisp_input --out visualizations --graph --graph-nodes document,keyword

# Step 4: Try different graph layouts
crispviz --inp crisp_input --out visualizations --graph --graph-layout circular
```

**About `--graph-nodes`:**

The `--graph-nodes` option allows you to filter which node types are included in the graph visualization. For example, to show only documents and keywords, use:

```bash
crispviz --inp crisp_input --out visualizations --graph --graph-nodes document,keyword
```

Valid node types: `document`, `keyword`, `cluster`, `metadata`. If omitted or set to `all`, all node types are included. Edges are only shown if both endpoints are present in the filtered node set.

The graph visualization shows:
- **Documents** (red nodes): Your corpus documents
- **Keywords** (teal nodes): Keywords extracted from documents
- **Clusters** (light green nodes): Document clusters (if clustering analysis was performed)
- **Metadata** (yellow nodes): Metadata from DataFrame (if present with aligning ID field)

**Note**: If documents don't have keywords assigned, run keyword assignment first using text analysis features before generating the graph.
