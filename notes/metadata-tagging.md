**Automatic Metadata Tagging from Filenames:**
When importing text files (.txt) or PDFs, if a filename contains a dash (`-`) or underscore (`_`) character, the first part of the filename before the separator is automatically extracted and added as a `tag` in the document's metadata. This enables easy grouping and filtering of documents by file naming convention.

Examples:
- `interview-1.txt` → `metadata['tag'] = 'interview'`
- `report_2025.pdf` → `metadata['tag'] = 'report'`
- `research-data-analysis.txt` → `metadata['tag'] = 'research'`
- `document.txt` → No tag added (no separator found)

This feature is useful for:
- Organizing documents by category or type
- Filtering documents by tag using the `--filters` option (e.g., `--filters tag=interview`)
- Conducting comparative analysis across document groups