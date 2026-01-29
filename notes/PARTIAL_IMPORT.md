# Partial Data Import Feature

## Overview

The partial import feature allows users to limit the amount of data imported when using the `--source` flag. This is particularly useful for:

- **Testing with large datasets**: Import a small subset to verify your workflow before processing the full dataset
- **Quick exploration**: Get a sense of the data structure without waiting for full import
- **Development**: Work with manageable data sizes during development
- **Memory constraints**: Avoid memory issues when working with very large datasets

## Usage

### Limiting Text/PDF Files

Use the `--num` (or `-n`) flag with `--source` to limit the number of text and PDF files imported:

```bash
# Import only the first 10 text/PDF files
crisp --source my_data_folder --num 10 --out test_corpus

# Short form
crisp -s my_data_folder -n 5 -o test_corpus
```

### Limiting CSV Rows

Use the `--rec` (or `-r`) flag with `--source` to limit the number of CSV rows imported:

```bash
# Import only the first 100 rows from CSV files
crisp --source my_data_folder --rec 100 --out test_corpus

# Short form
crisp -s my_data_folder -r 50 -o test_corpus
```

### Combining Both Limits

You can use both flags together to limit both text files and CSV rows:

```bash
# Import 5 text/PDF files and 50 CSV rows
crisp --source my_data_folder --num 5 --rec 50 --out test_corpus

# Short form
crisp -s my_data_folder -n 5 -r 50 -o test_corpus
```

## Examples

### Example 1: Quick Data Exploration

You have a folder with 100 interview transcripts and a large CSV file with 10,000 survey responses. You want to quickly explore the data structure:

```bash
# Import just 3 interviews and 10 survey responses
crisp --source research_data --num 3 --rec 10

# This will show you:
# - The structure of the text documents
# - The columns in your CSV
# - How the data is organized
```

### Example 2: Testing Your Analysis Pipeline

Before running expensive analyses on your full dataset, test your pipeline:

```bash
# Import a small subset
crisp --source research_data --num 10 --rec 100 --out test_corpus

# Test your analysis commands
crisp --inp test_corpus --topics --sentiment --num 3

# Once verified, process the full dataset
crisp --source research_data --out full_corpus
crisp --inp full_corpus --topics --sentiment --num 5
```

### Example 3: Working with Memory Constraints

Your dataset is too large to fit in memory, but you want to experiment with different analyses:

```bash
# Process the dataset in chunks by using different limits
crisp --source huge_dataset --num 20 --rec 500 --out chunk1
crisp --inp chunk1 --topics --out chunk1_analyzed

# Process another chunk (manually select different files or use scripts)
# Or use this as a way to estimate processing time and resource requirements
```

### Example 4: Comparing Subsets

Compare different portions of your dataset:

```bash
# Create multiple small test corpora
crisp --source research_data --num 5 --rec 50 --out early_sample
crisp --source research_data --num 5 --rec 50 --out later_sample

# Compare the characteristics
crisp --inp early_sample --sentiment
crisp --inp later_sample --sentiment
```

## Important Notes

### Default Behavior

When `--num` and `--rec` are **not** specified with `--source`, the system imports **all** available files. The default values (3) only apply when used with analysis commands.

### Backward Compatibility

The `--num` and `--rec` flags have dual purposes:

1. **With `--source`**: Limits import of text/PDF files and CSV rows
2. **With analysis commands**: Controls analysis parameters (e.g., number of topics, clusters)

This design maintains backward compatibility with existing scripts while adding the new import limiting functionality.

### File Selection

Files are selected in the order returned by the filesystem (typically alphabetical). Text files are processed first, followed by PDF files, until the limit is reached.

### Multiple Sources

When using `--sources` (multiple source directories), the limits apply to **each** source individually:

```bash
# Import 5 files from EACH source
crisp --sources folder1 folder2 folder3 --num 5 --rec 100
```

## Technical Details

### Implementation

- **Text/PDF files**: The system counts text and PDF files together. If you set `--num 10`, it will import up to 10 total files (text + PDF combined).
- **CSV rows**: The system uses pandas `nrows` parameter for efficient reading, loading only the specified number of rows.
- **CSV text columns**: When importing CSV with `--unstructured` columns, the row limit applies to the CSV before documents are created.

### Performance

Limiting imports significantly reduces:
- I/O time (reading fewer files)
- Memory usage (storing less data)
- Processing time (analyzing smaller datasets)

### Limitations

- The feature does not provide smart sampling (e.g., random sampling). Files are selected in filesystem order.
- Once imported, you cannot easily "add more" files without re-importing the entire source.
- Different file types (txt vs pdf) are not limited separately.

## Best Practices

1. **Start small**: When working with a new dataset, start with small limits to understand the structure
2. **Test your pipeline**: Use limits to test your analysis pipeline before running on full data
3. **Document your approach**: When using limits, document which files/rows you imported for reproducibility
4. **Progressive scaling**: Gradually increase limits to find the right balance between speed and comprehensiveness

## See Also

- [INSTRUCTION.md](INSTRUCTION.md) - Complete CRISP-T functionality reference
- [DEMO.md](DEMO.md) - Step-by-step usage examples
- [README.md](../README.md) - Main documentation
