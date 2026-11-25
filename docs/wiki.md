# GitHub Wiki Update Instructions

This document provides step-by-step instructions for updating the CRISP-T GitHub Wiki to reflect recent changes in the codebase.

## Overview

The GitHub Wiki for CRISP-T is hosted at: https://github.com/dermatologist/crisp-t/wiki

Wiki pages are maintained separately from the repository documentation and require manual updates through the GitHub interface.

---

## Recent Changes: Renaming `--cart` to `--apriori`

### Background

The `--cart` CLI flag has been renamed to `--apriori` to correctly reflect that it implements the **Apriori algorithm** for association rule mining, not CART (Classification and Regression Trees).

### Changes Made in Repository

1. **CLI Implementation** (`src/crisp_t/cli.py`)
   - Flag renamed from `--cart` to `--apriori`
   - Help text updated to remove "(CART)" reference
   - Parameter name changed from `cart` to `apriori`
   - All conditional checks updated
   - Output header changed from "Association Rules (CART)" to "Association Rules (Apriori)"

2. **Documentation** (`docs/cheatsheet.md`)
   - Table entry updated: `--cart` → `--apriori`

### Wiki Pages That Need Updating

#### 1. **Main Wiki Pages**

Check and update the following wiki pages if they exist:

- **Home Page** - If it contains CLI examples with `--cart`
- **CLI Reference** - Update all `--cart` references to `--apriori`
- **Getting Started** - Update any tutorial examples
- **Examples** - Update any code snippets

#### 2. **Specific Pages to Check**

Based on the repository structure, check these pages:

- **What's Brewing: Agent Harness** (already linked in README)
- **Topological Data Analysis** (already linked in README)
- Any other pages that mention association rules or the `--cart` flag

---

## Step-by-Step Wiki Update Process

### Method 1: Using GitHub Web Interface

1. **Navigate to the Wiki**
   - Go to: https://github.com/dermatologist/crisp-t/wiki
   - Click on the page you want to edit

2. **Edit the Page**
   - Click the **Edit** button (pencil icon) in the top right
   - Make your changes using GitHub's markdown editor

3. **Search for `--cart`**
   - Use your browser's find function (Ctrl+F / Cmd+F)
   - Search for `--cart` or `cart` flag references
   - Replace with `--apriori`

4. **Update Examples**
   - Change command examples:

     ```bash
     # OLD
     crisp --inp ./corpus --cart --outcome Gender

     # NEW
     crisp --inp ./corpus --apriori --outcome Gender
     ```

5. **Commit Changes**
   - Add a commit message: "Update --cart to --apriori flag"
   - Click **Save changes**

### Method 2: Using Git (Advanced)

If you have write access to the wiki repository:

1. **Clone the Wiki Repository**

   ```bash
   git clone git@github.com:dermatologist/crisp-t.wiki.git
   cd crisp-t.wiki
   ```

2. **Make Changes**
   - Edit pages locally
   - Replace all `--cart` with `--apriori`

3. **Push Changes**
   ```bash
   git add .
   git commit -m "Update --cart to --apriori flag throughout wiki"
   git push origin master
   ```

---

## Common Patterns to Update

### Command Examples

**Before:**

```bash
crisp --inp ./corpus --cart --outcome Gender
crisp --inp ./corpus --ml --cart --regression
```

**After:**

```bash
crisp --inp ./corpus --apriori --outcome Gender
crisp --inp ./corpus --ml --apriori --regression
```

### Descriptive Text

**Before:**

- "Use the `--cart` flag to generate association rules"
- "CART (Classification and Regression Trees) analysis"

**After:**

- "Use the `--apriori` flag to generate association rules"
- "Apriori algorithm for association rule mining"

### Help Text References

**Before:**

```
--cart    Generate association rules using the Apriori algorithm (CART).
```

**After:**

```
--apriori Generate association rules using the Apriori algorithm.
```

---

## Verification Checklist

After updating the wiki, verify:

- [ ] All `--cart` references replaced with `--apriori`
- [ ] Command examples use the new flag
- [ ] Descriptive text mentions "Apriori algorithm" not "CART"
- [ ] Screenshots (if any) updated to show new flag
- [ ] Links to examples still work
- [ ] No broken references to old flag

---

## Additional Resources

### Repository Documentation

These files in the repository are already updated and can serve as reference:

- `docs/cheatsheet.md` - CLI reference
- `README.md` - Main documentation
- `notes/INSTRUCTION.md` - Detailed instructions
- `docs/DEMO.md` - Usage examples

### Testing the Changes

After updating the wiki, test the new flag:

```bash
# Check help text
crisp --help | grep -i apriori

# Test with sample data
crisp --inp ./workspace/coffee_corpus --apriori --outcome Gender
```

---

## Maintenance Notes

### When to Update the Wiki

Update the wiki whenever:

- CLI flags are added, removed, or renamed
- Command behavior changes significantly
- New features are released
- Documentation examples are updated

### Wiki vs Repository Docs

**Repository Documentation** (`docs/`, `README.md`, etc.):

- Version controlled with code
- Updated via pull requests
- Automatically deployed with releases

**GitHub Wiki**:

- Separate repository (`.wiki`)
- Manual updates through web interface or git
- Better for tutorials and user guides
- Not versioned with releases

### Best Practices

1. **Keep in Sync**: Always update both repository docs and wiki together
2. **Test Examples**: Verify all command examples work after changes
3. **Clear Migration**: For breaking changes, provide migration instructions
4. **Version Notes**: Consider adding version numbers to wiki pages

---

## Contact

For questions about wiki updates or documentation:

- Open an issue in the main repository
- Contact the maintainers
- Submit a pull request with suggested changes

---

## Appendix: Full List of Changes

### Files Modified

1. `src/crisp_t/cli.py` - 9 occurrences updated
2. `docs/cheatsheet.md` - 1 occurrence updated

### Breaking Changes

- **Flag Renamed**: `--cart` → `--apriori`
- **No Backward Compatibility**: Old flag will not work
- **User Action Required**: Users must update their scripts

### Migration Guide for Users

**If you were using:**

```bash
crisp --inp ./corpus --cart --outcome Gender
```

**Update to:**

```bash
crisp --inp ./corpus --apriori --outcome Gender
```

**The functionality is identical** - only the flag name has changed to better reflect the underlying algorithm.

---

## Recent Changes: Clarifying `--cls` Dual-Algorithm Behavior

### Background

The `--cls` CLI flag runs **two classification algorithms sequentially**:

1. **Support Vector Machine (SVM)**
2. **Random Forest** (previously mislabeled as "Decision Tree")

This dual-algorithm approach enables triangulation by providing different perspectives on the same classification problem.

### Changes Made in Repository

1. **CLI Implementation** (`src/crisp_t/cli.py`)
   - Renamed "Decision Tree" to "Random Forest" in all output messages
   - Added summary header: "📊 Note: Running both SVM and Random Forest classifiers..."
   - Updated help text to clarify both algorithms run sequentially
   - Updated algorithm descriptions to accurately reflect Random Forest as "Tree-based ensemble classifier"

2. **Documentation** (`docs/wiki.md`)
   - Added comprehensive section explaining dual-algorithm approach

### Why Two Algorithms?

- **SVM**: Excellent for finding decision boundaries in high-dimensional data; provides confusion matrix
- **Random Forest**: Ensemble method using 100 decision trees; provides both confusion matrix and feature importance rankings

Using both algorithms allows researchers to:

- Validate findings across different methods (triangulation)
- Compare accuracy metrics
- Understand which features are most predictive (Random Forest)
- See decision boundaries in feature space (SVM)

### Wiki Pages That Need Updating

#### 1. **Create New Wiki Page: "Classification with --cls"**

Create a new wiki page with the following content:

```markdown
# Classification with --cls: Understanding SVM and Random Forest

## Overview

The `--cls` flag runs **two classification algorithms** sequentially:

1. **Support Vector Machine (SVM)**
2. **Random Forest**

Both algorithms analyze the same data but provide different insights.

## Why Two Algorithms?

- **SVM**: Excellent for finding decision boundaries in high-dimensional data
- **Random Forest**: Provides feature importance and handles non-linear relationships well

Using both enables **triangulation** - validating findings across different methods.

## Understanding the Output

### SVM Output

SVM provides a **confusion matrix** showing:

- **True Positive (TP)**: Correctly predicted positive cases
- **False Positive (FP)**: Incorrectly predicted as positive
- **True Negative (TN)**: Correctly predicted negative cases
- **False Negative (FN)**: Incorrectly predicted as negative

Example output:
```

True Positive: 45
False Positive: 3
True Negative: 42
False Negative: 5

```

### Random Forest Output

Random Forest provides:
1. **Confusion matrix** (same format as SVM)
2. **Feature importance rankings** showing which variables are most predictive

Example output:
```

Feature Importance:
Feature: sleep_hours, Score: 0.34215
Feature: exercise_freq, Score: 0.28934
Feature: diet_quality, Score: 0.21456

````

Higher scores indicate more important predictors.

## Example Usage

### Basic Classification

```bash
crisp --inp ./data --outcome target_variable --cls
````

### With Feature Selection

```bash
crisp --inp ./data --outcome satisfaction --cls --include "age,income,education"
```

### With Top N Features

```bash
crisp --inp ./data --outcome churn --cls --rec 5 --outcome churn
```

Shows top 5 most important features from Random Forest.

### Save Results

```bash
crisp --inp ./data --outcome target --cls --out ./results/
```

Creates two files:

- `svm_results.json` - SVM confusion matrix
- `decision_tree_results.json` - Random Forest results (note: filename kept for backward compatibility)

## Best Practices

1. **Ensure outcome is categorical**: Classification requires a categorical target variable
2. **Use `--include` to select relevant features**: Reduces noise and improves model performance
3. **Compare results from both algorithms**: Look for agreement between SVM and Random Forest
4. **Use `--rec` to control feature importance display**: Default shows top 10 features
5. **Save results with `--out`**: Enables later comparison and reporting

## Interpreting Results

### When Algorithms Agree

If both SVM and Random Forest show similar accuracy and identify similar important features, your findings are more robust.

### When Algorithms Disagree

- **Different accuracy**: May indicate overfitting in one algorithm
- **Different feature importance**: Random Forest may capture non-linear relationships SVM misses
- **Investigate further**: Try different feature subsets or preprocessing

## Troubleshooting

### "Need at least 2 classes for classification"

**Problem**: Outcome variable has only one unique value

**Solution**: Check your data:

```bash
crisp --inp ./data --describe
```

Ensure outcome column has at least 2 distinct values.

### "Error in Random Forest: Not enough observations"

**Problem**: Too few data points for reliable classification

**Solution**:

- Increase dataset size
- Reduce number of features with `--include`
- Consider using only one algorithm if dataset is very small

### Low Accuracy from Both Algorithms

**Possible causes**:

- Insufficient features to predict outcome
- Noisy or irrelevant features
- Imbalanced classes
- Too few observations

**Try**:

- Add more predictive features
- Use `--include` to select only relevant variables
- Check class balance in outcome variable
- Consider data quality issues

## Technical Details

### Random Forest Configuration

- **Number of trees**: 100 estimators
- **Feature importance**: Computed using permutation importance
- **Test/train split**: 25% test, 75% training

### SVM Configuration

- **Kernel**: Linear
- **Feature scaling**: StandardScaler applied
- **Test/train split**: 25% test, 75% training

### Linkage Support

Both algorithms support text metadata linkage:

- `--linkage id`: Link by document ID
- `--linkage embedding`: Link by semantic similarity
- `--linkage temporal`: Link by temporal proximity
- `--linkage keyword`: Link by keyword matching

Use `--aggregation` to control how multiple linked documents are combined:

- `majority`: Most common value (default for classification)
- `first`: First encountered value
- `mode`: Most frequent value

## Comparison with Other ML Tools

### vs. `--regression`

- `--cls`: For categorical outcomes (e.g., yes/no, high/medium/low)
- `--regression`: For continuous outcomes (e.g., satisfaction score 1-10)

### vs. `--nnet`

- `--cls`: Two interpretable algorithms (SVM + Random Forest)
- `--nnet`: Neural network (black box, may perform better on complex patterns)

### vs. MCP Tools

MCP server provides separate tools:

- `svm_classification`: Run only SVM
- `decision_tree_classification`: Run only Random Forest

CLI `--cls` runs both for convenience and triangulation.

````

#### 2. **Update Existing Wiki Pages**

Check and update these pages if they exist:

- **CLI Reference** - Update `--cls` description to mention both algorithms
- **Getting Started** - If classification examples exist, clarify dual output
- **Machine Learning** - Add section explaining SVM + Random Forest approach

### Method 1: Using GitHub Web Interface

1. **Navigate to the Wiki**
   - Go to: https://github.com/dermatologist/crisp-t/wiki
   - Click **"Add a new page"**

2. **Create New Page**
   - Page name: `Classification`
   - Paste the content from above
   - Click **Save changes**

3. **Update Existing Pages**
   - Find pages mentioning `--cls` or classification
   - Update to reflect that both SVM and Random Forest run
   - Change "Decision Tree" to "Random Forest" where applicable

4. **Commit Changes**
   - Add commit message: "Document --cls dual-algorithm behavior (SVM + Random Forest)"
   - Click **Save changes**

### Method 2: Using Git (Advanced)

If you have write access to the wiki repository:

1. **Clone the Wiki Repository**

   ```bash
   git clone git@github.com:dermatologist/crisp-t.wiki.git
   cd crisp-t.wiki
````

2. **Create New Page**
   - Create file: `Classification.md`
   - Add content from above

3. **Update Existing Pages**
   - Search for "Decision Tree" and update to "Random Forest"
   - Search for `--cls` and clarify dual-algorithm behavior

4. **Push Changes**
   ```bash
   git add .
   git commit -m "Document --cls dual-algorithm behavior (SVM + Random Forest)"
   git push origin main
   ```

---

## Verification

After updating the wiki, verify:

1. ✅ New "Classification" page exists with comprehensive documentation
2. ✅ All references to "Decision Tree" updated to "Random Forest"
3. ✅ Examples show both SVM and Random Forest outputs
4. ✅ Troubleshooting section addresses common issues
5. ✅ Links to related pages work correctly

## Related Documentation

- `docs/cheatsheet.md` - Quick reference for CLI flags
- `docs/DEMO.md` - Usage examples
- `notes/INSTRUCTION.md` - Comprehensive user guide
- `docs/MCP_SERVER.md` - MCP tool documentation (separate tools for SVM and Random Forest)
