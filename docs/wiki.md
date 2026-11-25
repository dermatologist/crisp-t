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
