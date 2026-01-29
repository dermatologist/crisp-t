#!/usr/bin/env python
"""
Example script demonstrating partial import functionality.

This script shows how to use the --num and --rec flags to limit
the amount of data imported when using --source.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def create_sample_dataset():
    """Create a sample dataset with multiple files."""
    tmpdir = tempfile.mkdtemp(prefix="crisp_example_")
    tmpdir_path = Path(tmpdir)
    
    print(f"Creating sample dataset in: {tmpdir}")
    
    # Create 10 text files
    for i in range(10):
        text_file = tmpdir_path / f"interview_{i+1}.txt"
        text_file.write_text(
            f"Interview {i+1}\n\n"
            f"This is a sample interview transcript number {i+1}. "
            f"The participant discussed various topics including "
            f"their experiences, perspectives, and insights. "
            f"This text would typically be much longer in a real study."
        )
    
    # Create a CSV file with 50 rows
    csv_file = tmpdir_path / "survey_data.csv"
    csv_lines = ["id,age,satisfaction,category\n"]
    for i in range(1, 51):
        csv_lines.append(f"{i},{20+i%50},{1+i%5},Category_{i%3}\n")
    csv_file.write_text("".join(csv_lines))
    
    print(f"✓ Created 10 text files and 1 CSV with 50 rows")
    return tmpdir


def run_crisp_command(args):
    """Run a crisp command and return the output."""
    cmd = ["crisp"] + args
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
    else:
        print(f"✓ Success!")
        # Print relevant output lines
        for line in result.stdout.split('\n'):
            if 'Successfully loaded' in line or 'document(s)' in line or 'shape:' in line:
                print(f"  {line.strip()}")
    return result


def main():
    """Demonstrate partial import feature."""
    print("=" * 60)
    print("CRISP-T Partial Import Feature Demo")
    print("=" * 60)
    
    # Create sample dataset
    source_dir = create_sample_dataset()
    output_dir = tempfile.mkdtemp(prefix="crisp_output_")
    
    try:
        # Example 1: Import all files (no limits)
        print("\n" + "=" * 60)
        print("Example 1: Import ALL files (baseline)")
        print("=" * 60)
        run_crisp_command([
            "--source", source_dir,
            "--out", os.path.join(output_dir, "full_corpus")
        ])
        
        # Example 2: Import only 3 text files
        print("\n" + "=" * 60)
        print("Example 2: Import only 3 text files (--num 3)")
        print("=" * 60)
        run_crisp_command([
            "--source", source_dir,
            "--num", "3",
            "--out", os.path.join(output_dir, "limited_texts")
        ])
        
        # Example 3: Import only 10 CSV rows
        print("\n" + "=" * 60)
        print("Example 3: Import only 10 CSV rows (--rec 10)")
        print("=" * 60)
        run_crisp_command([
            "--source", source_dir,
            "--rec", "10",
            "--out", os.path.join(output_dir, "limited_csv")
        ])
        
        # Example 4: Import 5 text files and 20 CSV rows
        print("\n" + "=" * 60)
        print("Example 4: Import 5 texts + 20 CSV rows (--num 5 --rec 20)")
        print("=" * 60)
        run_crisp_command([
            "--source", source_dir,
            "--num", "5",
            "--rec", "20",
            "--out", os.path.join(output_dir, "limited_both")
        ])
        
        print("\n" + "=" * 60)
        print("✓ Demo completed successfully!")
        print("=" * 60)
        print(f"\nSample data was created in: {source_dir}")
        print(f"Output corpora were saved in: {output_dir}")
        print("\nYou can now explore these corpora using:")
        print(f"  crisp --inp {output_dir}/full_corpus --print corpus")
        print(f"  crisp --inp {output_dir}/limited_texts --print corpus")
        print(f"  crisp --inp {output_dir}/limited_csv --print corpus")
        print(f"  crisp --inp {output_dir}/limited_both --print corpus")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    
    print(f"\n💡 Tip: Clean up temporary files when done:")
    print(f"  rm -rf {source_dir}")
    print(f"  rm -rf {output_dir}")


if __name__ == "__main__":
    main()
