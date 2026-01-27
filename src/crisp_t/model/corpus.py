"""
Copyright (C) 2025 Bell Eapen

This file is part of crisp-t.

crisp-t is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

crisp-t is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with crisp-t.  If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .document import Document


class Corpus(BaseModel):
    """
    Corpus model for storing a collection of documents.
    """

    id: str = Field(..., description="Unique identifier for the corpus.")
    name: str | None = Field(None, description="Name of the corpus.")
    description: str | None = Field(None, description="Description of the corpus.")
    score: float | None = Field(
        None, description="Score associated with the corpus."
    )
    documents: list[Document] = Field(
        default_factory=list, description="List of documents in the corpus."
    )
    df: pd.DataFrame | None = Field(
        None, description="Numeric data associated with the corpus."
    )
    visualization: dict[str, Any] = Field(
        default_factory=dict, description="Visualization data associated with the corpus."
    )
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # required for pandas DataFrame
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the corpus."
    )

    def pretty_print(self, show="all"):
        """
        Print the corpus information in a human-readable format.

        Args:
            show: Display option. Can be:
                - "all": Show all corpus information
                - "documents": Show first 5 documents
                - "documents N": Show first N documents (e.g., "documents 10")
                - "documents metadata": Show document-specific metadata
                - "dataframe": Show DataFrame head
                - "dataframe metadata": Show DataFrame metadata columns (metadata_*)
                - "dataframe stats": Show DataFrame statistics
                - "metadata": Show all corpus metadata
                - "metadata KEY": Show specific metadata key (e.g., "metadata pca")
                - "stats": Show DataFrame statistics (deprecated, use "dataframe stats")
        """
        # Color codes for terminal output
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        MAGENTA = "\033[95m"
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        # Parse the show parameter to support subcommands
        parts = show.split(maxsplit=1)
        main_command = parts[0]
        sub_command = parts[1] if len(parts) > 1 else None

        # Print basic corpus info for most commands
        if main_command in ["all", "documents", "dataframe", "metadata"]:
            print(f"{BOLD}{BLUE}Corpus ID:{RESET} {self.id}")
            print(f"{BOLD}{BLUE}Name:{RESET} {self.name}")
            print(f"{BOLD}{BLUE}Description:{RESET} {self.description}")

        # Handle documents command
        if main_command in ["all", "documents"]:
            if sub_command == "metadata":
                # Show document-specific metadata
                print(f"\n{BOLD}{GREEN}=== Document Metadata ==={RESET}")
                if not self.documents:
                    print("No documents in corpus")
                else:
                    for i, doc in enumerate(self.documents, 1):
                        print(f"\n{CYAN}Document {i}:{RESET}")
                        print(f"  {BOLD}ID:{RESET} {doc.id}")
                        print(f"  {BOLD}Name:{RESET} {doc.name}")
                        if doc.metadata:
                            print(f"  {BOLD}Metadata:{RESET}")
                            for key, value in doc.metadata.items():
                                # Truncate long values
                                val_str = str(value)
                                if len(val_str) > 100:
                                    val_str = val_str[:97] + "..."
                                print(f"    {YELLOW}{key}:{RESET} {val_str}")
                        else:
                            print(f"  {BOLD}Metadata:{RESET} (none)")
            else:
                # Determine how many documents to show
                num_docs = 5  # default
                if sub_command:
                    try:
                        num_docs = int(sub_command)
                    except ValueError:
                        print(f"{RED}Invalid number for documents: {sub_command}. Using default (5).{RESET}")

                print(f"\n{BOLD}{GREEN}=== Documents ==={RESET}")
                print(f"Total documents: {len(self.documents)}")
                print(f"Showing first {min(num_docs, len(self.documents))} document(s):\n")

                for i, doc in enumerate(self.documents[:num_docs], 1):
                    print(f"{CYAN}Document {i}:{RESET}")
                    print(f"  {BOLD}Name:{RESET} {doc.name}")
                    print(f"  {BOLD}ID:{RESET} {doc.id}")
                    # Show a snippet of text if available
                    if hasattr(doc, 'text') and doc.text:
                        text_snippet = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                        print(f"  {BOLD}Text:{RESET} {text_snippet}")
                    print()

        # Handle dataframe command
        if main_command in ["all", "dataframe"]:
            if self.df is not None:
                if sub_command == "metadata":
                    # Show DataFrame metadata columns (columns starting with metadata_)
                    print(f"\n{BOLD}{GREEN}=== DataFrame Metadata Columns ==={RESET}")
                    metadata_cols = [col for col in self.df.columns if col.startswith("metadata_")]
                    if metadata_cols:
                        print(f"Found {len(metadata_cols)} metadata column(s):")
                        for col in metadata_cols:
                            print(f"  {YELLOW}{col}{RESET}")
                            # Show some statistics for the metadata column
                            print(f"    Non-null values: {self.df[col].notna().sum()}")
                            print(f"    Null values: {self.df[col].isna().sum()}")
                            # Show unique values if not too many
                            unique_count = self.df[col].nunique()
                            if unique_count <= 10:
                                print(f"    Unique values ({unique_count}): {list(self.df[col].unique())}")
                            else:
                                print(f"    Unique values: {unique_count}")
                    else:
                        print("No metadata columns found (columns starting with 'metadata_')")
                elif sub_command == "stats":
                    # Show DataFrame statistics
                    self._print_dataframe_stats()
                else:
                    # Show DataFrame head
                    print(f"\n{BOLD}{GREEN}=== DataFrame ==={RESET}")
                    print(f"Shape: {self.df.shape}")
                    print(f"Columns: {list(self.df.columns)}")
                    print("\nFirst few rows:")
                    print(self.df.head())
            else:
                if main_command == "dataframe":
                    print(f"\n{BOLD}{RED}No DataFrame available{RESET}")

        # Handle metadata command
        if main_command in ["all", "metadata"]:
            if sub_command:
                # Show specific metadata key
                print(f"\n{BOLD}{GREEN}=== Metadata: {sub_command} ==={RESET}")
                if sub_command in self.metadata:
                    value = self.metadata[sub_command]
                    # Format the output based on the type of value
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"{YELLOW}{k}:{RESET} {v}")
                    elif isinstance(value, list):
                        for i, item in enumerate(value, 1):
                            print(f"{i}. {item}")
                    else:
                        print(value)
                else:
                    print(f"{RED}Metadata key '{sub_command}' not found{RESET}")
                    available_keys = list(self.metadata.keys())
                    if available_keys:
                        print(f"Available keys: {', '.join(available_keys)}")
            else:
                # Show all metadata
                print(f"\n{BOLD}{GREEN}=== Corpus Metadata ==={RESET}")
                if not self.metadata:
                    print("No metadata available")
                else:
                    for key, value in self.metadata.items():
                        print(f"\n{MAGENTA}{key}:{RESET}")
                        # Truncate long values
                        val_str = str(value)
                        if len(val_str) > 500:
                            val_str = val_str[:497] + "..."
                        print(f"  {val_str}")

        # Handle stats command (deprecated, redirect to dataframe stats)
        if main_command == "stats":
            print(f"{YELLOW}Note: 'stats' is deprecated. Use 'dataframe stats' instead.{RESET}")
            if self.df is not None:
                self._print_dataframe_stats()
            else:
                print(f"{RED}No DataFrame available{RESET}")

        print(f"\n{BOLD}Display completed for '{show}'{RESET}")

    def _print_dataframe_stats(self):
        """Helper method to print DataFrame statistics."""
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        GREEN = "\033[92m"

        print(f"\n{BOLD}{GREEN}=== DataFrame Statistics ==={RESET}")
        print(self.df.describe())
        print(f"\n{BOLD}Distinct values per column:{RESET}")
        for col in self.df.columns:
            nunique = self.df[col].nunique()
            print(f"  {YELLOW}{col}:{RESET} {nunique} distinct value(s)")
            # If distinct values < 10, show value counts
            if nunique <= 10:
                print("    Value counts:")
                for val, count in self.df[col].value_counts().items():
                    print(f"      {val}: {count}")
                print()
    def get_all_df_column_names(self):
        """
        Get a list of all column names in the DataFrame.

        Returns:
            List of column names.
        """
        if self.df is not None:
            return self.df.columns.tolist()
        return []

    def get_descriptive_statistics(self):
        """
        Get descriptive statistics of the DataFrame.

        Returns:
            DataFrame containing descriptive statistics, or None if DataFrame is None.
        """
        if self.df is not None:
            return self.df.describe()
        return None

    def get_row_count(self):
        """
        Get the number of rows in the DataFrame.

        Returns:
            Number of rows in the DataFrame, or 0 if DataFrame is None.
        """
        if self.df is not None:
            return len(self.df)
        return 0

    def get_row_by_index(self, index: int) -> pd.Series | None:
        """
        Get a row from the DataFrame by its index.

        Args:
            index: Index of the row to retrieve.
        Returns:
            Row as a pandas Series if index is valid, else None.
        """
        if self.df is not None and 0 <= index < len(self.df):
            return self.df.iloc[index]
        return None

    def get_all_document_ids(self):
        """
        Get a list of all document IDs in the corpus.

        Returns:
            List of document IDs.
        """
        return [doc.id for doc in self.documents]

    def get_document_by_id(self, document_id: str) -> Document | None:
        """
        Get a document by its ID.

        Args:
            document_id: ID of the document to retrieve.

        Returns:
            Document object if found, else None.
        """
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        return None

    def add_document(self, document: Document):
        """
        Add a document to the corpus.

        Args:
            document: Document object to add.
        """
        self.documents.append(document)

    def remove_document_by_id(self, document_id: str):
        """
        Remove a document from the corpus by its ID.

        Args:
            document_id: ID of the document to remove.
        """
        self.documents = [
            doc for doc in self.documents if doc.id != document_id
        ]

    def update_metadata(self, key: str, value: Any):
        """
        Update the metadata of the corpus.

        Args:
            key: Metadata key to update.
            value: New value for the metadata key.
        """
        self.metadata[key] = value

    def add_relationship(self, first: str, second: str, relation: str):
        """
        Add a relationship between two documents in the corpus.

        Args:
            first: keywords from text documents in the format text:keyword or columns from dataframe in the format numb:column
            second: keywords from text documents in the format text:keyword or columns from dataframe in the format numb:column
            relation: Description of the relationship. (One of "correlates", "similar to", "cites", "references", "contradicts", etc.)
        """
        if "relationships" not in self.metadata:
            self.metadata["relationships"] = []
        self.metadata["relationships"].append(
            {"first": first, "second": second, "relation": relation}
        )

    def clear_relationships(self):
        """
        Clear all relationships in the corpus metadata.
        """
        if "relationships" in self.metadata:
            self.metadata["relationships"] = []

    def get_relationships(self):
        """
        Get all relationships in the corpus metadata.

        Returns:
            List of relationships, or empty list if none exist.
        """
        return self.metadata.get("relationships", [])

    def get_all_relationships_for_keyword(self, keyword: str):
        """
        Get all relationships involving a specific keyword.

        Args:
            keyword: Keyword to search for in relationships.

        Returns:
            List of relationships involving the keyword.
        """
        rels = self.get_relationships()
        return [
            rel
            for rel in rels
            if keyword in rel["first"] or keyword in rel["second"]
        ]
