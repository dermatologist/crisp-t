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

from typing import Dict, Optional, Any
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from .document import Document


class Corpus(BaseModel):
    """
    Corpus model for storing a collection of documents.
    """

    id: str = Field(..., description="Unique identifier for the corpus.")
    name: Optional[str] = Field(None, description="Name of the corpus.")
    description: Optional[str] = Field(None, description="Description of the corpus.")
    score: Optional[float] = Field(
        None, description="Score associated with the corpus."
    )
    documents: list[Document] = Field(
        default_factory=list, description="List of documents in the corpus."
    )
    df: Optional[pd.DataFrame] = Field(
        None, description="Numeric data associated with the corpus."
    )
    visualization: Dict[str, Any] = Field(
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
        """
        if show not in ["all", "documents", "dataframe", "metadata", "stats"]:
            return
        print(f"Corpus ID: {self.id}")
        print(f"Name: {self.name}")
        print(f"Description: {self.description}")
        # print(f"Score: {self.score}")
        if show in ["all", "documents"]:
            print("Documents:")
            for doc in self.documents[:5]:  # Print only first 5 documents for brevity
                print(f"Printing first 5 documents out of {len(self.documents)}")
                print(f"   Name: {doc.name}")
                print(f"   ID: {doc.id}")
                # Print metadata as key-value pairs
                for key, value in doc.metadata.items():
                    print(f"      - {key}: {value}")
                print()
        if show in ["all", "dataframe"]:
            if self.df is not None:
                print("DataFrame:")
                print(self.df.head())
        if show in ["all", "visualization"]:
            if self.visualization is not None:
                print("Visualization:")
            print(self.visualization.keys())
        if show in ["all", "metadata"]:
            print("Metadata:")
            for key, value in self.metadata.items():
                print(f" - {key}\n: {value}")
        if show in ["all", "stats"]:
            if self.df is not None:
                print("DataFrame Statistics:")
                print(self.df.describe())
                ## Print number of distinct values for each column
                print("Distinct values per column:")
                for col in self.df.columns:
                    print(f" - {col}: {self.df[col].nunique()} distinct values")
                    # if distinct values < 10 print the values with counts
                    if self.df[col].nunique() <= 10:
                        print(self.df[col].value_counts())
        print(f"Showing completed for '{show}'")
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

    def get_row_by_index(self, index: int) -> Optional[pd.Series]:
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

    def get_document_by_id(self, document_id: str) -> Optional[Document]:
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
