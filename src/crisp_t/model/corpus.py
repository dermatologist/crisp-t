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
    visualization: Optional[Dict[str, Any]] = Field(
        None, description="Visualization data associated with the corpus."
    )
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # required for pandas DataFrame
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the corpus."
    )

    def pretty_print(self):
        """
        Print the corpus information in a human-readable format.
        """
        print(f"Corpus ID: {self.id}")
        print(f"Name: {self.name}")
        print(f"Description: {self.description}")
        print(f"Score: {self.score}")
        print("Documents:")
        for doc in self.documents:
            print(f" - {doc.name}")
        if self.df is not None:
            print("DataFrame:")
            print(self.df.head())
        if self.visualization is not None:
            print("Visualization:")
            print(self.visualization)
        print("Metadata:")
        for key, value in self.metadata.items():
            print(f" - {key}: {value}")