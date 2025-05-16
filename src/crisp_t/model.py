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

from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class Document(BaseModel):
    """
    Document model for storing text and metadata.
    """

    text: str = Field(..., description="The text content of the document.")
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the document."
    )
    id: str = Field(..., description="Unique identifier for the document.")
    score: float = Field(0.0, description="Score associated with the document.")
    name: Optional[str] = Field(None, description="Name of the corpus.")
    description: Optional[str] = Field(None, description="Description of the corpus.")


class Corpus(BaseModel):
    """
    Corpus model for storing a collection of documents.
    """

    documents: list[Document] = Field(
        default_factory=list, description="List of documents in the corpus."
    )
    df: Optional[pd.DataFrame] = Field(
        None, description="Numeric data associated with the corpus."
    )
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # required for pandas DataFrame
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the corpus."
    )
    id: str = Field(..., description="Unique identifier for the corpus.")
    score: Optional[float] = Field(
        None, description="Score associated with the corpus."
    )
    name: Optional[str] = Field(None, description="Name of the corpus.")
    description: Optional[str] = Field(None, description="Description of the corpus.")
