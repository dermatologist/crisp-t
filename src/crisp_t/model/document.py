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


from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Document model for storing text and metadata.
    """
    id: str = Field(..., description="Unique identifier for the document.")
    name: str | None = Field(None, description="Name of the corpus.")
    description: str | None = Field(None, description="Description of the corpus.")
    score: float = Field(0.0, description="Score associated with the document.")
    text: str = Field(..., description="The text content of the document.")
    timestamp: str | None = Field(None, description="Optional timestamp in ISO 8601 format (e.g., '2025-01-15T10:30:00Z').")
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the document."
    )

    def pretty_print(self):
        """
        Print the document information in a human-readable format.
        """
        print(f"Document ID: {self.id}")
        print(f"Name: {self.name}")
        print(f"Description: {self.description}")
        print(f"Score: {self.score}")
        print(f"Text: {self.text[:100]}...")  # Print first 100 characters of text
        print(f"Metadata: {self.metadata}")
        print(f"Length of Text: {len(self.text)} characters")
        print(f"Number of Metadata Entries: {len(self.metadata)}")
