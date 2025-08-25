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
