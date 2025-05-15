from pydantic import BaseModel, Field
from typing import Optional


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
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the corpus."
    )
    id: str = Field(..., description="Unique identifier for the corpus.")
    score: Optional[float] = Field(None, description="Score associated with the corpus.")
    name: Optional[str] = Field(None, description="Name of the corpus.")
    description: Optional[str] = Field(None, description="Description of the corpus.")
