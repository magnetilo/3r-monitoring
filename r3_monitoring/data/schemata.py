"""Data schema definitions for 3R Monitoring."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Schema for a document in the dataset."""
    pmid: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    mesh_terms: Optional[List[str]] = None


@dataclass
class LabeledDocument:
    """Schema for a labeled document."""
    document: Document
    labels: List[str]


@dataclass
class DatasetSplit:
    """Schema for a dataset split."""
    name: str
    documents: List[LabeledDocument]


@dataclass
class EvaluationResult:
    """Schema for evaluation results."""
    precision: float
    recall: float
    f1_score: float
    accuracy: Optional[float] = None
    support: Optional[int] = None