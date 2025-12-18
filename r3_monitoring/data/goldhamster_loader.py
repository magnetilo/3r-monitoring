"""
Data loaders for various text classification datasets.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pandas as pd
# from abc import ABC, abstractmethod


# class BaseDataLoader(ABC):
#     """Abstract base class for dataset loaders."""
    
#     def __init__(self, labels: List[str]):
#         """
#         Initialize the base data loader.
        
#         Args:
#             labels: List of label names to use for classification
#         """
#         self.all_labels = labels
    
#     @abstractmethod
#     def read_text(self, doc_id: str) -> str:
#         """
#         Read text for a document.
        
#         Args:
#             doc_id: Document identifier
            
#         Returns:
#             Text content of the document
#         """
#         pass
    
#     @abstractmethod
#     def create_dataframe_from_split(self, split_file: str) -> Tuple[pd.DataFrame, List[int]]:
#         """
#         Create a dataframe from a split file.
        
#         Args:
#             split_file: Path to the split file
            
#         Returns:
#             Tuple of (dataframe with text and labels, list of skipped indices)
#         """
#         pass
    
#     @abstractmethod
#     def load_data(self, train_file: str, dev_file: str, test_file: Optional[str] = None) -> Dict[str, Tuple[pd.DataFrame, List[int]]]:
#         """
#         Load data for all splits.
        
#         Args:
#             train_file: Path to training data file
#             dev_file: Path to development data file
#             test_file: Path to test data file (optional)
            
#         Returns:
#             Dictionary with dataframes and skipped indices for each split
#         """
#         pass


class GoldhamsterDataLoader():
    """Data loader for Goldhamster dataset with papers and labels."""
    
    def __init__(
        self, 
        docs_dir: Path, 
        labels_dir: Path, 
        labels: Optional[List[str]] = None,
        use_titles: bool = False,
        use_abstracts: bool = True,
        use_mesh_terms: bool = False,
        text_column: str = "TEXT",
        id_column: str = "PMID"
    ):
        """
        Initialize the data loader with the directories containing papers and labels.
        
        Args:
            docs_dir: Directory containing paper JSON files
            labels_dir: Directory containing label files
            labels: Optional list of labels to use (defaults to all Goldhamster labels)
            use_titles: Whether to include paper titles in the text
            use_abstracts: Whether to include paper abstracts in the text
            use_mesh_terms: Whether to include MeSH terms in the text
            text_column: Name of the column containing text
            id_column: Name of the column containing document IDs
        """
        self.docs_dir = docs_dir
        self.labels_dir = labels_dir
        self.use_titles = use_titles
        self.use_abstracts = use_abstracts
        self.use_mesh_terms = use_mesh_terms
        self.text_column = text_column
        self.id_column = id_column
        
        # Use provided labels or default to Goldhamster labels
        default_labels = [
            'in_silico', 'organs', 'other', 'human', 'in_vivo',
            'invertebrate', 'primary_cells', 'immortal_cell_line'
        ]
        self.all_labels = labels if labels is not None else default_labels
    
    def read_text(self, pmid: str) -> str:
        """
        Read the text for a paper from its JSON file.
        
        Args:
            pmid: PubMed ID of the paper
            
        Returns:
            Text of the paper based on selected components (title, abstract, mesh terms)
        """
        file = self.docs_dir / f"{pmid}.json"
        paper_dict = json.loads(file.read_text())
        
        text_parts = []
        
        if self.use_titles and "title" in paper_dict:
            text_parts.append(paper_dict["title"])
            
        if self.use_abstracts and "abstract" in paper_dict:
            text_parts.append(paper_dict["abstract"])
            
        if self.use_mesh_terms and "mesh_terms" in paper_dict:
            mesh_terms = paper_dict["mesh_terms"]
            if isinstance(mesh_terms, list):
                mesh_text = " ".join(mesh_terms)
            else:
                mesh_text = str(mesh_terms)
            text_parts.append(mesh_text)
                
        return " ".join(text_parts).replace("\n", " ").replace("\t", " ")
    
    
    def create_dataframe_from_split(self, split_file: str) -> Tuple[pd.DataFrame, List[int]]:
        """
        Create a dataframe from a split file.
        
        Args:
            split_file: Name of the split file (train, dev, test)
            
        Returns:
            Tuple of (dataframe with text and labels, list of skipped indices)
        """
        rows = []
        skipped = []
        doc_index = 0
        
        split_path = self.labels_dir / split_file
        with open(split_path, "r") as reader:
            lines = reader.readlines()
            for line in lines:
                pmid, str_labels = line.strip().split("\t")
                labels = str_labels.split(",")
                text = self.read_text(pmid)
                
                # exclude documents w/o text
                if len(text) == 0:
                    skipped.append(doc_index)
                    doc_index += 1
                    continue
                
                # Create row with pmid, text and binary label indicators
                row = {self.id_column: pmid, self.text_column: text}
                for label in self.all_labels:
                    row[label] = 1 if label in labels else 0
                
                rows.append(row)
                doc_index += 1
        
        # Create dataframe
        df = pd.DataFrame(rows)
        
        # Create categorical label columns
        for label in self.all_labels:
            if label in df:
                df[label + '_label'] = pd.Categorical(df[label])
                df[label] = df[label + '_label'].cat.codes
        
        return df, skipped
    
    # def load_data(self, train_file: str, dev_file: Optional[str] = None, test_file: Optional[str] = None) -> Dict[str, Tuple[pd.DataFrame, List[int]]]:
    #     """
    #     Load all required data splits.
        
    #     Args:
    #         train_file: Name of the training split file
    #         dev_file: Name of the development split file (optional)
    #         test_file: Name of the test split file (optional)
            
    #     Returns:
    #         Dictionary with dataframes and skipped indices for each split
    #     """
    #     data = {}
        
    #     if train_file:
    #         data['train'] = self.create_dataframe_from_split(train_file)
        
    #     if dev_file:
    #         data['dev'] = self.create_dataframe_from_split(dev_file)
        
    #     if test_file:
    #         data['test'] = self.create_dataframe_from_split(test_file)
        
    #     return data


# class GenericTextDataLoader(BaseDataLoader):
#     """
#     Generic data loader for text classification datasets.
    
#     This class can be used as a template for implementing other data loaders.
#     """
    
#     def __init__(
#         self, 
#         data_dir: Path, 
#         labels: List[str],
#         text_column: str = "text",
#         id_column: str = "id"
#     ):
#         """
#         Initialize the generic data loader.
        
#         Args:
#             data_dir: Directory containing data files
#             labels: List of label names to use for classification
#             text_column: Name of the column containing text
#             id_column: Name of the column containing document IDs
#         """
#         self.data_dir = data_dir
#         self.text_column = text_column
#         self.id_column = id_column
#         super().__init__(labels)
    
#     def read_text(self, doc_id: str) -> str:
#         """
#         Read text for a document. Override this method for specific implementations.
        
#         Args:
#             doc_id: Document identifier
            
#         Returns:
#             Text content of the document
#         """
#         # This is a placeholder that should be overridden in subclasses
#         return ""
    
#     def create_dataframe_from_split(self, split_file: str) -> Tuple[pd.DataFrame, List[int]]:
#         """
#         Create a dataframe from a split file. This example assumes CSV format.
        
#         Args:
#             split_file: Path to the split file
            
#         Returns:
#             Tuple of (dataframe with text and labels, list of skipped indices)
#         """
#         # Implement specific logic for reading dataset splits
#         # This is a placeholder that should be overridden in subclasses
#         df = pd.read_csv(self.data_dir / split_file)
#         skipped = []
        
#         # Process labels
#         for label in self.all_labels:
#             if label in df:
#                 df[label + '_label'] = pd.Categorical(df[label])
#                 df[label] = df[label + '_label'].cat.codes
        
#         # Standardize column names
#         if self.text_column != "TEXT":
#             df["TEXT"] = df[self.text_column]
        
#         if self.id_column != "PMID":
#             df["PMID"] = df[self.id_column]
        
#         return df, skipped
    
#     def load_data(self, train_file: str, dev_file: Optional[str] = None, test_file: Optional[str] = None) -> Dict[str, Tuple[pd.DataFrame, List[int]]]:
#         """
#         Load all required data splits.
        
#         Args:
#             train_file: Path to training data file
#             dev_file: Path to development data file (optional)
#             test_file: Path to test data file (optional)
            
#         Returns:
#             Dictionary with dataframes and skipped indices for each split
#         """
#         data = {}
        
#         if train_file:
#             data['train'] = self.create_dataframe_from_split(train_file)
        
#         if dev_file:
#             data['dev'] = self.create_dataframe_from_split(dev_file)
        
#         if test_file:
#             data['test'] = self.create_dataframe_from_split(test_file)
        
#         return data