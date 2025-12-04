from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from transformers import BertModel, BertTokenizerFast


class BioBERTClassifier:
    """Generic BioBERT model for multi-label text classification."""
    
    def __init__(
        self, 
        labels: List[str],
        model_path: Path,
        model_pretrainded: str = "dmis-lab/biobert-v1.1",
        max_length: int = 256,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 10,
        load_model: Optional[bool] = False
    ):
        """
        Initialize the BioBERT model.
        
        Args:
            model_dir: Directory to save models
            labels: List of labels to predict
            model_pretrainded: Pre-trained Hugging Face model name to use instead of BioBERT
            max_length: Maximum sequence length for BERT
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs for training
            load_model: Whether to load a pretrained model from model_path
        """
        self.model_path = model_path
        self.model_pretrainded = model_pretrainded
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.all_labels = labels
        self.load_model = load_model

        self.model = None
        self.tokenizer = None
        
        # Initialize BioBERT components
        self._setup_biobert(model_pretrainded)

        # Load the model if specified
        if load_model:
            self._load_model(model_path)
    
    def _setup_biobert(self, model_pretrainded) -> None:
        """Set up the BioBERT tokenizer and model."""
        # Load BioBERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_pretrainded)
        
        # Load the Transformers BERT model
        self.transformer_model = TFBertModel.from_pretrained(
            model_pretrainded, 
            from_pt=True
        )
        self.config = self.transformer_model.config
    
    # def set_custom_bert_model(self, model_name: str) -> None:
    #     """
    #     Use a custom BERT model instead of BioBERT.
        
    #     Args:
    #         model_name: Name of the BERT model to use (from Hugging Face)
    #     """
    #     self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
    #     self.transformer_model = TFBertModel.from_pretrained(model_name, from_pt=True)
    #     self.config = self.transformer_model.config
    
    def _prepare_data(self, data: pd.DataFrame, text_column: str = "TEXT") -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Prepare input and output data for the model.
        
        Args:
            data: DataFrame with text and labels
            text_column: Name of the column containing text
            
        Returns:
            Tuple of (input dict, output dict)
        """
        # Prepare labels
        y_dict = {}
        for label in self.all_labels:
            y_dict[label] = to_categorical(data[label])
        
        # Prepare input text
        x_encoded = self.tokenizer(
            text=data[text_column].fillna('').astype(str).to_list(),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True, 
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=False,
            verbose=True
        )
        
        x_dict = {'input_ids': x_encoded['input_ids']}
        
        return x_dict, y_dict
    
    def _build_model(self, data: pd.DataFrame) -> None:
        """
        Build the BioBERT model architecture.
        
        Args:
            data: DataFrame used to determine output dimensions
        """
        # Build model input
        input_ids = Input(shape=(self.max_length,), name='input_ids', dtype='int32')
        inputs = {'input_ids': input_ids}
        
        # Add the BERT layer
        bert = self.transformer_model.layers[0]
        bert_output = bert(inputs)[1]
        
        # Add dropout
        dropout = Dropout(self.config.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(bert_output, training=False)
        
        # Add output layers (one per label)
        outputs = {}
        for label in self.all_labels:
            label_output = Dense(
                units=len(data[label + '_label'].value_counts()),
                kernel_initializer=TruncatedNormal(stddev=self.config.initializer_range),
                name=label
            )(pooled_output)
            outputs[label] = label_output
        
        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
        
        # Compile the model
        loss = {label: CategoricalCrossentropy(from_logits=True) for label in self.all_labels}
        metrics = {label: CategoricalAccuracy('accuracy') for label in self.all_labels}
        
        optimizer = Adam(
            self.learning_rate,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss, 
            metrics=metrics
        )
        
        # Print model summary
        self.model.summary()
    
    def train(self, train_data: pd.DataFrame, dev_data: pd.DataFrame, text_column: str = "TEXT") -> tf.keras.callbacks.History:
        """
        Train the BioBERT model.
        
        Args:
            train_data: Training data
            dev_data: Validation data
            text_column: Name of the column containing text
            
        Returns:
            Training history
        """
        # Check if model is built
        if self.model is None:
            self._build_model(train_data)
        
        # Prepare training and validation data
        x_train, y_train = self._prepare_data(train_data, text_column)
        x_val, y_val = self._prepare_data(dev_data, text_column)
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        
        # Save the model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return history
    
    def _load_model(self, model_path: Path) -> None:
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the model file
        """
        self.model = load_model(model_path)
        self.model_path = model_path
        print(f"Model loaded from {model_path}")
        self.model.summary()
    
    def predict(self, test_data: pd.DataFrame, text_column: str = "TEXT") -> Dict[str, Any]:
        """
        Make predictions with the model.
        
        Args:
            test_data: Test data
            text_column: Name of the column containing text
            
        Returns:
            Dictionary with predictions for each label
        """
        # Prepare test data
        x_test, _ = self._prepare_data(test_data, text_column)
        
        # Run predictions
        predictions = self.model.predict(x_test)
        
        return predictions
    
    def format_predictions(
        self, 
        predictions: Dict[str, Any], 
        test_data: pd.DataFrame, 
        skipped_indices: List[int],
        id_column: str = "PMID"
    ) -> pd.DataFrame:
        """
        Format predictions into a dataframe.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            skipped_indices: List of indices that were skipped
            id_column: Name of the column containing document IDs
            
        Returns:
            DataFrame with document IDs and predicted labels
        """
        results = []
        doc_ids = test_data[id_column].tolist()
        
        for i, doc_id in enumerate(doc_ids):
            predicted_labels = []
            
            for label in self.all_labels:
                pred_array = predictions[label][i]
                if pred_array[1] > pred_array[0]:
                    predicted_labels.append(label)
            
            results.append({
                id_column: doc_id,
                'predicted_labels': ','.join(predicted_labels)
            })
        
        # Add skipped documents with empty predictions
        # (this requires access to the original test file which we don't have here)
        
        return pd.DataFrame(results)
    
    def save_predictions(
        self, 
        predictions: Dict[str, Any], 
        test_data: pd.DataFrame, 
        skipped_indices: List[int],
        output_file: Path,
        id_column: str = "PMID"
    ) -> None:
        """
        Save predictions to a file.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            skipped_indices: List of indices that were skipped
            output_file: Path to save predictions
            id_column: Name of the column containing document IDs
        """
        results_df = self.format_predictions(predictions, test_data, skipped_indices, id_column)
        
        with open(output_file, 'w') as f:
            for _, row in results_df.iterrows():
                f.write(f"{row[id_column]}\t{row['predicted_labels']}\n")
        
        print(f"Predictions saved to {output_file}")