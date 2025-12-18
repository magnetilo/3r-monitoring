"""
PyTorch BioBERT model for multi-label text classification.
"""


from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import BertModel, BertTokenizerFast


class TextDataset(Dataset):
    """Custom dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: BertTokenizerFast, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }


class BioBERTClassifier(nn.Module):
    """PyTorch BioBERT model for multi-label text classification."""
    
    def __init__(
        self, 
        labels: List[str],
        model_path: Path,
        model_pretrainded: str = "dmis-lab/biobert-v1.1",  # Keep typo for backward compatibility
        max_length: int = 256,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 10,
        dropout_rate: float = 0.1,
        load_model: Optional[bool] = False,
        device: Optional[str] = None
    ):
        """
        Initialize the BioBERT model.
        
        Args:
            labels: List of labels to predict
            model_path: Path to save/load model
            model_pretrainded: Pre-trained Hugging Face model name (typo kept for compatibility)
            max_length: Maximum sequence length for BERT
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs for training
            dropout_rate: Dropout rate
            load_model: Whether to load existing model
            device: Device to use ('cuda', 'mps', 'cpu' or None for auto)
        """
        super(BioBERTClassifier, self).__init__()
        
        # Fix the typo in parameter name
        model_pretrained = model_pretrainded
        
        self.model_path = model_path
        self.model_pretrained = model_pretrained
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.all_labels = labels
        self.num_labels = len(labels)
        self.dropout_rate = dropout_rate
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer and BERT model
        self.tokenizer = BertTokenizerFast.from_pretrained(model_pretrained)
        self.bert = BertModel.from_pretrained(model_pretrained)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifiers = nn.ModuleDict({
            label: nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification per label
            for label in labels
        })
        
        # Label binarizers for each label
        self.label_binarizers = {label: LabelBinarizer() for label in labels}
        
        # Move model to device
        self.to(self.device)
        
        # Load model if specified
        if load_model and model_path.exists():
            self.load_model(model_path)
        elif load_model:
            print(f"Model file not found at {model_path}, model initialized but not trained.")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each label
        logits = {}
        for label in self.all_labels:
            logits[label] = self.classifiers[label](pooled_output)
        
        return logits
    
    def _prepare_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare multi-label binary matrix."""
        # Initialize label matrix
        label_matrix = np.zeros((len(data), len(self.all_labels)))
        
        for i, label in enumerate(self.all_labels):
            # Fit label binarizer and transform labels
            label_col = data[label].values
            self.label_binarizers[label].fit(label_col)
            # Convert to binary (assuming labels are 0/1 or boolean)
            label_matrix[:, i] = label_col.astype(int)
        
        return label_matrix
    
    def _create_data_loader(self, data: pd.DataFrame, text_column: str = "TEXT", is_training: bool = True) -> DataLoader:
        """Create DataLoader for the dataset."""
        texts = data[text_column].fillna('').astype(str).tolist()
        
        if is_training:
            labels = self._prepare_labels(data)
        else:
            # For prediction, create dummy labels
            labels = np.zeros((len(texts), len(self.all_labels)))
        
        dataset = TextDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_training)
    
    def train_model(self, train_data: pd.DataFrame, dev_data: pd.DataFrame, text_column: str = "TEXT", 
                    patience: int = 3) -> Dict[str, List[float]]:
        """
        Train the BioBERT model.
        
        Args:
            train_data: Training data
            dev_data: Validation data
            text_column: Name of the column containing text
            
        Returns:
            Training history
        """
        # Create data loaders
        train_loader = self._create_data_loader(train_data, text_column, is_training=True)
        val_loader = self._create_data_loader(dev_data, text_column, is_training=True)
        
        # Set up optimizer and loss function
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Best model tracking and early stopping
        best_val_accuracy = 0.0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(self.epochs):
            # Training phase
            super().train()  # Set to training mode
            total_train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(input_ids, attention_mask)
                
                # Calculate loss for each label
                total_loss = 0
                for i, label in enumerate(self.all_labels):
                    label_logits = outputs[label]
                    label_targets = labels[:, i].long()
                    loss = criterion(label_logits, label_targets)
                    total_loss += loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                total_train_loss += total_loss.item()
            
            # Validation phase
            self.eval()
            total_val_loss = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self(input_ids, attention_mask)
                    
                    # Calculate validation loss
                    total_loss = 0
                    batch_predictions = []
                    
                    for i, label in enumerate(self.all_labels):
                        label_logits = outputs[label]
                        label_targets = labels[:, i].long()
                        loss = criterion(label_logits, label_targets)
                        total_loss += loss
                        
                        # Get predictions
                        pred = torch.argmax(label_logits, dim=1)
                        batch_predictions.append(pred.cpu().numpy())
                    
                    total_val_loss += total_loss.item()
                    all_predictions.extend(np.column_stack(batch_predictions))
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
            
            # Update history
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(accuracy)
            
            # Check if this is the best model so far
            if accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_model_state = self.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0  # Reset patience counter
                print(f"Epoch {epoch + 1}/{self.epochs} - NEW BEST MODEL!")
            else:
                patience_counter += 1
                print(f"Epoch {epoch + 1}/{self.epochs}")
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            if best_epoch > 0:
                print(f"Best Val Accuracy: {best_val_accuracy:.4f} (Epoch {best_epoch})")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Stopping at epoch {epoch + 1}, best model was at epoch {best_epoch}")
                break
        
        # Restore the best model before saving
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"\nRestored best model from epoch {best_epoch} (Val Accuracy: {best_val_accuracy:.4f})")
            # Update history to reflect the best model's performance
            history['best_epoch'] = best_epoch
            history['best_val_accuracy'] = best_val_accuracy
        
        # Save the best model
        self.save_model()
        
        return history
    
    def _get_predictions(self, test_data: pd.DataFrame, text_column: str = "TEXT", return_proba: bool = False) -> Dict[str, np.ndarray]:
        """
        Internal method to get predictions or probabilities.
        
        Args:
            test_data: Test data
            text_column: Name of the column containing text
            return_proba: If True, return probabilities; if False, return binary predictions
            
        Returns:
            Dictionary with predictions/probabilities for each label
        """
        self.eval()
        test_loader = self._create_data_loader(test_data, text_column, is_training=False)
        
        predictions = {label: [] for label in self.all_labels}
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self(input_ids, attention_mask)
                
                for label in self.all_labels:
                    logits = outputs[label]  # Shape: [batch_size, 2]
                    
                    if return_proba:
                        # Apply softmax and get probability of positive class (class 1)
                        probs = torch.softmax(logits, dim=1)[:, 1]  # Take probability of class 1
                        predictions[label].extend(probs.cpu().numpy())
                    else:
                        # Return binary predictions using argmax
                        preds = torch.argmax(logits, dim=1)  # 0 or 1
                        predictions[label].extend(preds.cpu().numpy())
        
        # Convert to numpy arrays
        for label in self.all_labels:
            predictions[label] = np.array(predictions[label])
        
        return predictions
    
    def predict(self, test_data: pd.DataFrame, text_column: str = "TEXT") -> Dict[str, np.ndarray]:
        """
        Make binary predictions with the model.
        
        Args:
            test_data: Test data
            text_column: Name of the column containing text
            
        Returns:
            Dictionary with binary predictions (0/1) for each label
        """
        return self._get_predictions(test_data, text_column, return_proba=False)
    
    def predict_proba(self, test_data: pd.DataFrame, text_column: str = "TEXT") -> Dict[str, np.ndarray]:
        """
        Make probability predictions with the model.
        
        Args:
            test_data: Test data
            text_column: Name of the column containing text
            
        Returns:
            Dictionary with probability scores (0.0-1.0) for each label
        """
        return self._get_predictions(test_data, text_column, return_proba=True)
    
    def save_model(self) -> None:
        """Save the model and tokenizer."""
        model_dir = self.model_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'labels': self.all_labels,
            'model_config': {
                'model_pretrained': self.model_pretrained,
                'max_length': self.max_length,
                'dropout_rate': self.dropout_rate,
                'num_labels': self.num_labels
            },
            'label_binarizers': self.label_binarizers
        }, self.model_path)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self, model_path: Path) -> None:
        """Load a saved model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.all_labels = checkpoint['labels']
        self.label_binarizers = checkpoint['label_binarizers']
        
        print(f"Model loaded from {model_path}")
    
    def format_predictions(
        self, 
        predictions: Dict[str, Any], 
        test_data: pd.DataFrame, 
        skipped_indices: List[int] = None,
        id_column: str = "PMID"
    ) -> pd.DataFrame:
        """
        Format predictions into a dataframe.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            skipped_indices: List of indices that were skipped (unused)
            id_column: Name of the column containing document IDs
            
        Returns:
            DataFrame with document IDs and predicted labels
        """
        results = []
        doc_ids = test_data[id_column].tolist()
        
        for i, doc_id in enumerate(doc_ids):
            predicted_labels = []
            
            for label in self.all_labels:
                if predictions[label][i] == 1:  # Positive prediction
                    predicted_labels.append(label)
            
            results.append({
                id_column: doc_id,
                'predicted_labels': ','.join(predicted_labels)
            })
        
        return pd.DataFrame(results)
    
    def save_predictions(
        self, 
        predictions: Dict[str, Any], 
        test_data: pd.DataFrame, 
        skipped_indices: List[int] = None,
        output_file: Path = None,
        id_column: str = "PMID"
    ) -> None:
        """
        Save predictions to a file.
        
        Args:
            predictions: Model predictions
            test_data: Test data
            skipped_indices: List of indices that were skipped (unused)
            output_file: Path to save predictions
            id_column: Name of the column containing document IDs
        """
        results_df = self.format_predictions(predictions, test_data, skipped_indices, id_column)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in results_df.iterrows():
                f.write(f"{row[id_column]}\t{row['predicted_labels']}\n")
        
        print(f"Predictions saved to {output_file}")
