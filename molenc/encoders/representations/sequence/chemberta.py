"""ChemBERTa encoder implementation."""

import numpy as np
from typing import Optional, List, Dict, Any
import torch
from torch import Tensor

from molenc.core.base import BaseEncoder
from molenc.core.registry import register_encoder
from molenc.core.exceptions import InvalidSMILESError, EncoderInitializationError
from molenc.core.dependency_utils import require_dependencies
from molenc.core.encoder_utils import EncoderUtils
from molenc.core.encoder_mixins import (
    SMILESValidationMixin, 
    ParameterValidationMixin, 
    DeviceManagementMixin,
    ModelLoadingMixin
)
from molenc.core.exception_decorators import handle_encoding_errors, handle_initialization_errors, handle_batch_processing_errors


@register_encoder('chemberta')
@require_dependencies(['torch', 'transformers'], 'ChemBERTa')
class ChemBERTaEncoder(BaseEncoder, SMILESValidationMixin, ParameterValidationMixin, 
                      DeviceManagementMixin, ModelLoadingMixin):
    """ChemBERTa encoder.

    ChemBERTa is a RoBERTa-based transformer model pre-trained on molecular SMILES
    strings to learn contextualized molecular representations.
    """

    def __init__(self,
                 model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 max_length: int = 512,
                 pooling_strategy: str = "cls",
                 device: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize ChemBERTa encoder.

        Args:
            model_name: Name of the pre-trained ChemBERTa model (default: "seyonec/ChemBERTa-zinc-base-v1")
            max_length: Maximum sequence length (default: 512)
            pooling_strategy: How to pool token embeddings ("cls", "mean", "max")
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            **kwargs: Additional parameters passed to BaseEncoder
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # Validate parameters
        EncoderUtils.validate_positive_int(max_length, "max_length")
        EncoderUtils.validate_choice(pooling_strategy, ["cls", "mean", "max"], "pooling_strategy")



        # Set device
        self.device = EncoderUtils.setup_device(device)

        # Initialize model and tokenizer
        self._load_model()

    @handle_initialization_errors("ChemBERTa")
    def _load_model(self) -> None:
        """
        Load pre-trained ChemBERTa model and tokenizer.
        """
        from transformers import AutoTokenizer, AutoModel

        # Load ChemBERTa tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

        # Get hidden size
        self.hidden_size = self.model.config.hidden_size

    def _tokenize_smiles(self, smiles: str) -> Dict[str, Tensor]:
        """
        Tokenize SMILES string.

        Args:
            smiles: SMILES string to tokenize

        Returns:
            Tokenized inputs dictionary
        """
        inputs = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool token embeddings to get sentence-level representation.

        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, hidden_size]
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(
                -1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling_strategy == "max":
            # Max pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(
                -1).expand(hidden_states.size()).float()
            # Set padding tokens to large negative value
            hidden_states[mask_expanded == 0] = -1e9
            return torch.max(hidden_states, 1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    @handle_encoding_errors(reraise_as=InvalidSMILESError)
    def _encode_single(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string using ChemBERTa.

        Args:
            smiles: SMILES string to encode

        Returns:
            ChemBERTa embedding as numpy array

        Raises:
            InvalidSMILESError: If SMILES is invalid
        """
        # Basic SMILES validation
        if not smiles or not isinstance(smiles, str):
            raise InvalidSMILESError(smiles, "Invalid SMILES format")

        # Tokenize SMILES
        inputs = self._tokenize_smiles(smiles)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Pool embeddings
            pooled_embedding = self._pool_embeddings(
                hidden_states, inputs["attention_mask"]
            )

        # Convert to numpy
        embedding = pooled_embedding.cpu().numpy().squeeze()

        return embedding.astype(np.float32)

    @handle_batch_processing_errors(skip_invalid=True, log_skipped=True)
    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Encode a batch of SMILES strings efficiently.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Batch of embeddings as numpy array
        """
        if not smiles_list:
            return np.array([], dtype=np.float32)

        # Tokenize all SMILES
        all_inputs: List[Dict[str, Tensor]] = []
        for smiles in smiles_list:
            inputs = self._tokenize_smiles(smiles)
            all_inputs.append(inputs)

        # Batch the inputs
        batch_inputs = {
            "input_ids": torch.cat([inp["input_ids"] for inp in all_inputs], dim=0),
            "attention_mask": torch.cat([inp["attention_mask"] for inp in all_inputs], dim=0)
        }

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
            hidden_states = outputs.last_hidden_state

            # Pool embeddings
            pooled_embeddings = self._pool_embeddings(
                hidden_states, batch_inputs["attention_mask"]
            )

        # Convert to numpy
        embeddings = pooled_embeddings.cpu().numpy()

        return embeddings.astype(np.float32)

    def get_output_dim(self) -> int:
        """
        Get the output dimension of ChemBERTa embeddings.

        Returns:
            Hidden size of the model
        """
        return int(self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.

        Returns:
            Configuration dictionary
        """
        config: Dict[str, Any] = super().get_config()
        config.update({
            'model_name': self.model_name,
            'max_length': self.max_length,
            'pooling_strategy': self.pooling_strategy,
            'device': str(self.device)
        })
        return config

    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the embedding dimensions.

        Returns:
            List of feature names
        """
        return [f"chemberta_dim_{i}" for i in range(self.hidden_size)]

    def __repr__(self) -> str:
        return (
            f"ChemBERTaEncoder(model_name='{self.model_name}', "
            f"hidden_size={self.hidden_size}, pooling='{self.pooling_strategy}')"
        )