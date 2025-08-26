"""MolEnc - 分子编码器统一库

一个统一的分子编码器库，集成了多种分子表示学习方法，提供简单易用的API接口。
"""

__version__ = "0.1.0"
__author__ = "MolEnc Team"
__email__ = "molenc@example.com"

# Core imports
from .core.base import BaseEncoder
from .core.registry import MolEncoder, register_encoder
from .core.exceptions import MolEncError, EncoderNotFoundError, InvalidSMILESError

# Environment management
from .environments import (
    check_dependencies,
    AdvancedDependencyManager,
    get_dependency_manager,
    check_encoder_readiness
)

# Isolation environment management
from .isolation import (
    EnvironmentManager as IsolationEnvironmentManager,
    get_environment_manager as get_isolation_manager
)

# Encoder imports (with error handling for optional dependencies)
try:
    from .encoders.descriptors.fingerprints import MorganEncoder, MACCSEncoder
except ImportError:
    MorganEncoder = None
    MACCSEncoder = None

try:
    from .encoders.representations.sequence.chemberta import ChemBERTaEncoder
except ImportError:
    ChemBERTaEncoder = None

try:
    from .encoders.representations.graph import GCNEncoder
except ImportError:
    GCNEncoder = None

try:
    from .encoders.representations.multimodal import UniMolEncoder
except ImportError:
    UniMolEncoder = None

# Preprocessing imports
from .preprocessing import (
    SMILESStandardizer,
    MolecularFilters,
    SMILESValidator,
    preprocess_smiles_list
)

__all__ = [
    # Core
    "BaseEncoder",
    "MolEncoder", 
    "register_encoder",
    "MolEncError",
    "EncoderNotFoundError",
    "InvalidSMILESError",
    
    # Environment
    "check_dependencies",
    "AdvancedDependencyManager",
    "get_dependency_manager",
    "check_encoder_readiness",
    
    # Isolation Environment
    "IsolationEnvironmentManager",
    "get_isolation_manager",
    
    # Encoders (conditionally available)
    "MorganEncoder",
    "MACCSEncoder",
    "ChemBERTaEncoder",
    "GCNEncoder",
    "UniMolEncoder",
    
    # Preprocessing
    "SMILESStandardizer",
    "MolecularFilters",
    "SMILESValidator",
    "preprocess_smiles_list",
]