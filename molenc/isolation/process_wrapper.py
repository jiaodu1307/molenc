"""Process wrapper for isolated encoder execution.

This module provides a mechanism to execute encoders in isolated processes
with their own environments, avoiding dependency conflicts while maintaining
seamless integration with the main application.
"""

import os
import sys
import subprocess
import json
import tempfile
import logging
import pickle
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class ProcessWrapperError(Exception):
    """Exception raised for process wrapper errors."""
    pass


class ProcessWrapper:
    """Wrapper for executing encoders in isolated processes."""
    
    def __init__(self, env_path: Optional[Path] = None, python_exe: Optional[str] = None):
        self.env_path = env_path
        self.python_exe = python_exe or sys.executable
        self.logger = logging.getLogger(__name__)
        
    def _setup_environment(self, env_vars: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Set up environment variables for process execution."""
        # Start with current environment
        proc_env = os.environ.copy()
        
        # Add custom environment variables
        if env_vars:
            proc_env.update(env_vars)
            
        # If we have a virtual environment, adjust PATH and PYTHONPATH
        if self.env_path:
            if sys.platform == "win32":
                bin_dir = self.env_path / "Scripts"
            else:
                bin_dir = self.env_path / "bin"
                
            # Prepend virtual environment bin directory to PATH
            proc_env["PATH"] = str(bin_dir) + os.pathsep + proc_env.get("PATH", "")
            
            # Set VIRTUAL_ENV
            proc_env["VIRTUAL_ENV"] = str(self.env_path)
            
            # Remove PYTHONHOME if set
            proc_env.pop("PYTHONHOME", None)
            
            # Use the Python executable from the virtual environment
            self.python_exe = str(bin_dir / "python") if sys.platform != "win32" else str(bin_dir / "python.exe")
            
        return proc_env
    
    def execute_encoder(self, 
                       encoder_type: str,
                       smiles_data: Union[str, List[str]],
                       encoder_config: Optional[Dict[str, Any]] = None,
                       timeout: int = 300) -> np.ndarray:
        """
        Execute encoder in isolated process.
        
        Args:
            encoder_type: Type of encoder to use
            smiles_data: SMILES string or list of SMILES strings
            encoder_config: Configuration for the encoder
            timeout: Timeout in seconds
            
        Returns:
            Encoded vectors as numpy array
        """
        # Create temporary files for communication
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.json"
            output_file = temp_path / "output.pkl"
            error_file = temp_path / "error.txt"
            
            # Prepare input data
            input_data = {
                "encoder_type": encoder_type,
                "smiles_data": smiles_data,
                "encoder_config": encoder_config or {}
            }
            
            with open(input_file, "w") as f:
                json.dump(input_data, f)
            
            # Create execution script
            script_content = self._create_execution_script()
            script_file = temp_path / "execute_encoder.py"
            
            with open(script_file, "w") as f:
                f.write(script_content)
            
            # Set up environment
            env_vars = self._setup_environment()
            
            # Execute in subprocess
            try:
                cmd = [self.python_exe, str(script_file), 
                       str(input_file), str(output_file), str(error_file)]
                
                result = subprocess.run(
                    cmd,
                    env=env_vars,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                # Check if execution was successful
                if result.returncode != 0:
                    # Check for error output
                    if error_file.exists():
                        with open(error_file, "r") as f:
                            error_msg = f.read()
                        raise ProcessWrapperError(f"Encoder execution failed: {error_msg}")
                    else:
                        raise ProcessWrapperError(
                            f"Encoder execution failed with return code {result.returncode}: {result.stderr}"
                        )
                
                # Load results
                if output_file.exists():
                    with open(output_file, "rb") as f:
                        result_data = pickle.load(f)
                    
                    if result_data.get("success"):
                        return np.array(result_data["embeddings"], dtype=np.float32)
                    else:
                        raise ProcessWrapperError(f"Encoder error: {result_data.get('error')}")
                else:
                    raise ProcessWrapperError("No output file generated")
                    
            except subprocess.TimeoutExpired:
                raise ProcessWrapperError(f"Encoder execution timed out after {timeout} seconds")
            except Exception as e:
                raise ProcessWrapperError(f"Process execution failed: {e}")
    
    def _create_execution_script(self) -> str:
        """Create the script to be executed in the isolated process."""
        return '''
import sys
import json
import pickle
import numpy as np
from molenc import MolEncoder

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file output_file error_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    error_file = sys.argv[3]
    
    try:
        # Load input data
        with open(input_file, "r") as f:
            input_data = json.load(f)
        
        encoder_type = input_data["encoder_type"]
        smiles_data = input_data["smiles_data"]
        encoder_config = input_data["encoder_config"]
        
        # Create encoder
        encoder = MolEncoder(encoder_type, **encoder_config)
        
        # Execute encoding
        if isinstance(smiles_data, str):
            embeddings = encoder.encode(smiles_data)
        else:
            embeddings = encoder.encode_batch(smiles_data)
        
        # Save results
        result = {
            "success": True,
            "embeddings": embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        }
        
        with open(output_file, "wb") as f:
            pickle.dump(result, f)
            
    except Exception as e:
        # Save error
        error_result = {
            "success": False,
            "error": str(e)
        }
        
        with open(output_file, "wb") as f:
            pickle.dump(error_result, f)
        
        # Also write to error file for debugging
        with open(error_file, "w") as f:
            f.write(str(e))
        
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def execute_function(self, 
                        func_module: str,
                        func_name: str,
                        args: tuple = (),
                        kwargs: dict = None,
                        timeout: int = 300) -> Any:
        """
        Execute a function in isolated process.
        
        Args:
            func_module: Module containing the function
            func_name: Name of the function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds
            
        Returns:
            Function result
        """
        kwargs = kwargs or {}
        
        # Create temporary files for communication
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.pkl"
            output_file = temp_path / "output.pkl"
            error_file = temp_path / "error.txt"
            
            # Prepare input data
            input_data = {
                "func_module": func_module,
                "func_name": func_name,
                "args": args,
                "kwargs": kwargs
            }
            
            with open(input_file, "wb") as f:
                pickle.dump(input_data, f)
            
            # Create execution script
            script_content = self._create_function_execution_script()
            script_file = temp_path / "execute_function.py"
            
            with open(script_file, "w") as f:
                f.write(script_content)
            
            # Set up environment
            env_vars = self._setup_environment()
            
            # Execute in subprocess
            try:
                cmd = [self.python_exe, str(script_file), 
                       str(input_file), str(output_file), str(error_file)]
                
                result = subprocess.run(
                    cmd,
                    env=env_vars,
                    capture_output=True,
                    text=False,  # Binary data
                    timeout=timeout
                )
                
                # Check if execution was successful
                if result.returncode != 0:
                    # Check for error output
                    if error_file.exists():
                        with open(error_file, "r") as f:
                            error_msg = f.read()
                        raise ProcessWrapperError(f"Function execution failed: {error_msg}")
                    else:
                        raise ProcessWrapperError(
                            f"Function execution failed with return code {result.returncode}"
                        )
                
                # Load results
                if output_file.exists():
                    with open(output_file, "rb") as f:
                        result_data = pickle.load(f)
                    
                    if result_data.get("success"):
                        return result_data["result"]
                    else:
                        raise ProcessWrapperError(f"Function error: {result_data.get('error')}")
                else:
                    raise ProcessWrapperError("No output file generated")
                    
            except subprocess.TimeoutExpired:
                raise ProcessWrapperError(f"Function execution timed out after {timeout} seconds")
            except Exception as e:
                raise ProcessWrapperError(f"Process execution failed: {e}")
    
    def _create_function_execution_script(self) -> str:
        """Create the script to execute a function in isolated process."""
        return '''
import sys
import pickle
import importlib

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file output_file error_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    error_file = sys.argv[3]
    
    try:
        # Load input data
        with open(input_file, "rb") as f:
            input_data = pickle.load(f)
        
        func_module = input_data["func_module"]
        func_name = input_data["func_name"]
        args = input_data["args"]
        kwargs = input_data["kwargs"]
        
        # Import module and get function
        module = importlib.import_module(func_module)
        func = getattr(module, func_name)
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Save results
        result_data = {
            "success": True,
            "result": result
        }
        
        with open(output_file, "wb") as f:
            pickle.dump(result_data, f)
            
    except Exception as e:
        # Save error
        error_result = {
            "success": False,
            "error": str(e)
        }
        
        with open(output_file, "wb") as f:
            pickle.dump(error_result, f)
        
        # Also write to error file for debugging
        with open(error_file, "w") as f:
            f.write(str(e))
        
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


def create_process_wrapper(env_path: Optional[Path] = None, 
                          python_exe: Optional[str] = None) -> ProcessWrapper:
    """Create a process wrapper instance."""
    return ProcessWrapper(env_path, python_exe)