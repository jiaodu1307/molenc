"""
Unified batch processing utilities for molecular encoding.

This module provides centralized batch processing functions to avoid code duplication
across different modules.
"""

import logging
import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .exceptions import InvalidSMILESError, DependencyError


def batch_process_with_fallback(items: List[Any],
                               process_func: Callable,
                               fallback_func: Optional[Callable] = None,
                               n_jobs: int = 1,
                               chunk_size: Optional[int] = None,
                               return_errors: bool = False) -> Union[List[Any], Dict[str, Any]]:
    """
    Generic batch processing with optional parallel execution and error handling.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        fallback_func: Optional fallback function for failed items
        n_jobs: Number of parallel jobs (1 for sequential)
        chunk_size: Size of chunks for parallel processing
        return_errors: Whether to return error information
        
    Returns:
        List of processed items or dict with results and errors
    """
    if not items:
        return [] if not return_errors else {'results': [], 'errors': []}
    
    results = []
    errors = []
    
    if n_jobs == 1:
        # Sequential processing
        for i, item in enumerate(items):
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                if fallback_func:
                    try:
                        result = fallback_func(item)
                        results.append(result)
                    except Exception as fallback_error:
                        errors.append({'index': i, 'item': item, 'error': str(fallback_error)})
                        results.append(None)
                else:
                    errors.append({'index': i, 'item': item, 'error': str(e)})
                    results.append(None)
    else:
        # Parallel processing
        if chunk_size is None:
            chunk_size = max(1, len(items) // n_jobs)
        
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(_process_chunk, chunk, process_func, fallback_func, chunk_idx * chunk_size)
                futures.append(future)
            
            for future in as_completed(futures):
                chunk_results, chunk_errors = future.result()
                results.extend(chunk_results)
                errors.extend(chunk_errors)
    
    if return_errors:
        return {'results': results, 'errors': errors}
    return results


def _process_chunk(chunk: List[Any],
                  process_func: Callable,
                  fallback_func: Optional[Callable],
                  start_index: int) -> tuple:
    """Process a chunk of items."""
    results = []
    errors = []
    
    for i, item in enumerate(chunk):
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            if fallback_func:
                try:
                    result = fallback_func(item)
                    results.append(result)
                except Exception as fallback_error:
                    errors.append({
                        'index': start_index + i,
                        'item': item,
                        'error': str(fallback_error)
                    })
                    results.append(None)
            else:
                errors.append({
                    'index': start_index + i,
                    'item': item,
                    'error': str(e)
                })
                results.append(None)
    
    return results, errors


def batch_normalize_smiles(smiles_list: List[str],
                          canonical: bool = True,
                          remove_invalid: bool = True,
                          n_jobs: int = 1) -> List[str]:
    """
    Normalize a batch of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        canonical: Whether to return canonical SMILES
        remove_invalid: Whether to remove invalid SMILES
        n_jobs: Number of parallel jobs
        
    Returns:
        List of normalized SMILES strings
    """
    from .utils import normalize_smiles
    
    def process_smiles(smiles: str) -> Optional[str]:
        return normalize_smiles(smiles, canonical)
    
    results = batch_process_with_fallback(
        smiles_list,
        process_smiles,
        n_jobs=n_jobs
    )
    
    if remove_invalid:
        return [smiles for smiles in results if smiles is not None]
    else:
        return [smiles if smiles is not None else original 
                for smiles, original in zip(results, smiles_list)]


def batch_encode_safe(items: List[Any],
                     encoder_func: Callable,
                     skip_invalid: bool = True,
                     log_errors: bool = True,
                     n_jobs: int = 1) -> List[Any]:
    """
    Safely encode a batch of items with error handling.
    
    Args:
        items: List of items to encode
        encoder_func: Encoding function
        skip_invalid: Whether to skip invalid items
        log_errors: Whether to log errors
        n_jobs: Number of parallel jobs
        
    Returns:
        List of encoded items (may contain None for failed items)
    """
    def safe_encode(item):
        try:
            return encoder_func(item)
        except Exception as e:
            if log_errors:
                logging.warning(f"Failed to encode item {item}: {e}")
            if skip_invalid:
                return None
            raise
    
    return batch_process_with_fallback(
        items,
        safe_encode,
        n_jobs=n_jobs
    )


def save_batch_results(results: List[Any],
                      file_path: str,
                      format: str = 'npy',
                      metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save batch processing results to file.
    
    Args:
        results: List of results to save
        file_path: Output file path
        format: File format ('npy', 'npz', 'json')
        metadata: Optional metadata to save
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'npy':
        np.save(file_path, np.array(results))
    elif format == 'npz':
        if metadata:
            np.savez(file_path, results=results, **metadata)
        else:
            np.savez(file_path, results=results)
    elif format == 'json':
        import json
        data = {'results': results}
        if metadata:
            data['metadata'] = metadata
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_batch_results(file_path: str,
                      format: str = 'npy') -> Union[List[Any], Dict[str, Any]]:
    """
    Load batch processing results from file.
    
    Args:
        file_path: Input file path
        format: File format ('npy', 'npz', 'json')
        
    Returns:
        Loaded results or dict with results and metadata
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if format == 'npy':
        return np.load(file_path).tolist()
    elif format == 'npz':
        data = np.load(file_path)
        return {key: data[key].tolist() if key == 'results' else data[key] 
                for key in data.files}
    elif format == 'json':
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")