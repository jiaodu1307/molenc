#!/usr/bin/env python3
"""Command-line interface for MolEnc.

This module provides a command-line interface for the MolEnc library,
allowing users to encode molecules, check dependencies, and manage environments
from the command line.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from . import MolEncoder, check_dependencies, EnvironmentManager
    from .core.exceptions import MolEncError, EncoderNotFoundError
    from .preprocessing import preprocess_smiles_list
except ImportError:
    # Fallback for direct execution
    import molenc
    MolEncoder = molenc.MolEncoder
    check_dependencies = molenc.check_dependencies
    EnvironmentManager = molenc.EnvironmentManager
    from molenc.core.exceptions import MolEncError, EncoderNotFoundError
    from molenc.preprocessing import preprocess_smiles_list


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def encode_command(args: argparse.Namespace) -> int:
    """Handle the encode command."""
    try:
        # Read input SMILES
        if args.input == '-':
            smiles_list = [line.strip() for line in sys.stdin if line.strip()]
        else:
            with open(args.input, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        
        if not smiles_list:
            print("Error: No SMILES found in input", file=sys.stderr)
            return 1
        
        # Preprocess if requested
        if args.preprocess:
            print(f"Preprocessing {len(smiles_list)} SMILES...", file=sys.stderr)
            smiles_list, _ = preprocess_smiles_list(
                smiles_list,
                standardize=True,
                validate=True,
                filter_molecules=True
            )
            print(f"After preprocessing: {len(smiles_list)} SMILES", file=sys.stderr)
        
        # Initialize encoder
        print(f"Initializing {args.encoder} encoder...", file=sys.stderr)
        encoder_kwargs = {}
        if args.config:
            with open(args.config, 'r') as f:
                encoder_kwargs = json.load(f)
        
        encoder = MolEncoder(args.encoder, **encoder_kwargs)
        
        # Encode molecules
        print(f"Encoding {len(smiles_list)} molecules...", file=sys.stderr)
        if args.batch_size and args.batch_size > 1:
            vectors = encoder.encode_batch(smiles_list, batch_size=args.batch_size)
        else:
            vectors = [encoder.encode(smiles) for smiles in smiles_list]
        
        # Output results
        if args.output == '-':
            output_file = sys.stdout
        else:
            output_file = open(args.output, 'w')
        
        try:
            if args.format == 'json':
                result = {
                    'encoder': args.encoder,
                    'smiles': smiles_list,
                    'vectors': [vec.tolist() for vec in vectors]
                }
                json.dump(result, output_file, indent=2)
            elif args.format == 'csv':
                import pandas as pd
                df = pd.DataFrame({
                    'smiles': smiles_list,
                    'vector': [vec.tolist() for vec in vectors]
                })
                df.to_csv(output_file, index=False)
            else:  # numpy
                import numpy as np
                np.save(output_file, vectors)
        finally:
            if args.output != '-':
                output_file.close()
        
        print(f"Successfully encoded {len(smiles_list)} molecules", file=sys.stderr)
        return 0
        
    except EncoderNotFoundError as e:
        print(f"Error: Encoder not found - {e}", file=sys.stderr)
        return 1
    except MolEncError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def list_encoders_command(args: argparse.Namespace) -> int:
    """Handle the list-encoders command."""
    try:
        from .core.registry import get_available_encoders
        
        encoders = get_available_encoders()
        
        if args.format == 'json':
            print(json.dumps(encoders, indent=2))
        else:
            print("Available Encoders:")
            print("=" * 50)
            for name, info in encoders.items():
                status = "✓" if info.get('available', False) else "✗"
                print(f"{status} {name:20} - {info.get('description', 'No description')}")
                if not info.get('available', False) and 'missing_deps' in info:
                    print(f"    Missing: {', '.join(info['missing_deps'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error listing encoders: {e}", file=sys.stderr)
        return 1


def check_deps_command(args: argparse.Namespace) -> int:
    """Handle the check-deps command."""
    try:
        features = args.features if args.features else ['core']
        
        print("Checking dependencies...")
        print("=" * 50)
        
        results = check_dependencies(features, verbose=True)
        
        if args.format == 'json':
            print(json.dumps(results, indent=2))
        else:
            # Print summary
            core_ok = results.get('core_satisfied', False)
            print(f"\nCore dependencies: {'✓ OK' if core_ok else '✗ Missing'}")
            
            if 'missing' in results:
                missing = results['missing']
                if missing.get('core'):
                    print(f"Missing core: {', '.join(missing['core'])}")
                if missing.get('optional'):
                    print(f"Missing optional: {', '.join(missing['optional'])}")
        
        return 0 if results.get('core_satisfied', False) else 1
        
    except Exception as e:
        print(f"Error checking dependencies: {e}", file=sys.stderr)
        return 1


def env_command(args: argparse.Namespace) -> int:
    """Handle environment management commands."""
    try:
        env_manager = EnvironmentManager()
        
        if args.env_action == 'list':
            envs = env_manager.list_environments()
            if args.format == 'json':
                print(json.dumps(envs, indent=2))
            else:
                print("Available Environments:")
                print("=" * 50)
                for name, info in envs.items():
                    status = "✓" if info.get('active', False) else " "
                    print(f"{status} {name:20} - {info.get('description', 'No description')}")
        
        elif args.env_action == 'create':
            if not args.name:
                print("Error: Environment name required for create", file=sys.stderr)
                return 1
            
            success = env_manager.create_environment(
                args.name,
                features=args.features or [],
                description=args.description
            )
            
            if success:
                print(f"Successfully created environment '{args.name}'")
                return 0
            else:
                print(f"Failed to create environment '{args.name}'", file=sys.stderr)
                return 1
        
        elif args.env_action == 'activate':
            if not args.name:
                print("Error: Environment name required for activate", file=sys.stderr)
                return 1
            
            success = env_manager.activate_environment(args.name)
            
            if success:
                print(f"Successfully activated environment '{args.name}'")
                return 0
            else:
                print(f"Failed to activate environment '{args.name}'", file=sys.stderr)
                return 1
        
        elif args.env_action == 'remove':
            if not args.name:
                print("Error: Environment name required for remove", file=sys.stderr)
                return 1
            
            success = env_manager.remove_environment(args.name)
            
            if success:
                print(f"Successfully removed environment '{args.name}'")
                return 0
            else:
                print(f"Failed to remove environment '{args.name}'", file=sys.stderr)
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error managing environments: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='molenc',
        description='MolEnc - Unified molecular encoding library',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # Encode command
    encode_parser = subparsers.add_parser(
        'encode',
        help='Encode molecules from SMILES'
    )
    encode_parser.add_argument(
        'encoder',
        help='Encoder to use (e.g., morgan, maccs, unimol)'
    )
    encode_parser.add_argument(
        '--input', '-i',
        default='-',
        help='Input file with SMILES (default: stdin)'
    )
    encode_parser.add_argument(
        '--output', '-o',
        default='-',
        help='Output file (default: stdout)'
    )
    encode_parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv', 'numpy'],
        default='json',
        help='Output format (default: json)'
    )
    encode_parser.add_argument(
        '--config', '-c',
        help='JSON config file for encoder parameters'
    )
    encode_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for encoding'
    )
    encode_parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Preprocess SMILES before encoding'
    )
    
    # List encoders command
    list_parser = subparsers.add_parser(
        'list-encoders',
        help='List available encoders'
    )
    list_parser.add_argument(
        '--format', '-f',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    
    # Check dependencies command
    deps_parser = subparsers.add_parser(
        'check-deps',
        help='Check dependencies'
    )
    deps_parser.add_argument(
        '--features',
        nargs='*',
        help='Features to check (default: core)'
    )
    deps_parser.add_argument(
        '--format', '-f',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    
    # Environment management command
    env_parser = subparsers.add_parser(
        'env',
        help='Manage environments'
    )
    env_parser.add_argument(
        'env_action',
        choices=['list', 'create', 'activate', 'remove'],
        help='Environment action'
    )
    env_parser.add_argument(
        '--name', '-n',
        help='Environment name'
    )
    env_parser.add_argument(
        '--features',
        nargs='*',
        help='Features to install in environment'
    )
    env_parser.add_argument(
        '--description', '-d',
        help='Environment description'
    )
    env_parser.add_argument(
        '--format', '-f',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    
    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'encode':
            return encode_command(args)
        elif args.command == 'list-encoders':
            return list_encoders_command(args)
        elif args.command == 'check-deps':
            return check_deps_command(args)
        elif args.command == 'env':
            return env_command(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())