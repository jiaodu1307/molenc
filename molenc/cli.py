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
    from .core.encoder_factory import (
        get_encoder_factory,
        EncoderMode,
        EncoderConfig,
    )
    from .core.exceptions import MolEncError, EncoderNotFoundError
    from .preprocessing import preprocess_smiles_list
    from .environments.advanced_dependency_manager import DependencyLevel
except ImportError:
    import molenc
    from molenc.core.encoder_factory import (
        get_encoder_factory,
        EncoderMode,
        EncoderConfig,
    )
    from molenc.core.exceptions import MolEncError, EncoderNotFoundError
    from molenc.preprocessing import preprocess_smiles_list
    from molenc.environments.advanced_dependency_manager import DependencyLevel


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
        
        # Initialize encoder via factory
        print(f"Initializing {args.encoder} encoder (mode={args.mode})...", file=sys.stderr)
        encoder_kwargs = {}
        if args.preset:
            from .core.config import Config
            preset = Config.from_preset(args.preset)
            encoder_kwargs.update(preset.to_dict())
        if args.config:
            with open(args.config, 'r') as f:
                encoder_kwargs = json.load(f)
        if args.backend == 'http' and 'base_url' not in encoder_kwargs:
            import os
            env_url = os.environ.get('MOLENC_REMOTE_URL')
            if env_url:
                encoder_kwargs['base_url'] = env_url
        
        factory = get_encoder_factory()
        config = EncoderConfig(
            mode=EncoderMode(args.mode),
            allow_fallback=True,
            auto_install=args.auto_install,
            max_dependency_level=DependencyLevel[args.dependency_level],
            user_preferences={'backend': args.backend} if args.backend else None,
        )
        encoder = factory.create_encoder(args.encoder, config=config, **encoder_kwargs)
        
        # Encode molecules
        print(f"Encoding {len(smiles_list)} molecules...", file=sys.stderr)
        if args.batch_size and args.batch_size > 1:
            bs = args.batch_size
            all_vecs = []
            for i in range(0, len(smiles_list), bs):
                chunk = smiles_list[i:i+bs]
                arr = encoder.encode_batch(chunk)
                for row in arr:
                    all_vecs.append(row)
            vectors = all_vecs
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
                    'vectors': [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in vectors]
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
        factory = get_encoder_factory()
        encoders = factory.get_available_encoders()
        
        if args.format == 'json':
            print(json.dumps(encoders, indent=2))
        else:
            print("Available Encoders:")
            print("=" * 50)
            for name, variants in encoders.items():
                status = "✓" if variants else "✗"
                variants_str = ', '.join(variants) if variants else 'None'
                print(f"{status} {name:20} - variants: {variants_str}")
        
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


def status_command(args: argparse.Namespace) -> int:
    try:
        from .core.encoder_factory import get_encoder_factory
        factory = get_encoder_factory()
        report = factory.get_encoder_status(args.type) if args.type else factory.get_encoder_status()
        print(report)
        return 0
    except Exception as e:
        print(f"Error showing status: {e}", file=sys.stderr)
        return 1


def compare_command(args: argparse.Namespace) -> int:
    try:
        # Read inputs
        if args.input == '-':
            smiles_list = [line.strip() for line in sys.stdin if line.strip()]
        else:
            with open(args.input, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        if not smiles_list:
            print("Error: No SMILES found in input", file=sys.stderr)
            return 1

        from .core.encoder_factory import get_encoder_factory, EncoderMode, EncoderConfig
        from .environments.advanced_dependency_manager import DependencyLevel
        factory = get_encoder_factory()

        # Prepare encoders
        kwargs1 = {}
        kwargs2 = {}
        if args.preset1:
            from .core.config import Config
            kwargs1.update(Config.from_preset(args.preset1).to_dict())
        if args.preset2:
            from .core.config import Config
            kwargs2.update(Config.from_preset(args.preset2).to_dict())

        # Inject base_url for http backend from env if needed
        import os
        if args.backend1 == 'http' and 'base_url' not in kwargs1:
            env_url = os.environ.get('MOLENC_REMOTE_URL')
            if env_url:
                kwargs1['base_url'] = env_url
        if args.backend2 == 'http' and 'base_url' not in kwargs2:
            env_url = os.environ.get('MOLENC_REMOTE_URL')
            if env_url:
                kwargs2['base_url'] = env_url

        config1 = EncoderConfig(
            mode=EncoderMode(args.mode1),
            allow_fallback=True,
            auto_install=False,
            max_dependency_level=DependencyLevel.FULL,
            user_preferences={'backend': args.backend1} if args.backend1 else None,
        )
        config2 = EncoderConfig(
            mode=EncoderMode(args.mode2),
            allow_fallback=True,
            auto_install=False,
            max_dependency_level=DependencyLevel.FULL,
            user_preferences={'backend': args.backend2} if args.backend2 else None,
        )

        enc1 = factory.create_encoder(args.encoder1, config=config1, **kwargs1)
        enc2 = factory.create_encoder(args.encoder2, config=config2, **kwargs2)

        # Encode and compare
        bs = args.batch_size or 0
        if bs and bs > 1:
            v1 = enc1.encode_batch(smiles_list)
            v2 = enc2.encode_batch(smiles_list)
        else:
            v1 = np.vstack([enc1.encode(s) for s in smiles_list])
            v2 = np.vstack([enc2.encode(s) for s in smiles_list])

        # Metrics and report
        import numpy as np, time, psutil
        report: Dict[str, Any] = {
            'smiles_count': len(smiles_list),
            'shape_a': list(v1.shape),
            'shape_b': list(v2.shape),
        }
        # Timing: rough measure using encode times already tracked in factory not available here, so remeasure single pass
        t0 = time.time(); _ = enc1.encode_batch(smiles_list); t1 = time.time(); _ = enc2.encode_batch(smiles_list); t2 = time.time()
        report['encode_time_a'] = t1 - t0
        report['encode_time_b'] = t2 - t1
        # Memory snapshot
        process = psutil.Process()
        mem_info = process.memory_info()
        report['rss_bytes'] = mem_info.rss
        # Difference
        if v1.shape == v2.shape and v1.size > 0:
            report['l2_diff'] = np.linalg.norm(v1 - v2, axis=1).tolist()
        else:
            report['shape_mismatch'] = [list(v1.shape), list(v2.shape)]
        # Optional CSV output
        if hasattr(args, 'format') and args.format == 'csv':
            import csv
            writer = csv.writer(sys.stdout)
            writer.writerow(['smiles_count','shape_a','shape_b','encode_time_a','encode_time_b','rss_bytes'])
            writer.writerow([report['smiles_count'], report['shape_a'], report['shape_b'], report['encode_time_a'], report['encode_time_b'], report['rss_bytes']])
            if 'l2_diff' in report:
                writer.writerow(['l2_diff'] + report['l2_diff'])
            else:
                writer.writerow(['shape_mismatch'] + report.get('shape_mismatch', []))
        else:
            print(json.dumps(report, indent=2))
        return 0
    except Exception as e:
        print(f"Error comparing encoders: {e}", file=sys.stderr)
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
        '--mode',
        choices=[m.value for m in EncoderMode],
        default='auto',
        help='Encoder selection mode'
    )
    encode_parser.add_argument(
        '--auto-install',
        action='store_true',
        help='Auto-install missing dependencies when possible'
    )
    encode_parser.add_argument(
        '--dependency-level',
        choices=[lvl.name for lvl in DependencyLevel],
        default='FULL',
        help='Max dependency installation level'
    )
    encode_parser.add_argument(
        '--backend',
        choices=['local', 'venv', 'conda', 'docker', 'http'],
        help='Preferred execution backend'
    )
    encode_parser.add_argument(
        '--preset',
        help='Load encoder configuration preset'
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

    # Encoder status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show encoder selection status report'
    )
    status_parser.add_argument(
        '--type', '-t',
        help='Encoder type (e.g., fingerprint, transformer, gcn)'
    )

    # Compare encoders command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare outputs from two encoders on the same input'
    )
    compare_parser.add_argument('encoder1', help='First encoder (e.g., morgan)')
    compare_parser.add_argument('encoder2', help='Second encoder (e.g., morgan)')
    compare_parser.add_argument('--input', '-i', default='-', help='Input file or stdin (-)')
    compare_parser.add_argument('--mode1', default='auto', help='Mode for first encoder')
    compare_parser.add_argument('--mode2', default='auto', help='Mode for second encoder')
    compare_parser.add_argument('--backend1', choices=['local','venv','conda','docker','http'])
    compare_parser.add_argument('--backend2', choices=['local','venv','conda','docker','http'])
    compare_parser.add_argument('--preset1')
    compare_parser.add_argument('--preset2')
    compare_parser.add_argument('--batch-size', '-b', type=int)
    compare_parser.add_argument('--format', choices=['json','csv'], default='json')
    
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
        elif args.command == 'status':
            return status_command(args)
        elif args.command == 'compare':
            return compare_command(args)
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