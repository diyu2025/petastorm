#!/usr/bin/env python3
"""
Script to print all factors that might affect random shuffle behavior in Petastorm.
This helps debug non-reproducible randomization across different nodes/runs.
"""

import os
import sys
import platform
import hashlib
import time
from datetime import datetime
import numpy as np
import random
from petastorm import make_reader
from petastorm.reader import Reader


def print_system_factors():
    """Print system-level factors that might affect randomization."""
    print("=" * 60)
    print("SYSTEM-LEVEL FACTORS")
    print("=" * 60)
    
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy random module: {np.random.__name__}")
    print(f"Process ID: {os.getpid()}")
    print(f"Thread ID: {os.getpid()}")  # In Python, thread ID is more complex
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current timestamp: {time.time()}")
    print(f"Current datetime: {datetime.now()}")
    
    # Environment variables that might affect randomization
    env_vars = ['PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 
                'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
    print(f"\nEnvironment variables:")
    for var in env_vars:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    print()


def print_numpy_random_state():
    """Print current NumPy random state information."""
    print("=" * 60)
    print("NUMPY RANDOM STATE")
    print("=" * 60)
    
    # Get current random state
    state = np.random.get_state()
    print(f"Random number generator: {state[0]}")
    print(f"State array shape: {state[1].shape}")
    print(f"State array first 10 values: {state[1][:10]}")
    print(f"Position in state: {state[2]}")
    print(f"Has cached Gaussian: {state[3]}")
    print(f"Cached Gaussian value: {state[4]}")
    
    # Test a few random numbers
    print(f"\nNext 5 random integers (0-100): {[np.random.randint(0, 100) for _ in range(5)]}")
    print()


def print_python_random_state():
    """Print current Python random state information."""
    print("=" * 60)  
    print("PYTHON RANDOM STATE")
    print("=" * 60)
    
    state = random.getstate()
    print(f"Random number generator: {state[0]}")
    print(f"State tuple length: {len(state[1])}")
    print(f"State tuple first 10 values: {state[1][:10]}")
    print(f"Position: {state[2]}")
    
    # Test a few random numbers
    print(f"\nNext 5 random integers (0-100): {[random.randint(0, 100) for _ in range(5)]}")
    print()


def print_petastorm_reader_factors(dataset_url=None, **reader_kwargs):
    """Print Petastorm Reader configuration factors that affect shuffling."""
    print("=" * 60)
    print("PETASTORM READER SHUFFLE FACTORS")
    print("=" * 60)
    
    # Default reader parameters that affect shuffling
    default_params = {
        'seed': None,
        'shuffle_rows': False,
        'shuffle_row_groups': True,
        'shuffle_row_drop_partitions': 1,
        'cur_shard': None,
        'shard_count': None,
        'shard_seed': None,  # Deprecated
        'num_epochs': 1,
        'reader_pool_type': 'thread',
        'workers_count': 10,
    }
    
    # Merge with provided parameters
    params = {**default_params, **reader_kwargs}
    
    print("Reader Parameters Affecting Shuffling:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print(f"\nDataset URL: {dataset_url}")
    
    if dataset_url:
        try:
            # Try to get dataset metadata
            print(f"\nAttempting to read dataset metadata...")
            with make_reader(dataset_url, num_epochs=1) as reader:
                print(f"  Dataset schema fields: {list(reader.schema.fields.keys())}")
                print(f"  Dataset path: {reader.dataset.paths}")
                if hasattr(reader.dataset, 'metadata') and reader.dataset.metadata:
                    print(f"  Number of row groups: {reader.dataset.metadata.num_row_groups}")
                    if reader.dataset.metadata.num_row_groups > 0:
                        print(f"  First row group num_rows: {reader.dataset.metadata.row_group(0).num_rows}")
                        
        except Exception as e:
            print(f"  Error reading dataset: {e}")
    
    print()


def print_worker_factors():
    """Print worker-specific factors that affect shuffling."""
    print("=" * 60)
    print("WORKER PROCESS FACTORS")
    print("=" * 60)
    
    print("Worker Pool Configuration:")
    print(f"  Default workers_count: 10")
    print(f"  Available reader_pool_types: ['thread', 'process', 'dummy']")
    
    print("\nWorker-level Randomization:")
    print("  - Each worker gets same seed but processes different row groups")
    print("  - PyDictReaderWorker uses pandas.DataFrame.sample() for row shuffling")
    print("  - ArrowReaderWorker uses np.random.permutation() for row shuffling")
    print("  - ConcurrentVentilator shuffles row group order")
    
    print("\nFactors affecting worker randomization:")
    print("  - Worker ID (0 to workers_count-1)")
    print("  - Row group assignment order")
    print("  - Timing of row group processing")
    print("  - Thread/process scheduling")
    print()


def print_dataset_factors(dataset_url=None):
    """Print dataset-specific factors that affect shuffling."""
    print("=" * 60)
    print("DATASET-SPECIFIC FACTORS")
    print("=" * 60)
    
    if not dataset_url:
        print("No dataset URL provided - showing general factors")
        print("Dataset factors that affect shuffling:")
        print("  - Number of parquet files")
        print("  - Number of row groups per file")
        print("  - Rows per row group")
        print("  - File naming/ordering")
        print("  - Partitioning scheme")
        print("  - File system (local vs HDFS vs S3)")
        print()
        return
    
    try:
        from pyarrow import parquet as pq
        dataset = pq.ParquetDataset(dataset_url)
        
        print(f"Dataset path: {dataset_url}")
        print(f"Dataset paths: {dataset.paths}")
        print(f"Dataset pieces: {len(dataset.pieces)}")
        
        if dataset.metadata:
            print(f"Total row groups: {dataset.metadata.num_row_groups}")
            print(f"Total rows: {dataset.metadata.num_rows}")
            
            # Row group information
            row_group_sizes = []
            for i in range(min(5, dataset.metadata.num_row_groups)):  # First 5 row groups
                rg = dataset.metadata.row_group(i)
                row_group_sizes.append(rg.num_rows)
                print(f"  Row group {i}: {rg.num_rows} rows")
            
            if len(row_group_sizes) > 0:
                print(f"Row group size stats: min={min(row_group_sizes)}, max={max(row_group_sizes)}, avg={sum(row_group_sizes)/len(row_group_sizes):.1f}")
        
        # File path hashing (affects sharding)
        print(f"\nFile path hashes (first 3 files):")
        for i, piece in enumerate(dataset.pieces[:3]):
            path_hash = hashlib.md5(piece.path.encode('utf-8')).hexdigest()[:8]
            print(f"  {piece.path}: {path_hash}")
            
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
    
    print()


def print_shuffle_algorithm_details():
    """Print details about the shuffling algorithms used."""
    print("=" * 60)
    print("SHUFFLE ALGORITHM DETAILS")
    print("=" * 60)
    
    print("1. Row Group Shuffling (shuffle_row_groups=True):")
    print("   - Performed by ConcurrentVentilator")
    print("   - Uses np.random.RandomState(seed).shuffle()")
    print("   - Shuffles the order of row groups before processing")
    print("   - Happens once per epoch at the beginning")
    
    print("\n2. Row Shuffling (shuffle_rows=True):")
    print("   - Performed by individual workers")
    print("   - PyDictReaderWorker: pandas.DataFrame.sample(frac=1, random_state=seed)")
    print("   - ArrowReaderWorker: np.random.permutation() with RandomState")
    print("   - Shuffles rows within each row group")
    
    print("\n3. Row Drop Partitioning (shuffle_row_drop_partitions > 1):")
    print("   - Splits each row group into N partitions")
    print("   - Each worker processes one partition")
    print("   - Increases randomization by breaking row group boundaries")
    
    print("\n4. Sharding (cur_shard, shard_count):")
    print("   - Filters row groups based on index % shard_count == cur_shard")
    print("   - If seed provided, shuffles row groups before sharding")
    print("   - Uses Python's random.Random(seed).shuffle()")
    
    print("\n5. Application-level Shuffling:")
    print("   - RandomShufflingBuffer for additional randomization")
    print("   - Uses np.random.randint() for random selection")
    print("   - Buffer size affects randomization quality")
    
    print()


def demonstrate_seed_effects(dataset_url=None):
    """Demonstrate how different seeds affect the same dataset."""
    print("=" * 60)
    print("SEED EFFECT DEMONSTRATION")
    print("=" * 60)
    
    if not dataset_url:
        print("No dataset URL provided - skipping demonstration")
        print()
        return
    
    seeds_to_test = [None, 42, 123, 42]  # Test None, different seeds, and repeated seed
    
    print("Testing different seeds on the same dataset:")
    print("(Reading first 5 IDs from each configuration)")
    
    for i, seed in enumerate(seeds_to_test):
        try:
            with make_reader(dataset_url, seed=seed, shuffle_rows=True, 
                           shuffle_row_groups=True, num_epochs=1) as reader:
                ids = []
                for j, row in enumerate(reader):
                    if hasattr(row, 'id'):
                        ids.append(row.id)
                    elif isinstance(row, dict) and 'id' in row:
                        ids.append(row['id'])
                    else:
                        ids.append(f"row_{j}")
                    
                    if len(ids) >= 5:
                        break
                
                print(f"  Seed {seed}: {ids}")
                
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
    
    print()


def print_debugging_tips():
    """Print tips for debugging shuffle behavior."""
    print("=" * 60)
    print("DEBUGGING TIPS")
    print("=" * 60)
    
    print("To ensure reproducible shuffling across nodes:")
    print()
    print("1. Set explicit seeds:")
    print("   reader = make_reader(url, seed=42, shuffle_rows=True, shuffle_row_groups=True)")
    print()
    print("2. Use deterministic node-specific seeds:")
    print("   node_seed = base_seed + node_id")
    print("   # or better: node_seed = [node_id, base_seed]")
    print()
    print("3. Check environment consistency:")
    print("   - Same Python/NumPy versions")
    print("   - Same PYTHONHASHSEED across nodes")
    print("   - Same dataset access pattern")
    print()
    print("4. Use SeedSequence for parallel safety:")
    print("   from numpy.random import SeedSequence")
    print("   ss = SeedSequence(master_seed)")
    print("   child_seeds = ss.spawn(num_nodes)")
    print()
    print("5. Monitor these factors:")
    print("   - System time differences")
    print("   - Process/thread IDs")
    print("   - File system access order")
    print("   - Network latency affecting timing")
    print()
    print("6. Test reproducibility:")
    print("   - Run same configuration multiple times")
    print("   - Compare outputs across nodes")
    print("   - Use small datasets for quick testing")
    print()


def main():
    """Main function to print all shuffle factors."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Print all factors affecting Petastorm shuffle behavior")
    parser.add_argument("--dataset-url", help="Path to Petastorm dataset for analysis")
    parser.add_argument("--seed", type=int, help="Seed to use for demonstrations")
    parser.add_argument("--shuffle-rows", action="store_true", help="Enable row shuffling")
    parser.add_argument("--shuffle-row-groups", action="store_true", default=True, help="Enable row group shuffling")
    parser.add_argument("--workers-count", type=int, default=10, help="Number of workers")
    
    args = parser.parse_args()
    
    print("PETASTORM SHUFFLE BEHAVIOR ANALYSIS")
    print("=" * 60)
    print(f"Analysis timestamp: {datetime.now()}")
    print()
    
    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Set global seeds to: {args.seed}")
        print()
    
    # Print all factors
    print_system_factors()
    print_numpy_random_state()
    print_python_random_state()
    
    reader_kwargs = {
        'seed': args.seed,
        'shuffle_rows': args.shuffle_rows,
        'shuffle_row_groups': args.shuffle_row_groups,
        'workers_count': args.workers_count,
    }
    
    print_petastorm_reader_factors(args.dataset_url, **reader_kwargs)
    print_worker_factors()
    print_dataset_factors(args.dataset_url)
    print_shuffle_algorithm_details()
    demonstrate_seed_effects(args.dataset_url)
    print_debugging_tips()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 