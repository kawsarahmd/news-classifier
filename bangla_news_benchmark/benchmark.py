"""
Benchmark script for Bangla news classification models
Trains and evaluates all models, saves results to CSV
"""
import os
import csv
import argparse
from pathlib import Path
import pandas as pd

from data.load_dataset import BanglaNewsDataset
from models.bilstm import train_bilstm
from models.cnn_lstm import train_cnn_lstm
from models.attention_bilstm import train_attention_bilstm
from models.xlmr import train_xlmr
from models.banglabert import train_banglabert
from utils.training_utils import set_seed


def save_results_to_csv(results_list, filepath='results.csv'):
    """
    Save benchmark results to CSV file

    Args:
        results_list: List of result dictionaries
        filepath: Path to save CSV file
    """
    # Prepare data for CSV
    csv_data = []
    for result in results_list:
        metrics = result['test_metrics']
        csv_data.append({
            'Model': result['model_name'],
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'Model Path': result['model_path']
        })

    # Write to CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Model Path'])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nResults saved to {filepath}")


def print_leaderboard(results_list):
    """
    Print formatted leaderboard of results

    Args:
        results_list: List of result dictionaries
    """
    print("\n" + "="*80)
    print("BANGLA NEWS CLASSIFICATION BENCHMARK - LEADERBOARD")
    print("="*80)

    # Sort by F1 score (descending)
    sorted_results = sorted(results_list, key=lambda x: x['test_metrics']['f1'], reverse=True)

    # Print header
    print(f"{'Rank':<6} {'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)

    # Print results
    for rank, result in enumerate(sorted_results, 1):
        metrics = result['test_metrics']
        print(f"{rank:<6} {result['model_name']:<20} "
              f"{metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

    print("="*80)


def run_benchmark(models_to_run=None, seed=42):
    """
    Run complete benchmark for all models

    Args:
        models_to_run: List of model names to run (None = all)
        seed: Random seed for reproducibility

    Returns:
        List of result dictionaries
    """
    # Set global seed
    set_seed(seed)

    # Create checkpoints directory
    Path('checkpoints').mkdir(exist_ok=True)

    # Initialize dataset
    print("="*80)
    print("BANGLA NEWS CLASSIFICATION BENCHMARK")
    print("="*80)

    dataset = BanglaNewsDataset()

    # Define all models
    all_models = {
        'bilstm': {
            'name': 'BiLSTM',
            'train_fn': train_bilstm,
            'data_prep': lambda: dataset.prepare_for_lstm(),
            'config': {
                'batch_size': 32,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'lr': 0.001,
                'num_epochs': 20,
                'patience': 5,
                'seed': seed,
                'model_save_path': 'checkpoints/bilstm_best.pt'
            }
        },
        'cnn-lstm': {
            'name': 'CNN-LSTM',
            'train_fn': train_cnn_lstm,
            'data_prep': lambda: dataset.prepare_for_lstm(),
            'config': {
                'batch_size': 32,
                'embedding_dim': 128,
                'num_filters': 100,
                'filter_sizes': [3, 4, 5],
                'lstm_hidden_dim': 128,
                'num_layers': 1,
                'dropout': 0.3,
                'lr': 0.001,
                'num_epochs': 20,
                'patience': 5,
                'seed': seed,
                'model_save_path': 'checkpoints/cnn_lstm_best.pt'
            }
        },
        'attention-bilstm': {
            'name': 'Attention-BiLSTM',
            'train_fn': train_attention_bilstm,
            'data_prep': lambda: dataset.prepare_for_lstm(),
            'config': {
                'batch_size': 32,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'lr': 0.001,
                'num_epochs': 20,
                'patience': 5,
                'seed': seed,
                'model_save_path': 'checkpoints/attention_bilstm_best.pt'
            }
        },
        'xlm-roberta': {
            'name': 'XLM-RoBERTa',
            'train_fn': train_xlmr,
            'data_prep': lambda: dataset.prepare_for_transformers("xlm-roberta-base"),
            'config': {
                'batch_size': 16,
                'lr': 2e-5,
                'weight_decay': 0.01,
                'num_epochs': 5,
                'patience': 3,
                'seed': seed,
                'model_save_path': 'checkpoints/xlmr_best'
            }
        },
        'banglabert': {
            'name': 'BanglaBERT',
            'train_fn': train_banglabert,
            'data_prep': lambda: dataset.prepare_for_transformers("csebuetnlp/banglabert"),
            'config': {
                'batch_size': 16,
                'lr': 2e-5,
                'weight_decay': 0.01,
                'num_epochs': 5,
                'patience': 3,
                'seed': seed,
                'model_save_path': 'checkpoints/banglabert_best'
            }
        }
    }

    # Filter models if specified
    if models_to_run:
        models_to_run = [m.lower() for m in models_to_run]
        all_models = {k: v for k, v in all_models.items() if k in models_to_run}

    # Train each model
    results = []

    for model_key, model_info in all_models.items():
        print("\n" + "="*80)
        print(f"Training {model_info['name']}...")
        print("="*80)

        try:
            # Prepare data
            data_dict = model_info['data_prep']()

            # Train model
            result = model_info['train_fn'](data_dict, model_info['config'])

            if result:  # Check if result is not None (for distributed training)
                results.append(result)
                print(f"\n{model_info['name']} training completed successfully!")

        except Exception as e:
            print(f"\nError training {model_info['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save and display results
    if results:
        save_results_to_csv(results, 'results.csv')
        print_leaderboard(results)

        # Also save as pandas DataFrame for easy analysis
        df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Accuracy': r['test_metrics']['accuracy'],
                'Precision': r['test_metrics']['precision'],
                'Recall': r['test_metrics']['recall'],
                'F1-Score': r['test_metrics']['f1']
            }
            for r in results
        ])
        df.to_csv('results_detailed.csv', index=False)
        print(f"\nDetailed results saved to results_detailed.csv")

    return results


def main():
    """
    Main function with argument parsing
    """
    parser = argparse.ArgumentParser(description='Benchmark Bangla news classification models')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['bilstm', 'cnn-lstm', 'attention-bilstm', 'xlm-roberta', 'banglabert'],
        help='Models to train (default: all)',
        default=None
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(models_to_run=args.models, seed=args.seed)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
