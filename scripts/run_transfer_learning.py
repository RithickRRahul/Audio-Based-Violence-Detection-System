"""
Standalone runner for transfer learning baseline evaluation.
Writes all output to a log file directly from Python to avoid
PowerShell's broken stderr redirect.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Suppress TF warnings BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set up file-based logging
log_path = os.path.join(project_root, 'transfer_learning_log.txt')
log_file = open(log_path, 'w', encoding='utf-8')

# Monkey-patch print to also write to log
import builtins
original_print = builtins.print
def patched_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    original_print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()
builtins.print = patched_print

if __name__ == '__main__':
    patched_print("=" * 60)
    patched_print("Transfer Learning Baseline Evaluation - Starting")
    patched_print("=" * 60)
    
    try:
        from scripts.evaluate_transfer_learning import run_transfer_learning_evaluation
        
        results = run_transfer_learning_evaluation()
        
        patched_print("\n" + "=" * 60)
        patched_print("FINAL RESULTS SUMMARY")
        patched_print("=" * 60)
        for ds, models in results.items():
            patched_print(f"\n{ds}:")
            for model, metrics in models.items():
                patched_print(f"  {model}: Acc={metrics['accuracy']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
        
        patched_print("\n" + "=" * 60)
        patched_print("COMPLETE!")
        patched_print("=" * 60)
        
    except Exception as e:
        patched_print(f"FATAL ERROR: {e}")
        import traceback
        patched_print(traceback.format_exc())
        sys.exit(1)
    finally:
        log_file.close()
