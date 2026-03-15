"""
Standalone runner for CPU baseline evaluation.
Writes all output to a log file directly from Python to avoid
PowerShell's broken stderr redirect that kills the process.
"""
import sys
import os

# Add project root to path so 'src' module can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Suppress TF warnings BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set up file-based logging
log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'baseline_eval_log.txt')
log_file = open(log_path, 'w', encoding='utf-8')

def log(msg):
    """Print to both console and log file."""
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

if __name__ == '__main__':
    log("=" * 60)
    log("CPU Baseline Evaluation - Starting")
    log("=" * 60)
    
    try:
        from scripts.evaluate_actual_baselines import run_real_baselines
        log("Successfully imported evaluation module")
        
        # Monkey-patch print to also write to log
        import builtins
        original_print = builtins.print
        def patched_print(*args, **kwargs):
            msg = ' '.join(str(a) for a in args)
            original_print(msg, flush=True)
            log_file.write(msg + '\n')
            log_file.flush()
        builtins.print = patched_print
        
        log("Starting full evaluation on CPU...")
        run_real_baselines()
        log("=" * 60)
        log("Evaluation COMPLETE!")
        log("=" * 60)
        
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
    finally:
        log_file.close()
