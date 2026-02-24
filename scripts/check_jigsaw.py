import pandas as pd
import os

folder = 'datasets/jigsaw_toxic'
files_to_check = ['train.csv', 'jigsaw-toxic-comment-train.csv', 'validation.csv', 'test.csv']

for f in files_to_check:
    path = os.path.join(folder, f)
    if os.path.exists(path):
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
            print(f"--- {f} ({size_mb:.2f} MB) ---")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            if "toxic" in df.columns:
                print(f"Toxic sum: {df['toxic'].sum()}")
            if "comment_text" in df.columns:
                print(f"Sample text:")
                print(df['comment_text'].head(2).tolist())
            print()
        except Exception as e:
            print(f"Error on {f}: {e}")
