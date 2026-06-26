# cleaner.py
import pandas as pd

def clean_csv(df, text_columns):
    original_len = len(df)
    report = {}

    # Drop rows where ALL selected text columns are null/empty
    df = df.dropna(subset=text_columns, how="all")
    df = df[~df[text_columns].apply(
        lambda row: all(str(v).strip() == "" for v in row if pd.notna(v)), axis=1
    )]
    after_null = len(df)
    report["rows_dropped_null"] = original_len - after_null

    # Remove duplicate rows
    df = df.drop_duplicates()
    after_dedup = len(df)
    report["rows_dropped_duplicates"] = after_null - after_dedup

    report["original_rows"] = original_len
    report["clean_rows"] = len(df)
    df = df.reset_index(drop=True)

    return df, report