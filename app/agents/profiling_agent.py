# app/agents/profiling_agent.py

import pandas as pd


class ProfilingAgent:
    """
    Profiles a CSV file and returns column-level metadata.
    Used by KnowledgeGraphService to build metadata_map for the LLM.
    """

    def profile(self, file_path: str) -> dict:
        """
        Returns {column_name: {row_count, distinct_count, null_percentage, sample_values}}
        for every column in the CSV at file_path.
        """
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"[ProfilingAgent] Could not read {file_path}: {e}")
            return {}

        total_rows = len(df)
        metadata = {}

        for col in df.columns:
            series       = df[col]
            null_count   = int(series.isna().sum())
            distinct     = int(series.nunique(dropna=True))
            null_pct     = round(null_count / total_rows * 100, 2) if total_rows else 0.0
            sample_vals  = [str(v) for v in series.dropna().head(5).tolist()]

            metadata[col] = {
                "row_count":        total_rows,
                "distinct_count":   distinct,
                "null_percentage":  null_pct,
                "null_count":       null_count,
                "sample_values":    sample_vals,
                "dtype":            str(series.dtype),
            }

        return metadata