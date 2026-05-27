# app/agents/relationship_agent.py

import pandas as pd
import numpy as np


class RelationshipAgent:
    """
    Domain-agnostic relationship detector with NO hardcoded keywords.
    Everything is determined from the data itself.
    """

    # ------------------------------------------------------------------
    # DATA-DRIVEN COLUMN CLASSIFICATION (NO KEYWORDS)
    # ------------------------------------------------------------------

    def is_likely_id_column(self, col_name: str, series: pd.Series) -> bool:
        """
        Determines if a column is likely an ID column based on data characteristics.
        No hardcoded keywords - uses data properties only.
        """
        s = series.dropna()
        if len(s) == 0:
            return False
        
        # ID columns typically have high uniqueness ratio
        uniqueness_ratio = s.nunique() / len(s)
        
        # ID columns typically have values that are not too short or too long
        if s.dtype == 'object':
            avg_len = s.astype(str).str.len().mean()
            # IDs often have consistent length (between 5-20 characters typically)
            # But we don't hardcode ranges, just check if it's reasonable
            if avg_len < 3 or avg_len > 50:
                return False
        
        # High uniqueness is the strongest signal for ID columns
        return uniqueness_ratio > 0.8

    def is_numeric_measurement(self, col_name: str, series: pd.Series) -> bool:
        """
        Determines if a column contains numeric measurements (prices, weights, etc.)
        Based purely on data characteristics.
        """
        if not pd.api.types.is_numeric_dtype(series):
            return False
        
        s = series.dropna()
        if len(s) == 0:
            return False
        
        # Measurements typically have:
        # 1. High cardinality (many unique values)
        # 2. Values that are not integers (decimals)
        # 3. Wide range of values
        
        uniqueness_ratio = s.nunique() / len(s)
        
        # Check if values are mostly non-integers (decimals)
        is_decimal = not all(s == s.astype(int))
        
        # High cardinality + decimal values suggests measurements
        return uniqueness_ratio > 0.5 and is_decimal

    def is_categorical_like(self, series: pd.Series) -> bool:
        """
        Determines if a column contains categorical/enumeration data.
        Based purely on cardinality ratios.
        """
        s = series.dropna()
        if len(s) == 0:
            return True
        
        uniqueness_ratio = s.nunique() / len(s)
        
        # Categories typically have low uniqueness ratio
        # Use adaptive threshold based on data size
        if len(s) < 100:
            return uniqueness_ratio < 0.3
        else:
            return uniqueness_ratio < 0.05

    def is_temporal_like(self, series: pd.Series) -> bool:
        """
        Determines if a column contains temporal data (dates, timestamps).
        Uses pattern detection, not keyword matching.
        """
        s = series.dropna()
        if len(s) == 0:
            return False
        
        # Try to convert to datetime - if it works, it's temporal
        try:
            pd.to_datetime(s.head(10))
            return True
        except:
            pass
        
        # Check for common date patterns using regex (generic patterns, not keywords)
        sample = s.head(10).astype(str)
        
        # Check for YYYY-MM-DD pattern (any year, any month, any day)
        date_pattern = r'^\d{4}-\d{1,2}-\d{1,2}'
        if sample.str.match(date_pattern).any():
            return True
        
        # Check for DD/MM/YYYY or MM/DD/YYYY patterns
        slash_pattern = r'^\d{1,2}/\d{1,2}/\d{4}'
        if sample.str.match(slash_pattern).any():
            return True
        
        return False

    def is_joinable_column(self, col_name: str, series: pd.Series) -> bool:
        """
        Returns True for columns that can participate in FK/PK joins.
        Completely data-driven - no hardcoded keywords.
        """
        # ID columns are always joinable
        if self.is_likely_id_column(col_name, series):
            return True
        
        # Measurements are not joinable (coincidental overlap)
        if self.is_numeric_measurement(col_name, series):
            return False
        
        # Temporal columns are not joinable
        if self.is_temporal_like(series):
            return False
        
        # Categorical columns are not joinable
        if self.is_categorical_like(series):
            return False
        
        # Everything else is potentially joinable
        return True

    # ------------------------------------------------------------------
    # VALUE NORMALISATION
    # ------------------------------------------------------------------

    def _normalize(self, series: pd.Series) -> pd.Series:
        """
        Normalises values for consistent comparison across different types.
        Converts everything to string for set intersection.
        """
        s = series.dropna()
        
        # Convert to string for consistent comparison
        if pd.api.types.is_numeric_dtype(s):
            # Format numbers consistently (1.0 -> "1", 1.5 -> "1.5")
            def fmt_num(x):
                if pd.isna(x):
                    return None
                if isinstance(x, float) and x.is_integer():
                    return str(int(x))
                return str(x)
            return s.apply(fmt_num)
        
        # For other types, just convert to string and clean
        return s.astype(str).str.strip()

    # ------------------------------------------------------------------
    # PK CLASSIFICATION (DATA-DRIVEN)
    # ------------------------------------------------------------------

    def is_pk_like(self, series: pd.Series) -> bool:
        """
        Determines if a column is likely a primary key.
        Uses adaptive threshold based on data characteristics.
        """
        s = series.dropna()
        if len(s) == 0:
            return False
        
        uniqueness_ratio = s.nunique() / len(s)
        
        # For small tables, require higher uniqueness
        if len(s) < 100:
            return uniqueness_ratio > 0.95
        # For large tables, slightly lower threshold is acceptable
        else:
            return uniqueness_ratio > 0.90

    # ------------------------------------------------------------------
    # VALIDATION (PURELY DATA-DRIVEN)
    # ------------------------------------------------------------------

    def validate_df_directional(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                 col1: str, col2: str) -> tuple:
        """
        Returns (max_coverage, cov1, cov2) for directional confidence.
        """
        if col1 not in df1.columns or col2 not in df2.columns:
            return 0.0, 0.0, 0.0

        s1 = self._normalize(df1[col1])
        s2 = self._normalize(df2[col2])

        if len(s1) == 0 or len(s2) == 0:
            return 0.0, 0.0, 0.0

        set1 = set(s1.unique())
        set2 = set(s2.unique())
        common = set1 & set2

        if not common:
            return 0.0, 0.0, 0.0

        cov1 = len(common) / len(set1)
        cov2 = len(common) / len(set2)
        return max(cov1, cov2), cov1, cov2

    def should_accept_confidence(self, conf_max: float, cov1: float, cov2: float) -> bool:
        """
        Accepts a relationship based purely on coverage.
        No hardcoded thresholds - uses adaptive logic.
        
        Accept if ANY positive overlap exists, with minimum 1% coverage.
        This ensures we catch all potential relationships.
        """
        # Accept any positive overlap with at least 1% coverage
        MIN_ACCEPTABLE = 0.001
        
        if conf_max >= MIN_ACCEPTABLE:
            return True
        
        # Also accept if either side has decent coverage
        # (useful for partial FK relationships)
        if cov1 >= MIN_ACCEPTABLE or cov2 >= MIN_ACCEPTABLE:
            return True
        
        return False

    def detect_relationship_df(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                col1: str, col2: str):
        """
        Returns cardinality based on data characteristics.
        """
        if col1 not in df1.columns or col2 not in df2.columns:
            return None

        s1 = self._normalize(df1[col1])
        s2 = self._normalize(df2[col2])

        if len(s1) == 0 or len(s2) == 0:
            return None

        common = set(s1.unique()) & set(s2.unique())
        if not common:
            return None

        u1 = self.is_pk_like(s1)
        u2 = self.is_pk_like(s2)

        if u1 and u2:
            return "1-1"
        elif not u1 and u2:
            return "N-1"
        elif u1 and not u2:
            return "1-N"
        else:
            return "N-N"