class KeyDetectionAgent:

    def detect_keys(self, metadata):

        keys = []

        for col, stats in metadata.items():

            row_count = stats.get("row_count", 0)
            distinct = stats.get("distinct_count", 0)
            nulls = stats.get("null_count", 0)

            if row_count > 0 and distinct == row_count and nulls == 0:
                keys.append(col)

        return keys