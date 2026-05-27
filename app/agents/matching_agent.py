from difflib import SequenceMatcher
import re
 
 
class MatchingAgent:
 
    def similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
 
    def is_id_like(self, col):
        col = col.lower()
        return any(x in col for x in ["id", "key", "code", "number", "no"])
 
    # 🔥 GENERIC TOKENIZER
    def tokenize(self, col):
        col = col.lower()
 
        # split camelCase + snake_case + spaces
        tokens = re.findall(r'[a-zA-Z]+', col)
 
        return tokens
 
    # 🔥 REMOVE GENERIC WORDS
    def clean_tokens(self, tokens):
        stopwords = {
            "id", "key", "code", "number", "no", "num",
            "type", "value", "name", "desc", "description"
        }
        return [t for t in tokens if t not in stopwords]
 
    # 🔥 DOMAIN-AGNOSTIC ENTITY MATCH
    def same_entity(self, c1, c2):
        tokens1 = self.clean_tokens(self.tokenize(c1))
        tokens2 = self.clean_tokens(self.tokenize(c2))
 
        if not tokens1 or not tokens2:
            return False
 
        # 🔥 DIRECT TOKEN OVERLAP
        if set(tokens1).intersection(set(tokens2)):
            return True
 
        # 🔥 SOFT MATCH (semantic-ish via similarity)
        for t1 in tokens1:
            for t2 in tokens2:
                if SequenceMatcher(None, t1, t2).ratio() > 0.8:
                    return True
 
        return False
 
    def match(self, table_columns, key_map, file_map=None):
        matches = []
 
        tables = list(table_columns.keys())
 
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
 
                t1 = tables[i]
                t2 = tables[j]
 
                cols1 = table_columns[t1]
                cols2 = table_columns[t2]
 
                for c1 in cols1:
                    for c2 in cols2:
 
                        # 🔥 SKIP SAME TABLE
                        if t1 == t2:
                            continue
 
                        similarity = self.similarity(c1, c2)
 
                        is_exact = c1.lower() == c2.lower()
 
                        is_valid_id_match = (
                            self.is_id_like(c1)
                            and self.is_id_like(c2)
                            and self.same_entity(c1, c2)  # 🔥 now generic
                        )
 
                        is_semantic = similarity > 0.6  # 🔥 slightly stricter
 
                        # ❌ REJECT WEAK MATCHES
                        if not (is_exact or is_valid_id_match or is_semantic):
                            continue
 
                        matches.append({
                            "table1": t1,
                            "table2": t2,
                            "col1": c1,
                            "col2": c2,
                            "score": similarity
                        })
 
        return matches
