import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMAgent:

    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key  = os.getenv("AZURE_OPENAI_API_KEY", "")

        if not endpoint or not api_key:
            raise ValueError("Missing credentials. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")

        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

        self.model = os.getenv("AZURE_OPENAI_MODEL", "Llama-3.3-70B-Instruct")

    def _safe_parse(self, content: str):
        try:
            return json.loads(content)
        except Exception as e:
            print("[LLM PARSE ERROR]", e)
            cleaned = content
            cleaned = re.sub(r"```json|```", "", cleaned).strip()
            cleaned = cleaned.replace("'", '"')
            cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            try:
                return json.loads(cleaned)
            except Exception as e2:
                print("[FINAL PARSE FAILED]", e2)
                return {"edges": []}

    def _trim_metadata(self, metadata_map, max_cols=15):
        trimmed = {}
        for table, cols in metadata_map.items():
            trimmed[table] = dict(list(cols.items())[:max_cols])
        return trimmed

    def generate_kg(self, metadata_map):
        metadata_map = self._trim_metadata(metadata_map)
        prompt = f"""You are a data modeling expert. Analyze these datasets and detect relationships.
Return STRICT JSON ONLY:
{{"edges": [{{"source": "table1.csv","target": "table2.csv","relationship": "1-N","source_column": "col","target_column": "col","confidence": 0.9,"reason": "explanation"}}]}}
RULES: double quotes only, no trailing commas, no markdown, return only JSON.
Metadata:
{json.dumps(metadata_map, indent=2)}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"```json|```", "", content).strip()
        return self._safe_parse(content)

    def explain_relationship(self, table1, col1, stats1, table2, col2, stats2, relationship, confidence):
        prompt = f"""You are a data analyst. Explain in ONE sentence why a relationship exists between:
Table1: {table1}, Column: {col1}, Stats: {json.dumps(stats1)}
Table2: {table2}, Column: {col2}, Stats: {json.dumps(stats2)}
Relationship: {relationship}, Confidence: {confidence}
Be concise. No extra text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only a single sentence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()