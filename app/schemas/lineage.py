from pydantic import BaseModel
from typing import List, Optional


class Node(BaseModel):
    id: str
    name: str
    type: str
    quality_score: Optional[float] = None


class Edge(BaseModel):
    source: str
    target: str
    relationship: str = "depends_on"


class GraphResponse(BaseModel):
    nodes: List[Node]
    edges: List[Edge]