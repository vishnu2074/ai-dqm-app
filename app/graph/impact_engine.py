from typing import List
 
# ── Weight constants ──────────────────────────────────────────────────────────
W_TABLE        = 3   # downstream actual dataset files (.csv nodes)
W_VIEW         = 5   # downstream view nodes
W_REPORT       = 8   # downstream reports/dashboards
W_DQ_RULE      = 4   # downstream DQ rules
W_BLAST_RADIUS = 2   # per real downstream node (folders excluded)
W_FOLDER       = 0   # structural/organizational folder nodes — no business impact
 
SEVERITY_CRITICAL = 60
SEVERITY_HIGH     = 35
SEVERITY_MEDIUM   = 15
 
 
def _get_graph():
    """Always use the live Azure-based lineage graph."""
    from app.graph.lineage_engine import LineageEngine
    return LineageEngine.get_full_graph()
 
 
def _is_folder_node(node_id: str, node_type: str) -> bool:
    """
    Folder nodes are structural path segments — not real datasets.
    A real dataset file always ends with _csv (from slugified blob path).
    Source/table nodes that don't end in _csv are folder nodes.
    DQ rules, views, reports are never folders.
    """
    if node_type in ("dq_rule", "view", "report"):
        return False
    if node_type in ("source", "table"):
        return not node_id.endswith("_csv")
    return False
 
 
class ImpactEngine:
 
    @staticmethod
    def calculate_downstream(node_id: str) -> List[str]:
        """BFS — returns ALL downstream node IDs from the Azure graph."""
        graph   = _get_graph()
        visited = set()
        stack   = [node_id]
 
        while stack:
            current = stack.pop()
            for edge in graph["edges"]:
                if edge["source"] == current and edge["target"] not in visited:
                    visited.add(edge["target"])
                    stack.append(edge["target"])
 
        return list(visited)
 
    @staticmethod
    def calculate_impact(node_id: str) -> dict:
        graph      = _get_graph()
        downstream = ImpactEngine.calculate_downstream(node_id)
 
        affected_datasets = []   # real .csv dataset files only
        affected_folders  = []   # structural folder nodes
        affected_reports  = []
        affected_rules    = []
 
        count_tables  = 0   # real dataset files
        count_folders = 0   # folder nodes — weight 0
        count_views   = 0
        count_reports = 0
        count_rules   = 0
 
        for node in graph["nodes"]:
            if node["id"] not in downstream:
                continue
            ntype    = node.get("type", "")
            node_id_ = node["id"]
 
            if _is_folder_node(node_id_, ntype):
                affected_folders.append(node_id_)
                count_folders += 1
 
            elif ntype in ("table", "source"):
                affected_datasets.append(node_id_)
                count_tables += 1
 
            elif ntype == "view":
                affected_datasets.append(node_id_)
                count_views += 1
 
            elif ntype == "report":
                affected_reports.append(node_id_)
                count_reports += 1
 
            elif ntype == "dq_rule":
                affected_rules.append(node_id_)
                count_rules += 1
 
        # Blast radius counts only real nodes (folders excluded)
        real_downstream  = count_tables + count_views + count_reports + count_rules
        total_downstream = len(downstream)
 
        score = (
            count_tables  * W_TABLE        +
            count_views   * W_VIEW         +
            count_reports * W_REPORT       +
            count_rules   * W_DQ_RULE      +
            real_downstream * W_BLAST_RADIUS
        )
 
        if score >= SEVERITY_CRITICAL:
            severity = "Critical"
        elif score >= SEVERITY_HIGH:
            severity = "High"
        elif score >= SEVERITY_MEDIUM:
            severity = "Medium"
        else:
            severity = "Low"
 
        score_breakdown = {
            "tables_contribution":       count_tables  * W_TABLE,
            "views_contribution":        count_views   * W_VIEW,
            "reports_contribution":      count_reports * W_REPORT,
            "dq_rules_contribution":     count_rules   * W_DQ_RULE,
            "blast_radius_contribution": real_downstream * W_BLAST_RADIUS,
        }
 
        return {
            "node_id":           node_id,
            "affected_datasets": affected_datasets,
            "affected_folders":  affected_folders,
            "affected_reports":  affected_reports,
            "affected_rules":    affected_rules,
            "impact_score":      score,
            "severity":          severity,
            "downstream_count":  total_downstream,
            "real_downstream":   real_downstream,
            "folder_count":      count_folders,
            "score_breakdown":   score_breakdown,
        }