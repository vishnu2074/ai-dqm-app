from typing import List, Dict

SEVERITY_CRITICAL = 60
SEVERITY_HIGH     = 35
SEVERITY_MEDIUM   = 15

RISK_BY_TYPE = {
    "table":   "High",
    "source":  "High",
    "view":    "High",
    "report":  "Critical",
    "dq_rule": "Medium",
}


def _get_graph():
    """Always use the live Azure-based lineage graph."""
    from app.graph.lineage_engine import LineageEngine
    return LineageEngine.get_full_graph()


class DQFailureEngine:

    @staticmethod
    def get_parent_datasets(rule_id: str) -> List[str]:
        """
        Find all data nodes (table/view/source) that feed INTO this DQ rule.
        In the Azure graph: dataset_file_node → dq_rule
        """
        graph   = _get_graph()
        parents = []
        for edge in graph["edges"]:
            if edge["target"] == rule_id:
                source_node = next(
                    (n for n in graph["nodes"] if n["id"] == edge["source"]), None
                )
                if source_node and source_node["type"] in ("table", "view", "source"):
                    parents.append(edge["source"])
        return parents

    @staticmethod
    def get_downstream_of(node_id: str, graph: dict) -> List[str]:
        """BFS — all downstream node IDs from a given node."""
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
    def simulate_failure(rule_id: str) -> dict:
        """
        Simulate what breaks when a DQ rule fails.

        In the Azure blob graph the propagation is:
          rule fails → parent dataset is untrustworthy
          → anything downstream of that dataset is at risk

        Because the Azure graph is hierarchical (source → layer → domain → file → rule),
        upstream folder nodes are ALSO surfaced so business impact is clear.
        """
        graph       = _get_graph()
        nodes_by_id: Dict[str, dict] = {n["id"]: n for n in graph["nodes"]}

        rule_node = nodes_by_id.get(rule_id)
        if not rule_node or rule_node.get("type") != "dq_rule":
            return {"error": f"Node '{rule_id}' is not a DQ rule or does not exist."}

        # Step 1 — find parent data nodes
        parent_datasets = DQFailureEngine.get_parent_datasets(rule_id)

        if not parent_datasets:
            return {
                "rule_id":           rule_id,
                "rule_name":         rule_node.get("name", rule_id),
                "parent_datasets":   [],
                "affected_nodes":    [],
                "at_risk_reports":   [],
                "at_risk_datasets":  [],
                "propagation_paths": [],
                "impact_score":      0,
                "severity":          "Low",
                "summary":           "This DQ rule has no connected parent dataset.",
            }

        # Step 2 — collect all downstream nodes per parent
        all_affected: Dict[str, dict] = {}
        propagation_paths = []

        for parent_id in parent_datasets:
            downstream_ids = DQFailureEngine.get_downstream_of(parent_id, graph)
            # Exclude the rule itself
            downstream_ids = [d for d in downstream_ids if d != rule_id]

            propagation_paths.append({
                "from_dataset": parent_id,
                "from_name":    nodes_by_id.get(parent_id, {}).get("name", parent_id),
                "affected":     downstream_ids,
            })

            for nid in downstream_ids:
                node = nodes_by_id.get(nid)
                if not node:
                    continue
                ntype = node.get("type", "unknown")
                all_affected[nid] = {
                    "node_id": nid,
                    "name":    node.get("name", nid),
                    "type":    ntype,
                    "risk":    RISK_BY_TYPE.get(ntype, "Low"),
                    "via":     parent_id,
                }

        # Step 3 — categorise
        at_risk_reports  = [v for v in all_affected.values() if v["type"] == "report"]
        at_risk_datasets = [v for v in all_affected.values() if v["type"] in ("table", "view", "source")]
        at_risk_rules    = [v for v in all_affected.values() if v["type"] == "dq_rule"]

        # Step 4 — score
        score = (
            len(at_risk_datasets) * 5  +
            len(at_risk_reports)  * 10 +
            len(at_risk_rules)    * 3  +
            len(all_affected)     * 2
        )

        if score >= SEVERITY_CRITICAL:
            severity = "Critical"
        elif score >= SEVERITY_HIGH:
            severity = "High"
        elif score >= SEVERITY_MEDIUM:
            severity = "Medium"
        else:
            severity = "Low"

        # Step 5 — summary
        summary_parts = []
        if at_risk_reports:
            names = [r["name"] for r in at_risk_reports]
            summary_parts.append(f"{len(at_risk_reports)} report(s) at risk: {', '.join(names)}")
        if at_risk_datasets:
            names = [d["name"] for d in at_risk_datasets]
            summary_parts.append(f"{len(at_risk_datasets)} dataset(s) affected: {', '.join(names)}")
        summary = ". ".join(summary_parts) if summary_parts else "No downstream impact detected."

        return {
            "rule_id":           rule_id,
            "rule_name":         rule_node.get("name", rule_id),
            "parent_datasets":   parent_datasets,
            "affected_nodes":    list(all_affected.values()),
            "at_risk_reports":   at_risk_reports,
            "at_risk_datasets":  at_risk_datasets,
            "at_risk_rules":     at_risk_rules,
            "propagation_paths": propagation_paths,
            "impact_score":      score,
            "severity":          severity,
            "summary":           summary,
        }