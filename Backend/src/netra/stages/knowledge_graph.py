from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from netra.core.base import BaseStage
from netra.core.registry import StageRegistry

logger = logging.getLogger("netra.knowledge_graph")

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
KB_DIR = _BACKEND_ROOT / "knowledge_base" / "graph"


@StageRegistry.register("knowledge_graph", variant="networkx")
@StageRegistry.register("knowledge_graph")
class GraphKBStage(BaseStage):
    """
    Builds and maintains a persistent knowledge graph from processed
    documents.  Extracts entities (persons, dates, amounts, PAN, addresses,
    document type) from LLM output and creates typed nodes + relationship
    edges using NetworkX with JSON persistence.
    """

    def _load_model(self) -> None:
        import networkx as nx

        self._nx = nx
        self._kb_dir = Path(
            self.config.get("params", {}).get("kb_dir", str(KB_DIR))
        )
        self._kb_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._kb_dir / "knowledge_graph.json"

        if self._graph_path.exists():
            with open(self._graph_path) as f:
                data = json.load(f)
            self._graph = nx.node_link_graph(data)
            logger.info(
                "Loaded graph: %d nodes, %d edges",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        else:
            self._graph = nx.DiGraph()

    def _save_graph(self) -> None:
        data = self._nx.node_link_data(self._graph)
        with open(self._graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _unload_model(self) -> None:
        if self._graph is not None:
            self._save_graph()
        self._graph = None

    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        llm_text = inputs.get("llm_output", {}).get("response", "")
        ocr_text = inputs.get("ocr_output", {}).get("text", "")
        image_path = inputs.get("image_path", "")
        target_lang = inputs.get("target_language", "")

        doc_id = hashlib.sha256(
            (ocr_text or image_path).encode()
        ).hexdigest()[:16]

        doc_node = f"doc:{doc_id}"
        self._graph.add_node(
            doc_node,
            label="Document",
            source=str(image_path),
            language=target_lang,
            text_preview=ocr_text[:200],
        )

        entities = self._extract_entities(llm_text, ocr_text)
        edges_created = 0

        for ent in entities:
            ent_id = f"{ent['type']}:{ent['id']}"
            self._graph.add_node(ent_id, label=ent["type"], **ent["props"])

            self._graph.add_edge(doc_node, ent_id, relation="HAS_ENTITY")
            edges_created += 1

        translated_text = inputs.get("nmt_output", {}).get("translated_text", "")
        if translated_text:
            trans_id = f"translation:{doc_id}:{target_lang}"
            self._graph.add_node(
                trans_id,
                label="Translation",
                language=target_lang,
                text_preview=translated_text[:200],
            )
            self._graph.add_edge(doc_node, trans_id, relation="TRANSLATED_TO")
            edges_created += 1

        self._link_similar_documents(doc_node, entities)
        self._save_graph()

        return {
            "doc_id": doc_id,
            "entities_extracted": len(entities),
            "edges_created": edges_created,
            "graph_nodes": self._graph.number_of_nodes(),
            "graph_edges": self._graph.number_of_edges(),
            "entities": entities,
        }

    def _extract_entities(
        self, llm_text: str, ocr_text: str,
    ) -> list[dict[str, Any]]:
        combined = f"{llm_text}\n{ocr_text}"
        entities: list[dict[str, Any]] = []
        seen: set[str] = set()

        pan_matches = re.findall(r"[A-Z]{5}\d{4}[A-Z]", combined)
        for pan in pan_matches:
            if pan not in seen:
                seen.add(pan)
                entities.append({
                    "type": "PAN",
                    "id": pan,
                    "props": {"number": pan},
                })

        amount_matches = re.findall(
            r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)", combined,
        )
        for amt_str in amount_matches:
            clean = amt_str.replace(",", "")
            key = f"amount:{clean}"
            if key not in seen:
                seen.add(key)
                entities.append({
                    "type": "Amount",
                    "id": clean,
                    "props": {"value": float(clean), "currency": "INR"},
                })

        date_patterns = [
            r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
            r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}",
        ]
        for pat in date_patterns:
            for m in re.finditer(pat, combined, re.IGNORECASE):
                date_str = m.group()
                key = f"date:{date_str}"
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "type": "Date",
                        "id": date_str.replace(" ", "_"),
                        "props": {"value": date_str},
                    })

        person_patterns = [
            r"(?:Shri|Smt|Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        ]
        for pat in person_patterns:
            for m in re.finditer(pat, combined):
                name = m.group(1).strip()
                key = f"person:{name}"
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "type": "Person",
                        "id": name.replace(" ", "_"),
                        "props": {"name": name},
                    })

        pincode_matches = re.findall(r"\b\d{6}\b", combined)
        for pin in pincode_matches:
            key = f"pin:{pin}"
            if key not in seen:
                seen.add(key)
                entities.append({
                    "type": "Address",
                    "id": pin,
                    "props": {"pincode": pin},
                })

        doc_types = [
            "tax notice", "aadhaar", "pan card", "ration card",
            "land record", "court notice", "government order",
            "birth certificate", "death certificate", "income certificate",
            "caste certificate", "domicile certificate",
        ]
        lower_combined = combined.lower()
        for dt in doc_types:
            if dt in lower_combined:
                key = f"doctype:{dt}"
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "type": "DocType",
                        "id": dt.replace(" ", "_"),
                        "props": {"name": dt.title()},
                    })

        return entities

    def _link_similar_documents(
        self, doc_node: str, entities: list[dict],
    ) -> None:
        entity_ids = {f"{e['type']}:{e['id']}" for e in entities}
        for node in list(self._graph.nodes):
            if node == doc_node or not node.startswith("doc:"):
                continue
            shared = set(self._graph.successors(node)) & entity_ids
            if shared:
                self._graph.add_edge(
                    doc_node, node,
                    relation="SIMILAR_TO",
                    shared_entities=len(shared),
                )

    def query(self, query_type: str, **kwargs: Any) -> list[dict]:
        """Public API for ad-hoc graph queries."""
        if query_type == "by_entity":
            return self._query_by_entity(kwargs["entity_type"], kwargs["entity_id"])
        if query_type == "by_person":
            return self._query_by_person(kwargs["name"])
        if query_type == "neighbors":
            return self._query_neighbors(kwargs["node_id"], kwargs.get("depth", 1))
        return []

    def _query_by_entity(self, entity_type: str, entity_id: str) -> list[dict]:
        target = f"{entity_type}:{entity_id}"
        if target not in self._graph:
            return []
        docs = [
            n for n in self._graph.predecessors(target)
            if n.startswith("doc:")
        ]
        return [dict(self._graph.nodes[d]) for d in docs]

    def _query_by_person(self, name: str) -> list[dict]:
        results = []
        for node, data in self._graph.nodes(data=True):
            if data.get("label") == "Person" and name.lower() in data.get("name", "").lower():
                docs = [
                    dict(self._graph.nodes[n])
                    for n in self._graph.predecessors(node)
                    if n.startswith("doc:")
                ]
                results.extend(docs)
        return results

    def _query_neighbors(self, node_id: str, depth: int = 1) -> list[dict]:
        if node_id not in self._graph:
            return []
        visited: set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for neighbor in list(self._graph.successors(n)) + list(self._graph.predecessors(n)):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited |= frontier
            frontier = next_frontier
        visited |= frontier
        visited.discard(node_id)
        return [{"id": n, **dict(self._graph.nodes[n])} for n in visited]
