"""Per-case R007 identity shared by frozen evaluation and analysis."""

import hashlib
import json

CASE_IDENTITY_PREFIX = "r007-case-sha256:"


def case_identity_tag(row, provenance):
    """Bind a report case to its exact exported row and source/group identity."""
    if not provenance.get("source_id") or not provenance.get("group_id"):
        raise ValueError("R007 case identity requires source_id and group_id")
    identity = {"provenance": provenance, "row": row}
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return CASE_IDENTITY_PREFIX + hashlib.sha256(canonical).hexdigest()


def attach_case_identity_tags(dataset, rows, provenance):
    """Carry frozen row identities through Laya's per-case `tags` report field."""
    if len(dataset.examples) != len(rows) or len(rows) != len(provenance):
        raise ValueError("R007 dataset/provenance length mismatch")
    for example, row, metadata in zip(dataset.examples, rows, provenance):
        if any(tag.startswith(CASE_IDENTITY_PREFIX) for tag in example.tags):
            raise ValueError("R007 identity tag already present")
        example.tags = (*example.tags, case_identity_tag(row, metadata))
