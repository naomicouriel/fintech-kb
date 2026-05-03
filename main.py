"""Self-expanding fintech knowledge base — minimal PoC.

Build order is checkpoint-driven (see plan):
  1. Pydantic models + load_kb + extract_pdf_text   <-- this file, current state
  2. extract_from_pdf
  3. resolve_entities
  4. diff_and_merge + validate_kb + save_kb
  5. query + demo flow
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader

MODEL = "claude-opus-4-7"
EXTRACTION_CACHE = Path("cache/extraction.json")


# ---------- Schema ----------

EntityType = Literal["regulator", "entity_class", "registry", "regulation", "concept"]


class Entity(BaseModel):
    id: str
    type: EntityType
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    source: str = "seed"


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    source: str = "seed"


class Regulation(BaseModel):
    id: str
    number: str
    issuer: str
    date: str
    summary: str


class ReviewItem(BaseModel):
    kind: Literal["entity", "relation"]
    payload: dict
    decision: Literal["MERGE", "UNCERTAIN"]
    merge_target_id: str | None
    confidence: float
    reasoning: str


class KB(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    regulations: list[Regulation] = Field(default_factory=list)
    pending_review: list[ReviewItem] = Field(default_factory=list)


class CandidateEntity(BaseModel):
    type: EntityType
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class CandidateRelation(BaseModel):
    subject_name: str
    predicate: str
    object_name: str


class ExtractionResult(BaseModel):
    regulation: Regulation
    candidate_entities: list[CandidateEntity]
    candidate_relations: list[CandidateRelation]


class ResolutionDecision(BaseModel):
    candidate_name: str
    decision: Literal["NEW", "MERGE", "UNCERTAIN"]
    merge_target_id: str | None = None
    confidence: float
    reasoning: str


# ---------- Loader ----------


def _kb_from_raw(raw: dict) -> KB:
    for e in raw.get("entities", []):
        e.setdefault("source", "seed")
    for r in raw.get("relations", []):
        r.setdefault("source", "seed")
    raw.setdefault("regulations", [])
    raw.setdefault("pending_review", [])
    return KB.model_validate(raw)


def load_kb(path: str | Path = "kb.json") -> KB:
    return _kb_from_raw(json.loads(Path(path).read_text(encoding="utf-8")))


def load_seed_kb_from_git(rev: str = "HEAD", path: str = "kb.json") -> KB:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        capture_output=True, text=True, check=True,
    )
    return _kb_from_raw(json.loads(out.stdout))


# ---------- PDF extraction ----------


def _strip_page_markers(text: str) -> str:
    return re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)


def _dehyphenate(text: str) -> str:
    return re.sub(r"-\n[ \t]*", "", text)


def extract_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    text = _strip_page_markers(text)
    text = _dehyphenate(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- LLM extraction ----------


EXTRACTION_PROMPT = """You are extracting structured information from a BCRA (Banco Central de la República Argentina) regulatory communication.

Goal: extract entity classes, regulators, registries, and concepts that the document defines or references, plus the factual relationships between them.

Output JSON schema:
{schema}

Rules:
- `regulation.id` MUST be exactly "{regulation_id}". `number` is the BCRA reference (e.g. "A 8432"). `issuer` is "bcra". `date` is ISO YYYY-MM-DD. `summary` is 1-2 sentences in English describing the substantive change.
- `candidate_entities`: include every distinct entity (regulator, entity_class, registry, regulation, concept) that the document defines or substantively references. Provide `type`, `name` (the official Spanish name), `aliases` (acronyms or alternate forms — strip stray spaces inside acronyms, e.g. "PS PCP" -> "PSPCP"), and `description` if the document defines one.
- `candidate_relations`: factual relationships between entities. Use snake_case predicates. `subject_name` and `object_name` MUST match `name` values from `candidate_entities` exactly. IDs are assigned later — do not invent them.

CRITICAL rules — follow strictly:

1. Only extract entities that the document DEFINES or SUBSTANTIVELY REGULATES. Do NOT extract entities that appear only in the addressee list at the top of the Comunicación (the "A LAS REDES DE..." / "A LOS PROVEEDORES DE..." cover lines). Mere mention as a recipient is not enough — there must be substantive regulatory content about the entity in the body of the document.

2. Do NOT include aliases or acronyms unless they appear LITERALLY in the document text. If you are tempted to add a commonly-known acronym from prior knowledge (e.g. PEP, PLA/FT, PSI, CSNU), do NOT. Provenance must be the source document, not your training data. Verify by mental search: if the exact letters do not appear in the document text, the alias does not go in.

3. Do NOT assert relations the document does not state. In particular: the UN Security Council Committee designates terrorism-related persons/entities, NOT politically exposed persons (PEPs). Those are separate AML concepts. Do not invent cross-concept relations.

4. A new acronym, a new defined term, or a new regulatory category is a NEW entity even if it shares a stem with an existing one. Treat "PSPCP como Servicio" as distinct from "PSPCP".

Output ONLY a JSON object matching the schema. No prose, no markdown fences, no commentary before or after.

Document text:
---
{text}
---"""


def _anthropic_client() -> Anthropic:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)
    return Anthropic(api_key=key)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def extract_from_pdf(text: str, regulation_id: str, client: Anthropic) -> ExtractionResult:
    schema = ExtractionResult.model_json_schema()
    prompt = EXTRACTION_PROMPT.format(
        schema=json.dumps(schema, indent=2),
        regulation_id=regulation_id,
        text=text,
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    u = resp.usage
    cache_w = getattr(u, "cache_creation_input_tokens", 0) or 0
    cache_r = getattr(u, "cache_read_input_tokens", 0) or 0
    print(
        f"[tokens] extract model={MODEL} input={u.input_tokens} output={u.output_tokens} "
        f"cache_write={cache_w} cache_read={cache_r}",
        file=sys.stderr,
    )
    raw = _strip_code_fences(resp.content[0].text)
    return ExtractionResult.model_validate_json(raw)


def cached_extract(
    text: str, regulation_id: str, client: Anthropic, force: bool = False
) -> ExtractionResult:
    if not force and EXTRACTION_CACHE.exists():
        print(f"[cache] hit: {EXTRACTION_CACHE}", file=sys.stderr)
        return ExtractionResult.model_validate_json(EXTRACTION_CACHE.read_text(encoding="utf-8"))
    print(f"[cache] miss: extracting (force={force})", file=sys.stderr)
    result = extract_from_pdf(text, regulation_id, client)
    EXTRACTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTION_CACHE.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


# ---------- Stage 1: deterministic normalization ----------


def _join_split_acronyms(s: str) -> str:
    """Glue runs of all-caps chunks separated by spaces when joined length is 2-6.
    Targets pypdf stray-space-in-acronym artifact, e.g. 'PS PCP' -> 'PSPCP'."""
    def repl(m: re.Match[str]) -> str:
        joined = re.sub(r"\s+", "", m.group(0))
        return joined if 2 <= len(joined) <= 6 else m.group(0)
    return re.sub(r"\b[A-Z]+(?:\s+[A-Z]+)+\b", repl, s)


def normalize_name(s: str) -> str:
    s = _join_split_acronyms(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"^(el|la|los|las)\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_forms(name: str, aliases: list[str]) -> set[str]:
    return {normalize_name(name), *(normalize_name(a) for a in aliases)}


def stage1_resolve(
    candidates: list[CandidateEntity], kb: KB
) -> tuple[dict[int, str], list[tuple[int, CandidateEntity]]]:
    """Returns (resolved: idx->existing_id, unresolved: [(idx, candidate), ...])."""
    existing: dict[str, str] = {}
    for e in kb.entities:
        for form in _norm_forms(e.name, e.aliases):
            existing[form] = e.id

    resolved: dict[int, str] = {}
    unresolved: list[tuple[int, CandidateEntity]] = []
    for i, c in enumerate(candidates):
        hit = next((existing[f] for f in _norm_forms(c.name, c.aliases) if f in existing), None)
        if hit:
            resolved[i] = hit
        else:
            unresolved.append((i, c))
    return resolved, unresolved


# ---------- Stage 2: LLM judge ----------


RESOLUTION_PROMPT = """You are deciding entity-resolution for a self-expanding regulatory knowledge base.

Existing knowledge base entities (the only valid merge targets):
{kb_entities}

The candidates below were extracted from a new document. They did NOT exact-match any existing entity by name or alias. For each candidate, decide:
- "NEW": a genuinely new entity. It will be added to the KB.
- "MERGE": the same real-world entity as an existing one, just under a different name or paraphrase. Provide `merge_target_id`.
- "UNCERTAIN": cannot tell from the information given.

CRITICAL: A new acronym, a new defined term, or a new regulatory category is a NEW entity even if it shares a stem with an existing one. Only MERGE when the candidate is the same real-world thing under a different name. For example, "PSPCP como Servicio" is a NEW entity_class, NOT a merge into "pspcp" — it is a distinct regulatory category defined for the first time in the source document.

Output a JSON array of decisions, one per candidate, in input order. Per item:
{{
  "candidate_name": <string, copy from input>,
  "decision": "NEW" | "MERGE" | "UNCERTAIN",
  "merge_target_id": <existing entity id, or null>,
  "confidence": <float in [0,1]>,
  "reasoning": <one sentence>
}}

Candidates:
{candidates}

Output ONLY the JSON array. No prose, no markdown fences."""


def stage2_resolve(
    unresolved: list[CandidateEntity], kb: KB, client: Anthropic
) -> list[ResolutionDecision]:
    if not unresolved:
        return []
    kb_view = [
        {"id": e.id, "type": e.type, "name": e.name,
         "aliases": e.aliases, "description": e.description}
        for e in kb.entities
    ]
    cand_view = [
        {"name": c.name, "type": c.type,
         "aliases": c.aliases, "description": c.description}
        for c in unresolved
    ]
    prompt = RESOLUTION_PROMPT.format(
        kb_entities=json.dumps(kb_view, indent=2, ensure_ascii=False),
        candidates=json.dumps(cand_view, indent=2, ensure_ascii=False),
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    u = resp.usage
    print(
        f"[tokens] resolve model={MODEL} input={u.input_tokens} output={u.output_tokens}",
        file=sys.stderr,
    )
    raw = _strip_code_fences(resp.content[0].text)
    data = json.loads(raw)
    return [ResolutionDecision.model_validate(d) for d in data]


# ---------- Query ----------


QUERY_PROMPT = """You answer questions about a regulatory knowledge base for Argentine fintech (BCRA).

Knowledge base (JSON):
{kb}

Question: {question}

Instructions:
- Answer in Spanish if the question is in Spanish; otherwise in English.
- Ground every claim in the KB. Cite entity IDs in brackets like [bcra], [pspcp_como_servicio]. Cite regulation IDs like [com_a_8432] when the source applies.
- If the KB does not contain enough information to answer, say so explicitly: "La base de conocimiento no contiene información sobre X." Do not fabricate facts beyond the KB.
- Be concise: 1-3 sentences for simple questions. Use a short list only if the question genuinely requires enumerating multiple items.
"""


def query(question: str, kb: KB, client: Anthropic) -> str:
    kb_json = json.dumps(kb.model_dump(), indent=2, ensure_ascii=False)
    prompt = QUERY_PROMPT.format(kb=kb_json, question=question)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    u = resp.usage
    print(
        f"[tokens] query input={u.input_tokens} output={u.output_tokens}",
        file=sys.stderr,
    )
    return resp.content[0].text.strip()


# ---------- Merge / validate / save ----------


def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFD", name)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "entity"


def _unique_id(slug: str, taken: set[str]) -> str:
    if slug not in taken:
        return slug
    i = 2
    while f"{slug}_{i}" in taken:
        i += 1
    return f"{slug}_{i}"


def diff_and_merge(
    extraction: ExtractionResult,
    stage1_resolved: dict[int, str],
    stage2_decisions: list[ResolutionDecision],
    unresolved_indices: list[int],
    kb: KB,
    auto_merge_threshold: float = 0.85,
) -> KB:
    new_kb = KB.model_validate(kb.model_dump())

    if not any(r.id == extraction.regulation.id for r in new_kb.regulations):
        new_kb.regulations.append(extraction.regulation)

    # Build name->id lookup, populated incrementally so later relations can resolve
    # against entities created in this same merge.
    name_to_id: dict[str, str] = {}
    for e in new_kb.entities:
        for f in _norm_forms(e.name, e.aliases):
            name_to_id[f] = e.id

    for idx, eid in stage1_resolved.items():
        c = extraction.candidate_entities[idx]
        for f in _norm_forms(c.name, c.aliases):
            name_to_id[f] = eid

    taken_ids = {e.id for e in new_kb.entities}
    decision_by_idx = dict(zip(unresolved_indices, stage2_decisions))

    for idx, d in decision_by_idx.items():
        c = extraction.candidate_entities[idx]
        if d.decision == "NEW":
            new_id = _unique_id(_slugify(c.name), taken_ids)
            taken_ids.add(new_id)
            new_kb.entities.append(Entity(
                id=new_id, type=c.type, name=c.name,
                aliases=c.aliases, description=c.description,
                source=extraction.regulation.id,
            ))
            for f in _norm_forms(c.name, c.aliases):
                name_to_id[f] = new_id
        elif (
            d.decision == "MERGE"
            and d.merge_target_id
            and d.confidence >= auto_merge_threshold
        ):
            target = next((e for e in new_kb.entities if e.id == d.merge_target_id), None)
            if target is None:
                new_kb.pending_review.append(ReviewItem(
                    kind="entity", payload=c.model_dump(),
                    decision="UNCERTAIN", merge_target_id=d.merge_target_id,
                    confidence=d.confidence,
                    reasoning=f"merge target {d.merge_target_id!r} not found in KB",
                ))
                continue
            for label in [c.name, *c.aliases]:
                if label and label != target.name and label not in target.aliases:
                    target.aliases.append(label)
            for f in _norm_forms(c.name, c.aliases):
                name_to_id[f] = d.merge_target_id
        else:
            new_kb.pending_review.append(ReviewItem(
                kind="entity", payload=c.model_dump(),
                decision="MERGE" if d.decision == "MERGE" else "UNCERTAIN",
                merge_target_id=d.merge_target_id, confidence=d.confidence,
                reasoning=d.reasoning,
            ))

    # Regulations are valid relation targets too — index by id and number.
    reg_lookup: dict[str, str] = {}
    for r in new_kb.regulations:
        reg_lookup[normalize_name(r.id)] = r.id
        reg_lookup[normalize_name(r.number)] = r.id

    def resolve_ref(name: str) -> str | None:
        n = normalize_name(name)
        return name_to_id.get(n) or reg_lookup.get(n)

    existing_rel_keys = {(r.subject, r.predicate, r.object) for r in new_kb.relations}
    for cr in extraction.candidate_relations:
        sid = resolve_ref(cr.subject_name)
        oid = resolve_ref(cr.object_name)
        if sid is None or oid is None:
            new_kb.pending_review.append(ReviewItem(
                kind="relation",
                payload={
                    "subject_name": cr.subject_name,
                    "predicate": cr.predicate,
                    "object_name": cr.object_name,
                    "unresolved_subject": sid is None,
                    "unresolved_object": oid is None,
                },
                decision="UNCERTAIN", merge_target_id=None, confidence=0.0,
                reasoning="could not resolve to entity or regulation",
            ))
            continue
        key = (sid, cr.predicate, oid)
        if key in existing_rel_keys:
            continue
        existing_rel_keys.add(key)
        new_kb.relations.append(Relation(
            subject=sid, predicate=cr.predicate, object=oid,
            source=extraction.regulation.id,
        ))

    return new_kb


def validate_kb(kb: KB) -> list[str]:
    errors: list[str] = []
    ids = [e.id for e in kb.entities]
    seen: set[str] = set()
    for i in ids:
        if not i:
            errors.append("Entity has empty id")
        if i in seen:
            errors.append(f"Duplicate entity id: {i!r}")
        seen.add(i)

    valid_targets = set(ids) | {r.id for r in kb.regulations}
    valid_sources = {"seed"} | {r.id for r in kb.regulations}

    for e in kb.entities:
        if not e.name:
            errors.append(f"Entity {e.id!r} has empty name")
        if e.source not in valid_sources:
            errors.append(f"Entity {e.id!r} has invalid source {e.source!r}")

    for r in kb.relations:
        if r.subject not in valid_targets:
            errors.append(f"Relation subject {r.subject!r} not in entities or regulations")
        if r.object not in valid_targets:
            errors.append(f"Relation object {r.object!r} not in entities or regulations")
        if r.source not in valid_sources:
            errors.append(f"Relation has invalid source {r.source!r}")

    return errors


def save_kb(kb: KB, path: str | Path = "kb.json") -> None:
    Path(path).write_text(
        json.dumps(kb.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------- CLI ----------


def main() -> None:
    raw_args = sys.argv[1:]
    force_reextract = "--force-reextract" in raw_args
    args = [a for a in raw_args if a != "--force-reextract"]
    if not args:
        print("usage: main.py <load|extract|extract_llm|resolve|merge> [--force-reextract]",
              file=sys.stderr)
        sys.exit(2)

    cmd = args[0]
    if cmd == "load":
        kb = load_kb("kb.json")
        print(f"Loaded KB: {len(kb.entities)} entities, "
              f"{len(kb.relations)} relations, "
              f"{len(kb.regulations)} regulations.")
        for e in kb.entities:
            print(f"  - {e.id} ({e.type}): {e.name} [source={e.source}]")
    elif cmd == "extract":
        text = extract_pdf_text("data/comunicacion.pdf")
        print(text)
    elif cmd == "extract_llm":
        text = extract_pdf_text("data/comunicacion.pdf")
        client = _anthropic_client()
        result = cached_extract(text, "com_a_8432", client, force=force_reextract)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    elif cmd == "resolve":
        _run_resolve(force_reextract=force_reextract)
    elif cmd == "merge":
        _run_merge(force_reextract=force_reextract)
    elif cmd == "demo":
        _run_demo()
    elif cmd == "query":
        if len(args) < 2:
            print("usage: main.py query \"<question>\"", file=sys.stderr)
            sys.exit(2)
        kb = load_kb("kb.json")
        client = _anthropic_client()
        print(query(" ".join(args[1:]), kb, client))
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)


def _run_resolve(force_reextract: bool = False) -> None:
    kb = load_kb("kb.json")
    text = extract_pdf_text("data/comunicacion.pdf")
    client = _anthropic_client()
    extraction = cached_extract(text, "com_a_8432", client, force=force_reextract)

    print("=== Stage 1 (deterministic) ===")
    resolved, unresolved = stage1_resolve(extraction.candidate_entities, kb)
    print(f"RESOLVED ({len(resolved)}):")
    for i, eid in sorted(resolved.items()):
        c = extraction.candidate_entities[i]
        print(f"  - {c.name!r}  ->  {eid}")
    print(f"UNRESOLVED -> Stage 2 ({len(unresolved)}):")
    for _, c in unresolved:
        print(f"  - {c.name!r}")

    print()
    print("=== Stage 2 (LLM judge) ===")
    candidates_only = [c for _, c in unresolved]
    decisions = stage2_resolve(candidates_only, kb, client)
    for d in decisions:
        line = f"  - {d.candidate_name!r}: {d.decision} (conf={d.confidence:.2f})"
        if d.merge_target_id:
            line += f" -> {d.merge_target_id}"
        line += f"  | {d.reasoning}"
        print(line)

    print()
    print("=== EVAL TARGET CHECK: PSPCP como Servicio ===")
    target = "PSPCP como Servicio"
    stage1_hit = None
    for i, eid in resolved.items():
        if extraction.candidate_entities[i].name.strip() == target:
            stage1_hit = eid
            break
    stage2_hit = next((d for d in decisions if d.candidate_name.strip() == target), None)

    if stage1_hit is not None:
        print(f"  Stage 1 resolved to {stage1_hit!r}  [FAIL — should be unresolved at stage 1]")
        sys.exit(1)
    if stage2_hit is None:
        print("  Not present in candidates at all  [FAIL — extraction regression]")
        sys.exit(1)
    if stage2_hit.decision == "NEW":
        print(f"  Stage 2: NEW (conf={stage2_hit.confidence:.2f})  [PASS]")
    elif stage2_hit.decision == "MERGE":
        print(f"  Stage 2: MERGE -> {stage2_hit.merge_target_id} (conf={stage2_hit.confidence:.2f})  [FAIL — STOP, must be NEW]")
        print(f"  reasoning: {stage2_hit.reasoning}")
        sys.exit(1)
    else:
        print(f"  Stage 2: UNCERTAIN (conf={stage2_hit.confidence:.2f})  [SOFT FAIL — review needed]")
        print(f"  reasoning: {stage2_hit.reasoning}")


def _run_merge(force_reextract: bool = False) -> None:
    kb = load_kb("kb.json")
    text = extract_pdf_text("data/comunicacion.pdf")
    client = _anthropic_client()
    extraction = cached_extract(text, "com_a_8432", client, force=force_reextract)

    resolved, unresolved = stage1_resolve(extraction.candidate_entities, kb)
    candidates_only = [c for _, c in unresolved]
    unresolved_indices = [i for i, _ in unresolved]
    decisions = stage2_resolve(candidates_only, kb, client)

    new_kb = diff_and_merge(
        extraction=extraction,
        stage1_resolved=resolved,
        stage2_decisions=decisions,
        unresolved_indices=unresolved_indices,
        kb=kb,
    )

    print("=== Merge summary ===")
    print(f"  entities:    {len(kb.entities)} -> {len(new_kb.entities)} "
          f"(+{len(new_kb.entities) - len(kb.entities)})")
    print(f"  relations:   {len(kb.relations)} -> {len(new_kb.relations)} "
          f"(+{len(new_kb.relations) - len(kb.relations)})")
    print(f"  regulations: {len(kb.regulations)} -> {len(new_kb.regulations)} "
          f"(+{len(new_kb.regulations) - len(kb.regulations)})")
    print(f"  pending_review: {len(new_kb.pending_review)}")
    if new_kb.pending_review:
        for item in new_kb.pending_review:
            print(f"    - [{item.kind}] {item.decision} conf={item.confidence:.2f} :: {item.reasoning}")

    print()
    print("=== Validation ===")
    errors = validate_kb(new_kb)
    if errors:
        print(f"  FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"    - {e}")
        proposed = "kb_proposed.json"
        save_kb(new_kb, proposed)
        print(f"  Proposed state dumped to {proposed}; kb.json NOT modified.")
        sys.exit(1)
    print("  OK")

    save_kb(new_kb, "kb.json")
    print()
    print("=== Saved kb.json ===")


Q1 = "¿Quién regula a los Proveedores de Servicios de Pago en Argentina?"
Q2 = "¿Qué es un PSPCP como Servicio y qué requisitos particulares debe cumplir?"


def _run_demo() -> None:
    client = _anthropic_client()
    seed_kb = load_seed_kb_from_git()
    current_kb = load_kb("kb.json")

    print("=" * 78)
    print(f"PRE-INGEST  (seed KB: {len(seed_kb.entities)} entities, "
          f"{len(seed_kb.relations)} relations, "
          f"{len(seed_kb.regulations)} regulations)")
    print("=" * 78)
    print(f"\nQ1: {Q1}\n")
    print(query(Q1, seed_kb, client))
    print(f"\nQ2: {Q2}\n")
    print(query(Q2, seed_kb, client))

    print()
    print("=" * 78)
    print(f"POST-INGEST (after Comunicación A 8432: {len(current_kb.entities)} entities, "
          f"{len(current_kb.relations)} relations, "
          f"{len(current_kb.regulations)} regulations)")
    print("=" * 78)
    print(f"\nQ1: {Q1}\n")
    print(query(Q1, current_kb, client))
    print(f"\nQ2: {Q2}\n")
    print(query(Q2, current_kb, client))


if __name__ == "__main__":
    main()
