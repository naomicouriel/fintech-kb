# fintech-kb

A minimal proof of concept for a self-expanding knowledge base over Argentine
fintech regulation (BCRA). Starting from a hand-curated seed of 5 entities and
4 relations, the script ingests one BCRA Comunicación, expands the KB to 36
entities and 24 relations, and demonstrates that the post-ingest KB can answer
a question the seed KB cannot.

## The eval

A single before/after question is the test:

> ¿Qué es un PSPCP como Servicio y qué requisitos particulares debe cumplir?

**Pre-ingest** the model refuses correctly:
*"La base de conocimiento no contiene información sobre 'PSPCP como Servicio'."*

**Post-ingest** the model answers with the definition (from §1.4.1.1 of
Comunicación A 8432), the structural relations (`is_subtype_of pspcp`,
`offers cuenta_de_pago`, `may_offer billetera_digital_interoperable`), and the
specific compliance obligation, citing entity IDs and `[com_a_8432]` provenance
throughout.

The pre/post diff is the demonstration of self-expansion. Run it with
`python main.py demo`.

## Quick start

Requires Python 3.11+ and an Anthropic API key.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

python main.py demo
```

The `demo` command loads the seed KB from git HEAD, runs Q1 + Q2 against it,
then runs Q1 + Q2 against the current `kb.json` (post-ingest) and prints both
blocks. ~4 query calls, ~$0.30 at list pricing for `claude-opus-4-7`.

To re-derive `kb.json` from scratch:

```
git show e8d54dd:kb.json > kb.json    # reset to seed
python main.py merge --force-reextract # re-extract + resolve + merge
```

Without `--force-reextract`, the merge step reuses `cache/extraction.json` —
see [Reproducibility](#reproducibility).

## Repo layout

```
main.py                  single-file implementation
kb.json                  the knowledge base (post-ingest state committed)
data/comunicacion.pdf    source document: BCRA Comunicación "A" 8432
cache/extraction.json    deterministic LLM extraction output (committed)
requirements.txt         pinned: anthropic, pypdf, pydantic, python-dotenv
```

## How it works

Five functions, one CLI, no frameworks.

**`extract_pdf_text`** — pypdf, with regex post-processing for two artifacts:
hyphenated word-wrap (`tomador` → `tomador`, not `toma-dor`) and page-number
markers between sections.

**`extract_from_pdf`** — single Claude call (`claude-opus-4-7`), strict JSON
output validated against a Pydantic schema. The prompt forbids fabricating
aliases not present in the document and forbids extracting entities that
appear only in the addressee list at the top of the Comunicación.

**`resolve_entities`** — two-stage:

- *Stage 1, deterministic.* Normalize each candidate (lowercase, NFD/strip
  combining marks, drop leading articles, collapse multi-spaces, glue
  split-acronym artifacts like `PS PCP` → `PSPCP`), then exact-match against
  every existing entity's normalized name and aliases. All 5 seed entities
  resolve here without an LLM call.
- *Stage 2, LLM judge.* For unresolved candidates only, Claude sees the full
  KB and decides per-candidate: `NEW` | `MERGE` | `UNCERTAIN` with confidence
  and reasoning. The prompt's load-bearing instruction is verbatim:
  *"A new acronym, a new defined term, or a new regulatory category is a NEW
  entity even if it shares a stem with an existing one."* That rule is what
  keeps `PSPCP como Servicio` (a distinct subcategory introduced in this
  Comunicación) from being silently merged into the existing `pspcp` entity.

**`diff_and_merge`** — NEW entities get a slugified ID; MERGE with confidence
≥ 0.85 auto-applies; everything else lands in `pending_review` for human eyes.
Relations resolve subject/object against entities **and** regulations — a
regulation in `regulations[]` is a valid relation target.

**`validate_kb`** — runs before save. Unique IDs, foreign-key integrity on
relations, valid `source` (either `"seed"` or a known regulation ID), no
empty fields. On failure the proposed state is dumped to `kb_proposed.json`
and `kb.json` is left untouched.

**`query`** — passes the full KB JSON + question to Claude with a grounding
prompt: cite entity IDs in brackets, cite regulation IDs for provenance,
refuse explicitly when the KB lacks the information rather than fabricating.

## Schema

```
Entity     (id, type, name, aliases[], description, source)
Relation   (subject, predicate, object, source)
Regulation (id, number, issuer, date, summary)
KB         (entities[], relations[], regulations[], pending_review[])
```

`type` is one of `regulator | entity_class | registry | regulation | concept`.
`source` is `"seed"` or a regulation `id` — every fact in the KB is
traceable to the document that introduced it.

## Reproducibility

The Opus 4.7 endpoint does not accept the `temperature` parameter, so the
extraction call is non-deterministic at the API level. To keep the demo
reproducible across clones, the canonical extraction is cached to
`cache/extraction.json` and committed to the repo. The merge step reuses
this cache by default. Pass `--force-reextract` to call the API again.

The query step is also non-deterministic, but the answers are short and the
correctness criterion (Q2 refuses pre-ingest, answers post-ingest with cited
provenance) is robust to phrasing variation across runs.

## Known limitations

- **One residual phantom alias.** `Ley 19.550` appears as an alias of `Ley
  General de Sociedades 19.550`. The number is in the document but the
  truncated form is not. Defensible contraction, not a fabricated acronym.
- **One hallucinated relation was deleted manually.** The first extraction
  produced `registro_nacional_de_reincidencia issues
  texto_ordenado_sobre_proveedores_de_servicios_de_pago`, which is wrong —
  the Registro de Reincidencia issues criminal-record certificates, not BCRA
  ordered texts. Removed by hand from `kb.json` rather than by re-extracting.
- **One inline acronym leak in the Q2 answer.** The post-ingest answer mentions
  "PLA/FT", the standard Spanish acronym for the AML/CFT regime, which is
  **not** in the KB (deliberately stripped during extraction). The full
  regulation name and entity ID it cites are correctly grounded; only the
  inline acronym leaked from the model's prior knowledge.
- **Addressee-filter rule didn't hold across re-extractions.** A first
  extraction filtered out the cover-line addressees of the Comunicación; a
  second extraction at unchanged-prompt brought them back. Reproducibility is
  enforced via the committed cache rather than further prompt hardening.

## What production would need

- `supersedes` / `superseded_by` relations to handle versioned amendments.
  Point 8 of Comunicación A 8432 extends a deadline from 6 to 12 months;
  here it lands as a new fact rather than a versioned amendment of the
  prior rule.
- Multi-document batch ingest with cross-run dedup.
- Prompt caching on the query stage. Deferred for time; would substantially
  reduce per-query latency and cost on repeated queries against the same KB.
- A contradiction-handling story when two regulations conflict.
- `effective_date` and `date_published` on every fact, not just regulations.
- Stricter grounding constraints on the query model. The `PLA/FT` leak above
  is the specific failure mode to design against.

## Scope

1–2 hour single-session PoC, single document, single eval. Total API spend
during the build was ~$2.10.
