#!/usr/bin/env python3
"""Unified primer validation and evaluation pipeline.

This module combines the local TmBLAST search, taxonomic summarization, and
tree-map rendering steps into one importable API and CLI.  The Streamlit app
uses the same functions, so command-line and web runs share the same behavior.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
TM_ALIGN_DIR = ROOT_DIR / "Tm_align"
if str(TM_ALIGN_DIR) not in sys.path:
    sys.path.insert(0, str(TM_ALIGN_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import TmAlign as tmalign  # noqa: E402
import TmBLAST as tmblast  # noqa: E402
import analyze_tm as tax_analyzer  # noqa: E402
import tree_map as tree_mapper  # noqa: E402


HIT_COLUMNS = [
    "query_id",
    "expanded_query",
    "Tm",
    "contig",
    "start",
    "end",
    "strand",
    "identity",
    "q_align",
    "db_align",
    "amplicon_length",
]
RANK_CHOICES = ("class", "order", "family", "genus")
KINGDOM_FLAGS = ("a", "p", "f", "r", "m")
DEFAULT_FASTA = TM_ALIGN_DIR / "wf_mt_tax.fasta"
DEFAULT_TAXDUMP = ROOT_DIR / "taxdump"


@dataclass
class PipelineSettings:
    fasta_path: Path = DEFAULT_FASTA
    taxdump_dir: Path = DEFAULT_TAXDUMP
    forward_primer: str = ""
    reverse_primer: str = ""
    k: int = 4
    min_tm: float = 30.0
    min_identity: float = 50.0
    cpus: int = 1
    max_expansions: int = 100_000
    max_records: Optional[int] = None
    rank: str = "family"
    depth: Optional[str] = "family"
    kingdom_filters: List[str] = field(default_factory=list)
    taxon_names: List[str] = field(default_factory=list)
    taxon_depths: List[str] = field(default_factory=list)
    output_dir: Optional[Path] = None
    render_tree: bool = True
    include_taxdump_index: bool = False
    acc2taxid_dir: Optional[Path] = None
    acc_cache: Optional[Path] = None
    acc_workers: int = 1


@dataclass
class PipelineResult:
    hits: pd.DataFrame
    best_hits: pd.DataFrame
    accession_metrics: pd.DataFrame
    taxid_summary: pd.DataFrame
    rank_summary: pd.DataFrame
    tree_path: Optional[Path]
    output_paths: Dict[str, Path]
    warnings: List[str]


@dataclass
class FastaTaxidInfo:
    record_to_taxid: Dict[str, str]
    accession_to_taxid: Dict[str, str]
    accessions: List[str]
    origin_taxids: List[str]


def normalize_record_limit(max_records: Optional[int]) -> Optional[int]:
    if max_records is None:
        return None
    try:
        max_records_int = int(max_records)
    except (TypeError, ValueError):
        return None
    return max_records_int if max_records_int > 0 else None


def open_text_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def discover_fasta_databases(root: Path = ROOT_DIR) -> List[Path]:
    paths: List[Path] = []
    for pattern in ("*.fa", "*.fasta", "*.fna", "*.ffn"):
        paths.extend(root.glob(pattern))
        paths.extend((root / "Tm_align").glob(pattern))
    clean = []
    for path in paths:
        if ".ipynb_checkpoints" in path.parts:
            continue
        if "primer" in path.name.lower():
            continue
        if path.is_file() and path not in clean:
            clean.append(path)
    clean.sort(key=lambda p: (0 if "tax" in p.name.lower() else 1, str(p)))
    return clean


def clean_primer_sequence(seq: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", seq or "").upper().replace("U", "T")
    if not cleaned:
        raise ValueError("Primer sequence is empty.")
    invalid = sorted({ch for ch in cleaned if ch not in tmblast.IUPAC_MAP})
    if invalid:
        raise ValueError(f"Unsupported base(s) in primer: {', '.join(invalid)}")
    return cleaned


def build_primer_records(forward: str, reverse: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    if forward and forward.strip():
        records.append(("q1", clean_primer_sequence(forward)))
    if reverse and reverse.strip():
        records.append(("q2", clean_primer_sequence(reverse)))
    if not records:
        raise ValueError("At least one primer sequence is required.")
    return records


def calculate_amplicon_length(start1: object, end1: object, start2: object, end2: object) -> int:
    positions = [int(start1), int(end1), int(start2), int(end2)]
    return max(positions) - min(positions) + 1


def attach_amplicon_lengths_to_hits(df: pd.DataFrame, primer_pair: Sequence[str] = ("q1", "q2")) -> pd.DataFrame:
    """Attach inclusive PCR product length to hit rows.

    For each contig, the best q1 and q2 rows are selected by Tm and identity.
    Their inclusive outer span is stored as ``amplicon_length`` for all rows on
    that contig.  Rows lacking a complete primer pair keep a missing value.
    """
    out = df.copy()
    if out.empty:
        out["amplicon_length"] = pd.Series(dtype="Int64")
        return out
    if len(primer_pair) < 2:
        out["amplicon_length"] = pd.NA
        return out

    pair = list(primer_pair[:2])
    work = out[out["query_id"].isin(pair)].copy()
    if work.empty:
        out["amplicon_length"] = pd.NA
        return out

    work["_pair_order"] = work["query_id"].map({pair[0]: 0, pair[1]: 1})
    work = work.sort_values(
        ["contig", "query_id", "Tm", "identity"],
        ascending=[True, True, False, False],
    )
    best = work.drop_duplicates(subset=["contig", "query_id"], keep="first")
    lengths: Dict[str, int] = {}
    for contig, group in best.groupby("contig", dropna=False):
        if set(group["query_id"]) >= set(pair):
            q1 = group[group["query_id"] == pair[0]].iloc[0]
            q2 = group[group["query_id"] == pair[1]].iloc[0]
            lengths[str(contig)] = calculate_amplicon_length(q1["start"], q1["end"], q2["start"], q2["end"])

    out["amplicon_length"] = out["contig"].astype(str).map(lengths)
    out["amplicon_length"] = pd.to_numeric(out["amplicon_length"], errors="coerce").astype("Int64")
    return out


def read_fasta_records(path: Path, max_records: Optional[int] = None) -> List[Tuple[str, str]]:
    max_records = normalize_record_limit(max_records)
    if max_records is None:
        return tmalign.read_fasta(str(path))

    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_chunks: List[str] = []
    with open_text_maybe_gzip(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks).upper()))
                    if len(records) >= max_records:
                        return records
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None and len(records) < max_records:
            records.append((header, "".join(seq_chunks).upper()))
    return records


def run_tmblast_search(
    primer_records: Sequence[Tuple[str, str]],
    fasta_path: Path,
    k: int = 4,
    min_tm: float = 30.0,
    min_identity: float = 50.0,
    cpus: int = 1,
    max_expansions: int = 100_000,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA database not found: {fasta_path}")

    records = read_fasta_records(fasta_path, max_records=max_records)
    if not records:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    rows: List[Tuple[object, ...]] = []
    worker_count = cpus if cpus and cpus > 0 else (os.cpu_count() or 1)
    for qid, raw_seq in primer_records:
        qseq = clean_primer_sequence(raw_seq)
        if k > len(qseq):
            raise ValueError(f"k={k} is longer than primer {qid} ({len(qseq)} bp).")
        expanded = tmblast.expand_degenerate(qseq, max_expansions=max_expansions, quiet=True)
        unique_expanded = list(dict.fromkeys(expanded))
        for exp_seq in unique_expanded:
            hits = tmblast.search_one_query(
                records,
                exp_seq,
                k,
                float(min_tm),
                float(min_identity),
                int(worker_count),
            )
            emitted = set()
            for tm, contig, start, end, strand, identity, q_align, db_align in hits:
                key = (qid, exp_seq, contig, start, end, strand, q_align, db_align)
                if key in emitted:
                    continue
                emitted.add(key)
                rows.append(
                    (
                        qid,
                        exp_seq,
                        round(float(tm), 2),
                        contig,
                        int(start),
                        int(end),
                        strand,
                        round(float(identity), 1),
                        q_align,
                        db_align,
                        None,
                    )
                )

    df = pd.DataFrame(rows, columns=HIT_COLUMNS)
    pair_ids = [qid for qid, _ in primer_records][:2]
    return attach_amplicon_lengths_to_hits(df, pair_ids)


def extract_taxid(text: object) -> Optional[str]:
    if text is None:
        return None
    value = str(text)
    patterns = (
        r"\|tax\|(\d+)(?:\||$)",
        r"\btaxid\s*[=:]\s*(\d+)\b",
        r"\[taxid\s*=\s*(\d+)\]",
    )
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def accession_candidates(text: object) -> List[str]:
    value = str(text or "").strip()
    candidates = []
    if value:
        candidates.append(value)
    try:
        accession = tax_analyzer.extract_accession_from_header(value)
    except Exception:
        accession = ""
    if accession:
        candidates.append(accession)
        candidates.append(accession.split(".", 1)[0])
    return list(dict.fromkeys(candidates))


def read_fasta_taxid_info(path: Path, max_records: Optional[int] = None) -> FastaTaxidInfo:
    max_records = normalize_record_limit(max_records)
    record_to_taxid: Dict[str, str] = {}
    accession_to_taxid: Dict[str, str] = {}
    accessions: List[str] = []
    origin_taxids: List[str] = []
    seen_records = 0

    with open_text_maybe_gzip(Path(path)) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            if not header:
                continue
            record_id = header.split()[0]
            taxid = extract_taxid(header)
            accession = tax_analyzer.extract_accession_from_header(header)
            if accession:
                accessions.append(accession)
            if taxid:
                record_to_taxid[record_id] = taxid
                if accession:
                    accession_to_taxid[accession] = taxid
                    accession_to_taxid[accession.split(".", 1)[0]] = taxid
                origin_taxids.append(taxid)
            seen_records += 1
            if max_records is not None and seen_records >= max_records:
                break

    return FastaTaxidInfo(
        record_to_taxid=record_to_taxid,
        accession_to_taxid=accession_to_taxid,
        accessions=list(dict.fromkeys(accessions)),
        origin_taxids=list(dict.fromkeys(origin_taxids)),
    )


def _find_existing_acc2taxid_dir(settings: PipelineSettings) -> Optional[Path]:
    candidates = []
    if settings.acc2taxid_dir:
        candidates.append(Path(settings.acc2taxid_dir))
    candidates.append(ROOT_DIR / "accession2taxid")
    candidates.append(TM_ALIGN_DIR / "accession2taxid")
    for path in candidates:
        if path.exists() and any(path.glob("*accession2taxid*.gz")):
            return path
    return None


def resolve_taxids(
    best_hits: pd.DataFrame,
    fasta_path: Path,
    settings: PipelineSettings,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warnings: List[str] = []
    info = read_fasta_taxid_info(fasta_path, max_records=settings.max_records)
    resolved = best_hits.copy()
    if "TaxID" not in resolved.columns:
        resolved["TaxID"] = pd.NA

    def from_local_sources(contig: object) -> Optional[str]:
        direct = extract_taxid(contig)
        if direct:
            return direct
        for candidate in accession_candidates(contig):
            if candidate in info.record_to_taxid:
                return info.record_to_taxid[candidate]
            if candidate in info.accession_to_taxid:
                return info.accession_to_taxid[candidate]
        return None

    if not resolved.empty:
        resolved["TaxID"] = resolved["contig"].map(from_local_sources)

    unresolved_contigs = []
    if not resolved.empty:
        unresolved_contigs = (
            resolved.loc[resolved["TaxID"].isna(), "contig"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    origin_taxids = list(info.origin_taxids)
    acc2taxid_dir = _find_existing_acc2taxid_dir(settings)
    if acc2taxid_dir and (unresolved_contigs or (not origin_taxids and info.accessions)):
        lookup_accs: List[str] = []
        for contig in unresolved_contigs:
            lookup_accs.extend(accession_candidates(contig))
        if not origin_taxids:
            lookup_accs.extend(info.accessions)
        lookup_accs = list(dict.fromkeys(lookup_accs))
        acc_map = tax_analyzer.map_accessions_to_taxids(
            lookup_accs,
            cache_path=str(settings.acc_cache) if settings.acc_cache else None,
            a2t_dir=str(acc2taxid_dir),
            acc_workers=settings.acc_workers,
        )
        if unresolved_contigs:
            def from_acc2taxid(contig: object) -> Optional[str]:
                for candidate in accession_candidates(contig):
                    taxid = acc_map.get(candidate)
                    if taxid:
                        return str(taxid)
                return None

            fallback = resolved.loc[resolved["TaxID"].isna(), "contig"].map(from_acc2taxid)
            resolved.loc[resolved["TaxID"].isna(), "TaxID"] = fallback
        if not origin_taxids:
            origin_taxids = [str(acc_map[acc]) for acc in info.accessions if acc_map.get(acc)]

    if not resolved.empty:
        unresolved_after = int(resolved["TaxID"].isna().sum())
        if unresolved_after:
            warnings.append(f"{unresolved_after} best-hit rows could not be resolved to TaxID.")
    if not origin_taxids:
        warnings.append("No origin TaxIDs were resolved from the FASTA database.")

    resolved["TaxID"] = resolved["TaxID"].astype("string")
    resolved.loc[resolved["TaxID"].isin(["None", "nan", "<NA>"]), "TaxID"] = pd.NA
    return resolved, origin_taxids, warnings


def build_rank_summary_from_db(
    taxdump: Dict[str, Dict[str, str]],
    rank: str,
    metrics: pd.DataFrame,
    origin_counts: Optional[pd.DataFrame],
    include_taxdump_index: bool = False,
) -> pd.DataFrame:
    if include_taxdump_index:
        return tax_analyzer.build_rank_summary(taxdump, rank, metrics, origin_counts)

    group_cols = tax_analyzer.GROUP_MAP[rank]
    frames: List[pd.DataFrame] = []
    if origin_counts is not None and not origin_counts.empty:
        frames.append(origin_counts[group_cols])
    if metrics is not None and not metrics.empty:
        frames.append(metrics[group_cols])

    if frames:
        base = pd.concat(frames, ignore_index=True).drop_duplicates()
    else:
        base = pd.DataFrame(columns=group_cols)

    if metrics is None or metrics.empty:
        metrics = pd.DataFrame(columns=group_cols + tax_analyzer.SUMMARY_METRIC_COLUMNS)
    merged = base.merge(metrics, on=group_cols, how="left")
    if origin_counts is not None and not origin_counts.empty:
        merged = merged.merge(origin_counts, on=group_cols, how="left")
    else:
        merged[tax_analyzer.ORIGIN_COUNT_COLUMN] = pd.NA

    for col in tax_analyzer.SUMMARY_TAXONOMY_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")
    for col in tax_analyzer.SUMMARY_METRIC_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    if tax_analyzer.ORIGIN_COUNT_COLUMN not in merged.columns:
        merged[tax_analyzer.ORIGIN_COUNT_COLUMN] = pd.NA

    merged = merged[tax_analyzer.SUMMARY_OUTPUT_COLUMNS]
    merged.sort_values(group_cols, inplace=True, ignore_index=True)
    return merged


def analyze_hits(
    hits: pd.DataFrame,
    fasta_path: Path,
    settings: PipelineSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if not Path(settings.taxdump_dir).exists():
        raise FileNotFoundError(f"taxdump directory not found: {settings.taxdump_dir}")

    raw_hits = hits.copy()
    if raw_hits.empty:
        raw_hits = pd.DataFrame(columns=HIT_COLUMNS)
    best_hits = tax_analyzer.filter_best_hits(raw_hits)
    best_hits, origin_taxids, tax_warnings = resolve_taxids(best_hits, fasta_path, settings)
    warnings.extend(tax_warnings)

    taxdump = tax_analyzer.load_taxdump(str(settings.taxdump_dir))
    hit_taxids = [str(t) for t in best_hits["TaxID"].dropna().unique()] if not best_hits.empty else []
    all_taxids = sorted(set(hit_taxids) | {str(t) for t in origin_taxids if t})
    lineage = tax_analyzer.lineage_from_taxdump(all_taxids, taxdump) if all_taxids else {}

    origin_counts = tax_analyzer.compute_origin_tax_counts(origin_taxids, lineage)
    accession_metrics = tax_analyzer.compute_accession_metrics(best_hits)
    accession_with_taxonomy = tax_analyzer.attach_taxonomy(accession_metrics, lineage)
    taxid_summary = tax_analyzer.compute_taxid_summary(accession_metrics, lineage)
    rank_metrics = tax_analyzer.summarize_for_tree_map(taxid_summary, settings.rank)
    rank_summary = build_rank_summary_from_db(
        taxdump,
        settings.rank,
        rank_metrics,
        origin_counts.get(settings.rank),
        include_taxdump_index=settings.include_taxdump_index,
    )
    return best_hits, accession_with_taxonomy, taxid_summary, rank_summary, warnings


def prepare_tree_input(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    for rank in tree_mapper.RANK_ORDER:
        if rank not in df.columns:
            df[rank] = pd.NA
        df[rank] = df[rank].replace("", pd.NA)

    numeric_cols = [
        "taxid_count",
        "mean_accession_count",
        "mean_amplicon_length",
        "mean_q1_Tm",
        "mean_q1_identity",
        "mean_q2_Tm",
        "mean_q2_identity",
        "origin_tax_count",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def render_tree_map(
    summary_df: pd.DataFrame,
    output_path: Path,
    taxdump_dir: Path = DEFAULT_TAXDUMP,
    depth: Optional[str] = "family",
    kingdom_filters: Optional[Sequence[str]] = None,
    taxon_names: Optional[Sequence[str]] = None,
    taxon_depths: Optional[Sequence[str]] = None,
) -> Path:
    if summary_df.empty:
        raise ValueError("Tree map cannot be rendered because the summary is empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_tree_input(summary_df)
    name_to_taxids, taxid_to_parent_rank, taxid_to_name = tree_mapper.load_taxdump(str(taxdump_dir))
    df_enriched = tree_mapper.fill_missing_ranks(df, name_to_taxids, taxid_to_parent_rank, taxid_to_name)

    flags = [flag for flag in (kingdom_filters or []) if flag in KINGDOM_FLAGS]
    df_filtered = tree_mapper.filter_by_flags(
        df_enriched,
        tree_mapper.RANK_ORDER,
        flags,
        name_to_taxids,
        taxid_to_parent_rank,
        taxid_to_name,
    )

    taxon_filters = tree_mapper.normalize_taxon_filters(list(taxon_names or []))
    df_filtered = tree_mapper.filter_by_taxon_names(
        df_filtered,
        taxon_filters,
        name_to_taxids,
        taxid_to_parent_rank,
        taxid_to_name,
    )

    taxon_depth_specs = tree_mapper.normalize_taxon_depths(list(taxon_depths or []))
    depth_limited_df, depth_series = tree_mapper.apply_taxon_depths(
        df_filtered,
        taxon_depth_specs,
        name_to_taxids,
        taxid_to_parent_rank,
        taxid_to_name,
    )

    if depth_series is not None:
        depth_limited_df = tree_mapper.aggregate_by_depth_assignments(depth_limited_df, depth_series)
    elif depth:
        depth_limited_df = tree_mapper.aggregate_to_depth(depth_limited_df, depth)

    ranks_in_use = [
        rank
        for rank in tree_mapper.RANK_ORDER
        if rank in depth_limited_df.columns and not depth_limited_df[rank].isna().all()
    ]
    if depth:
        allowed: List[str] = []
        for rank in tree_mapper.RANK_ORDER:
            allowed.append(rank)
            if rank == depth:
                break
        ranks_in_use = [rank for rank in ranks_in_use if rank in allowed]
    if not ranks_in_use:
        raise ValueError("No taxonomy remains after filtering.")

    root = tree_mapper.build_tree(
        depth_limited_df,
        ranks_in_use,
        name_to_taxids,
        taxid_to_parent_rank,
        taxid_to_name,
    )
    env_defaults = {
        "TREE_MAP_DPI": "120",
        "TREE_MAP_LIMIT_PIXELS": "12000",
        "TREE_MAP_MIN_HEIGHT": "8.0",
        "TREE_MAP_MIN_WIDTH": "12.0",
        "TREE_MAP_MAX_WIDTH": "48.0",
    }
    old_env: Dict[str, Optional[str]] = {}
    for key, value in env_defaults.items():
        old_env[key] = os.environ.get(key)
        os.environ.setdefault(key, value)
    try:
        tree_mapper.draw_tree(
            root,
            ranks_in_use,
            taxid_to_name,
            str(output_path),
            show_class_labels=bool(taxon_filters),
            exclude_label_ranks={depth} if depth else set(),
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return output_path


def write_outputs(result: PipelineResult, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hits": output_dir / "tm_hits.tsv",
        "best_hits": output_dir / "best_hits.tsv",
        "accession_metrics": output_dir / "accession_metrics.csv",
        "taxid_summary": output_dir / "taxid_summary.csv",
        "rank_summary": output_dir / "rank_summary.csv",
    }
    result.hits.to_csv(paths["hits"], sep="\t", index=False)
    result.best_hits.to_csv(paths["best_hits"], sep="\t", index=False)
    result.accession_metrics.to_csv(paths["accession_metrics"], index=False)
    result.taxid_summary.to_csv(paths["taxid_summary"], index=False)
    result.rank_summary.to_csv(paths["rank_summary"], index=False)
    return paths


def run_pipeline(settings: PipelineSettings) -> PipelineResult:
    settings.fasta_path = Path(settings.fasta_path)
    settings.taxdump_dir = Path(settings.taxdump_dir)
    if settings.rank not in RANK_CHOICES:
        raise ValueError(f"rank must be one of: {', '.join(RANK_CHOICES)}")
    if settings.depth and settings.depth not in RANK_CHOICES:
        raise ValueError(f"depth must be one of: {', '.join(RANK_CHOICES)}")

    primer_records = build_primer_records(settings.forward_primer, settings.reverse_primer)
    hits = run_tmblast_search(
        primer_records,
        settings.fasta_path,
        k=settings.k,
        min_tm=settings.min_tm,
        min_identity=settings.min_identity,
        cpus=settings.cpus,
        max_expansions=settings.max_expansions,
        max_records=settings.max_records,
    )
    best_hits, accession_metrics, taxid_summary, rank_summary, warnings = analyze_hits(
        hits,
        settings.fasta_path,
        settings,
    )

    output_paths: Dict[str, Path] = {}
    tree_path: Optional[Path] = None
    if settings.output_dir:
        output_dir = Path(settings.output_dir)
        placeholder = PipelineResult(
            hits=hits,
            best_hits=best_hits,
            accession_metrics=accession_metrics,
            taxid_summary=taxid_summary,
            rank_summary=rank_summary,
            tree_path=None,
            output_paths={},
            warnings=warnings,
        )
        output_paths = write_outputs(placeholder, output_dir)
        tree_path = output_dir / "tree_map.png"
    elif settings.render_tree:
        tmp = tempfile.NamedTemporaryFile(prefix="primer_tree_", suffix=".png", delete=False)
        tmp.close()
        tree_path = Path(tmp.name)

    if settings.render_tree and tree_path is not None:
        try:
            render_tree_map(
                rank_summary,
                tree_path,
                taxdump_dir=settings.taxdump_dir,
                depth=settings.depth,
                kingdom_filters=settings.kingdom_filters,
                taxon_names=settings.taxon_names,
                taxon_depths=settings.taxon_depths,
            )
            output_paths["tree"] = tree_path
        except Exception as exc:
            warnings.append(f"Tree map rendering skipped: {exc}")
            tree_path = None

    return PipelineResult(
        hits=hits,
        best_hits=best_hits,
        accession_metrics=accession_metrics,
        taxid_summary=taxid_summary,
        rank_summary=rank_summary,
        tree_path=tree_path,
        output_paths=output_paths,
        warnings=warnings,
    )


def relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Primer Evaluation Tool", layout="wide")

    @st.cache_data(show_spinner=False)
    def csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    @st.cache_data(show_spinner=False)
    def tsv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False, sep="\t").encode("utf-8")

    databases = discover_fasta_databases()
    default_db = TM_ALIGN_DIR / "test-mito.fa"
    if default_db not in databases and databases:
        default_db = databases[0]

    st.title("Primer Evaluation Tool")

    with st.sidebar:
        st.subheader("Database")
        if databases:
            db_labels = [relative_label(path) for path in databases]
            db_index = databases.index(default_db) if default_db in databases else 0
            selected_label = st.selectbox("FASTA database", db_labels, index=db_index)
            fasta_path = databases[db_labels.index(selected_label)]
        else:
            fasta_path = Path(st.text_input("FASTA database path", value=str(default_db)))

        custom_db = st.text_input("Custom FASTA path", value="")
        if custom_db.strip():
            fasta_path = Path(custom_db).expanduser()

        taxdump_dir = Path(st.text_input("taxdump directory", value=str(ROOT_DIR / "taxdump"))).expanduser()

        st.subheader("Search")
        k = st.number_input("k-mer size", min_value=1, max_value=12, value=4, step=1)
        min_tm = st.number_input("Minimum Tm", value=30.0, step=1.0)
        min_identity = st.number_input("Minimum identity (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
        cpus = st.number_input("CPU processes", min_value=1, max_value=16, value=1, step=1)
        max_expansions = st.number_input("Max degenerate expansions", min_value=1, value=100_000, step=1000)
        max_records = st.number_input("Record limit (0 = all)", min_value=0, value=0, step=100)

        st.subheader("Taxonomy")
        rank = st.selectbox("Summary rank", list(RANK_CHOICES), index=list(RANK_CHOICES).index("family"))
        depth = st.selectbox("Tree depth", list(RANK_CHOICES), index=list(RANK_CHOICES).index("family"))
        kingdom_filters = st.multiselect(
            "Kingdom filter",
            list(KINGDOM_FLAGS),
            format_func={
                "a": "Animalia / Metazoa",
                "p": "Plantae",
                "f": "Fungi",
                "r": "Protista / other eukaryotes",
                "m": "Monera",
            }.get,
        )
        taxon_text = st.text_input("Taxon filter", value="")
        render_tree = st.checkbox("Render tree map", value=True)
        include_taxdump_index = st.checkbox("Include all NCBI taxa at rank", value=False)

    col1, col2 = st.columns(2)
    with col1:
        forward_primer = st.text_area("Forward primer (q1)", value="GTCGGTAAAACTCGTGCCAGC", height=90)
    with col2:
        reverse_primer = st.text_area("Reverse primer (q2)", value="CATAGTGGGGTATCTAATCCCAGTTTG", height=90)

    if not st.button("Run evaluation", type="primary"):
        return

    taxon_names = [part.strip() for part in taxon_text.replace(",", "\n").splitlines() if part.strip()]
    with tempfile.TemporaryDirectory(prefix="primer_eval_streamlit_") as tmpdir:
        settings = PipelineSettings(
            fasta_path=fasta_path,
            taxdump_dir=taxdump_dir,
            forward_primer=forward_primer,
            reverse_primer=reverse_primer,
            k=int(k),
            min_tm=float(min_tm),
            min_identity=float(min_identity),
            cpus=int(cpus),
            max_expansions=int(max_expansions),
            max_records=int(max_records) if int(max_records) > 0 else None,
            rank=rank,
            depth=depth,
            kingdom_filters=list(kingdom_filters),
            taxon_names=taxon_names,
            output_dir=Path(tmpdir),
            render_tree=render_tree,
            include_taxdump_index=include_taxdump_index,
        )
        try:
            with st.spinner("Running primer evaluation..."):
                result = run_pipeline(settings)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        for warning in result.warnings:
            st.warning(warning)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hits", f"{len(result.hits):,}")
        m2.metric("Best hits", f"{len(result.best_hits):,}")
        m3.metric("Accessions", f"{len(result.accession_metrics):,}")
        m4.metric(f"{rank.title()} rows", f"{len(result.rank_summary):,}")

        tabs = st.tabs(["Hits", "Accessions", "Taxa", "Tree"])
        with tabs[0]:
            st.dataframe(result.hits, use_container_width=True, height=420)
            st.download_button("Download hits TSV", tsv_bytes(result.hits), "tm_hits.tsv", "text/tab-separated-values")
        with tabs[1]:
            st.dataframe(result.accession_metrics, use_container_width=True, height=420)
            st.download_button(
                "Download accession CSV",
                csv_bytes(result.accession_metrics),
                "accession_metrics.csv",
                "text/csv",
            )
        with tabs[2]:
            st.dataframe(result.rank_summary, use_container_width=True, height=420)
            st.download_button("Download summary CSV", csv_bytes(result.rank_summary), "rank_summary.csv", "text/csv")
        with tabs[3]:
            if result.tree_path and result.tree_path.exists():
                tree_png = result.tree_path.read_bytes()
                st.image(tree_png, use_container_width=True)
                st.download_button("Download tree PNG", tree_png, "tree_map.png", "image/png")
            else:
                st.info("Tree map was not generated.")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA), help="Target FASTA database")
    parser.add_argument("--taxdump", default=str(DEFAULT_TAXDUMP), help="taxdump directory")
    parser.add_argument("--forward", "--q1", dest="forward", required=True, help="Forward primer sequence")
    parser.add_argument("--reverse", "--q2", dest="reverse", required=True, help="Reverse primer sequence")
    parser.add_argument("--k", type=int, default=4, help="k-mer window size")
    parser.add_argument("--min-tm", type=float, default=30.0, help="Minimum Tm to report")
    parser.add_argument("--min-identity", type=float, default=50.0, help="Minimum identity percent")
    parser.add_argument("--cpus", type=int, default=1, help="CPU processes")
    parser.add_argument("--max-expansions", type=int, default=100_000, help="Max degenerate expansions per primer")
    parser.add_argument("--max-records", type=int, default=0, help="Limit FASTA records for test runs; 0 means all")
    parser.add_argument("--rank", choices=RANK_CHOICES, default="family", help="Summary rank")
    parser.add_argument("--depth", choices=RANK_CHOICES, default="family", help="Tree drawing depth")
    parser.add_argument("--kingdom", choices=KINGDOM_FLAGS, action="append", default=[], help="Tree kingdom filter")
    parser.add_argument("--taxon", action="append", default=[], help="Tree taxon filter")
    parser.add_argument("--taxon-depth", action="append", default=[], help="Tree taxon/depth filter, e.g. Chordata/order")
    parser.add_argument("--include-taxdump-index", action="store_true", help="Include all NCBI taxa at the selected rank")
    parser.add_argument("--no-tree", action="store_true", help="Skip tree image rendering")
    parser.add_argument("--output-dir", default="primer_eval_output", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    settings = PipelineSettings(
        fasta_path=Path(args.fasta),
        taxdump_dir=Path(args.taxdump),
        forward_primer=args.forward,
        reverse_primer=args.reverse,
        k=args.k,
        min_tm=args.min_tm,
        min_identity=args.min_identity,
        cpus=args.cpus,
        max_expansions=args.max_expansions,
        max_records=normalize_record_limit(args.max_records),
        rank=args.rank,
        depth=args.depth,
        kingdom_filters=args.kingdom,
        taxon_names=args.taxon,
        taxon_depths=args.taxon_depth,
        output_dir=Path(args.output_dir),
        render_tree=not args.no_tree,
        include_taxdump_index=args.include_taxdump_index,
    )
    result = run_pipeline(settings)
    print(f"[info] hits: {len(result.hits):,}")
    print(f"[info] best hits: {len(result.best_hits):,}")
    print(f"[info] accession metrics: {len(result.accession_metrics):,}")
    print(f"[info] {settings.rank} summary rows: {len(result.rank_summary):,}")
    for key, path in result.output_paths.items():
        print(f"[info] wrote {key}: {path}")
    for warning in result.warnings:
        print(f"[warn] {warning}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 1 or "--streamlit" in sys.argv:
        if "--streamlit" in sys.argv:
            sys.argv.remove("--streamlit")
        run_streamlit_app()
    else:
        main()
