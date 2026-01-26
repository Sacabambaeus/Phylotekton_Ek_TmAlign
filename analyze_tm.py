#!/usr/bin/env python3
"""
Organize tm_result_tm30.tsv offline using local accession2taxid and taxdump data,
with origin TaxID counts derived from a GenBank FASTA.

Outputs
-------
1) Optional per-AccessionID metrics with taxonomy.
2) Class/order/family summaries usable by tree_map1.15.py.
"""

import argparse
import gzip
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd


UNKNOWN_TAXON = "Unknown"
TAXONOMY_RANKS = ["phylum", "class", "order", "family", "genus", "species"]
SUMMARY_TAXONOMY_COLUMNS = ["phylum", "class", "order", "family", "genus"]
SUMMARY_METRIC_COLUMNS = [
    "taxid_count",
    "mean_accession_count",
    "mean_q1_Tm",
    "mean_q1_identity",
    "mean_q2_Tm",
    "mean_q2_identity",
]
ORIGIN_COUNT_COLUMN = "origin_tax_count"
SUMMARY_OUTPUT_COLUMNS = SUMMARY_TAXONOMY_COLUMNS + SUMMARY_METRIC_COLUMNS + [ORIGIN_COUNT_COLUMN]
GROUP_MAP = {
    "class": ["phylum", "class"],
    "order": ["phylum", "class", "order"],
    "family": ["phylum", "class", "order", "family"],
    "genus": ["phylum", "class", "order", "family", "genus"],
}
FASTA_PREFIXES = {"gb", "emb", "dbj", "ref", "gi", "sp", "tr", "lcl"}


def load_cache(path: Optional[str]) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_cache(path: Optional[str], mapping: Dict[str, object]) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    os.replace(tmp, path)


def normalize_taxon(value: object) -> str:
    if value is None:
        return UNKNOWN_TAXON
    try:
        if pd.isna(value):
            return UNKNOWN_TAXON
    except Exception:
        pass
    text = str(value).strip()
    return text if text else UNKNOWN_TAXON


def normalize_taxonomy_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = UNKNOWN_TAXON
        df[col] = df[col].map(normalize_taxon)
    return df


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str, label: str) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)
    print(f"[info] Wrote {label} -> {path}")


def open_text_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def extract_accession_from_header(header: str) -> str:
    tokens = re.split(r"[|\s]+", header.strip())
    for token in tokens:
        if not token:
            continue
        candidate = token.strip()
        lowered = candidate.lower()
        if lowered in FASTA_PREFIXES:
            continue
        if "=" in candidate:
            continue
        if not re.search(r"[A-Za-z]", candidate):
            continue
        if re.search(r"\d", candidate):
            return candidate
    return ""


def read_fasta_accessions(path: str) -> List[str]:
    accessions: List[str] = []
    with open_text_maybe_gzip(path) as fh:
        for line in fh:
            if line.startswith(">"):
                header = line[1:].strip()
                acc = extract_accession_from_header(header)
                if acc:
                    accessions.append(acc)
    return accessions


def _scan_accession2taxid_file(
    path: str,
    pending: set,
    base_to_accs: Dict[str, List[str]],
) -> Dict[str, str]:
    found: Dict[str, str] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            _ = fh.readline()  # skip header
            for line in fh:
                parts = line.rstrip("\n").split("\t", 3)
                if len(parts) < 3:
                    continue
                base_acc, acc_ver, taxid = parts[0], parts[1], parts[2]
                if acc_ver in pending:
                    found[acc_ver] = taxid
                if base_acc in base_to_accs:
                    for acc in base_to_accs[base_acc]:
                        if acc in pending:
                            found[acc] = taxid
    except FileNotFoundError:
        print(f"[warn] accession2taxid not found: {path}")
    except Exception as exc:
        print(f"[warn] failed reading {path}: {exc}")
    return found


def _map_accessions_to_taxids_local(
    accessions: List[str],
    a2t_files: List[str],
    workers: int = 1,
) -> Dict[str, Optional[str]]:
    """Resolve accession -> TaxID using local accession2taxid .gz files.

    Matches accession.version first; falls back to accession without version.
    """
    pending = {acc for acc in accessions if acc}
    base_to_accs: Dict[str, List[str]] = {}
    for acc in pending:
        base = acc.split(".", 1)[0]
        base_to_accs.setdefault(base, []).append(acc)

    found: Dict[str, str] = {}
    if workers > 1 and len(a2t_files) > 1 and pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_scan_accession2taxid_file, path, pending, base_to_accs)
                for path in a2t_files
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    found.update(result)
    else:
        for path in a2t_files:
            if not pending:
                break
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                    _ = fh.readline()  # skip header
                    for line in fh:
                        if not pending:
                            break
                        parts = line.rstrip("\n").split("\t", 3)
                        if len(parts) < 3:
                            continue
                        base_acc, acc_ver, taxid = parts[0], parts[1], parts[2]
                        if acc_ver in pending:
                            found[acc_ver] = taxid
                            pending.discard(acc_ver)
                        if base_acc in base_to_accs:
                            for acc in base_to_accs[base_acc]:
                                if acc in pending:
                                    found[acc] = taxid
                                    pending.discard(acc)
            except FileNotFoundError:
                print(f"[warn] accession2taxid not found: {path}")
            except Exception as exc:
                print(f"[warn] failed reading {path}: {exc}")

    out: Dict[str, Optional[str]] = {}
    for acc in accessions:
        out[acc] = found.get(acc)
    return out


def _find_a2t_files(a2t_dir: Optional[str]) -> List[str]:
    if not a2t_dir:
        return []
    try:
        entries = [
            os.path.join(a2t_dir, name)
            for name in os.listdir(a2t_dir)
            if name.endswith(".gz") and "accession2taxid" in name
        ]
    except OSError:
        entries = []

    def sort_key(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        if base.startswith("nucl_gb"):
            return (0, base)
        if base.startswith("nucl_wgs"):
            return (1, base)
        if base.startswith("prot"):
            return (2, base)
        return (9, base)

    entries.sort(key=sort_key)
    return entries


def map_accessions_to_taxids(
    accessions: List[str],
    cache_path: Optional[str],
    a2t_dir: Optional[str],
    acc_workers: int = 1,
) -> Dict[str, Optional[str]]:
    pending = list(dict.fromkeys(acc for acc in accessions if acc))

    cache_raw = load_cache(cache_path)
    cache = {str(k): str(v) for k, v in cache_raw.items() if v is not None}
    resolved: Dict[str, Optional[str]] = {}
    for acc in list(pending):
        if acc in cache:
            resolved[acc] = cache[acc]
    pending = [acc for acc in pending if acc not in resolved]

    if pending:
        a2t_files = _find_a2t_files(a2t_dir)
        if a2t_files:
            print(f"[info] Using local accession2taxid files: {len(a2t_files)} found")
            local_map = _map_accessions_to_taxids_local(pending, a2t_files, workers=acc_workers)
            resolved.update({k: v for k, v in local_map.items() if v})
            pending = [acc for acc in pending if acc not in resolved]

    if pending:
        print(f"[warn] {len(pending)} accessions unresolved locally; marking as None")
        for acc in pending:
            resolved[acc] = None

    if cache_path:
        cache_raw.update({k: v for k, v in resolved.items() if v})
        save_cache(cache_path, cache_raw)

    return resolved


def load_taxdump(taxdump_dir: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not taxdump_dir:
        raise SystemExit("--taxdump is required (expects nodes.dmp and names.dmp)")
    nodes_path = os.path.join(taxdump_dir, "nodes.dmp")
    names_path = os.path.join(taxdump_dir, "names.dmp")
    if not (os.path.exists(nodes_path) and os.path.exists(names_path)):
        raise SystemExit(f"taxdump directory missing nodes.dmp or names.dmp: {taxdump_dir}")

    parent: Dict[str, str] = {}
    rank: Dict[str, str] = {}
    with open(nodes_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            tid, par, rnk = parts[0], parts[1], parts[2]
            parent[tid] = par
            rank[tid] = rnk

    name: Dict[str, str] = {}
    with open(names_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            tid, nm, _, cls = parts[0], parts[1], parts[2], parts[3]
            if cls == "scientific name" and tid not in name:
                name[tid] = nm

    return {"parent": parent, "rank": rank, "name": name}


def lineage_from_taxdump(taxids: List[str], taxdump: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    parent = taxdump["parent"]
    rank = taxdump["rank"]
    name = taxdump["name"]

    out: Dict[str, Dict[str, str]] = {}
    for tid in dict.fromkeys(taxids):
        if not tid:
            continue
        lineage_ranks: Dict[str, str] = {}
        t = tid
        seen = set()
        while t and t not in seen:
            seen.add(t)
            nm = name.get(t, "")
            rnk = rank.get(t, "")
            if rnk and rnk not in lineage_ranks:
                lineage_ranks[rnk] = nm
            pt = parent.get(t)
            if pt == t:
                break
            t = pt
        info = {col: normalize_taxon(lineage_ranks.get(col, "")) for col in TAXONOMY_RANKS}
        info["scientific_name"] = normalize_taxon(name.get(tid, ""))
        out[tid] = info
    return out


def build_taxonomy_index(taxdump: Dict[str, Dict[str, str]], rank: str) -> pd.DataFrame:
    if rank not in GROUP_MAP:
        raise ValueError(f"Unsupported rank: {rank}")
    taxids = [tid for tid, rnk in taxdump["rank"].items() if rnk == rank]
    if not taxids:
        return pd.DataFrame(columns=GROUP_MAP[rank])

    lineage = lineage_from_taxdump(taxids, taxdump)
    records: List[Dict[str, str]] = []
    for tid in taxids:
        info = lineage.get(tid, {})
        record = {col: normalize_taxon(info.get(col, "")) for col in GROUP_MAP[rank]}
        records.append(record)

    df = pd.DataFrame(records, columns=GROUP_MAP[rank])
    df = normalize_taxonomy_frame(df, GROUP_MAP[rank])
    df.drop_duplicates(inplace=True)
    df.sort_values(GROUP_MAP[rank], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_origin_tax_counts(
    taxids: List[str],
    lineage: Dict[str, Dict[str, str]],
) -> Dict[str, pd.DataFrame]:
    base_cols = ["phylum", "class", "order", "family", "genus"]
    if not taxids:
        return {
            rank: pd.DataFrame(columns=GROUP_MAP[rank] + [ORIGIN_COUNT_COLUMN])
            for rank in ("class", "order", "family", "genus")
        }

    unique_taxids = [tid for tid in dict.fromkeys(taxids) if tid]
    records: List[Dict[str, str]] = []
    for tid in unique_taxids:
        info = lineage.get(tid, {})
        record: Dict[str, str] = {"TaxID": tid}
        for col in base_cols:
            record[col] = normalize_taxon(info.get(col, ""))
        records.append(record)

    df = pd.DataFrame(records)
    df = normalize_taxonomy_frame(df, base_cols)

    counts: Dict[str, pd.DataFrame] = {}
    for rank in ("class", "order", "family", "genus"):
        group_cols = GROUP_MAP[rank]
        if df.empty:
            counts[rank] = pd.DataFrame(columns=group_cols + [ORIGIN_COUNT_COLUMN])
        else:
            cnt = (
                df.groupby(group_cols, dropna=False)["TaxID"]
                .nunique()
                .reset_index()
                .rename(columns={"TaxID": ORIGIN_COUNT_COLUMN})
            )
            counts[rank] = cnt
    return counts


def read_tm_results(path: str) -> pd.DataFrame:
    columns = [
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
    ]
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        dtype={
            "query_id": str,
            "expanded_query": str,
            "Tm": float,
            "contig": str,
            "start": int,
            "end": int,
            "strand": str,
            "identity": float,
            "q_align": str,
            "db_align": str,
        },
    )
    return df


def compute_span_bp(rec: Dict[str, Optional[int]]) -> Optional[int]:
    q1_start = rec.get("q1_start")
    q1_end = rec.get("q1_end")
    q2_start = rec.get("q2_start")
    q2_end = rec.get("q2_end")
    if None in (q1_start, q1_end, q2_start, q2_end):
        return None
    positions = [int(q1_start), int(q1_end), int(q2_start), int(q2_end)]
    return max(positions) - min(positions)


def filter_best_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = ["query_id", "contig", "Tm", "identity"]
    df_sorted = df.sort_values(sort_cols, ascending=[True, True, False, False])
    best = df_sorted.drop_duplicates(subset=["query_id", "contig"], keep="first")
    return best


def compute_accession_metrics(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "TaxID",
        "AccessionID",
        "span_bp",
        "q1_start",
        "q1_end",
        "q1_Tm",
        "q1_identity",
        "q2_start",
        "q2_end",
        "q2_Tm",
        "q2_identity",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    df_work = df[df["query_id"].isin(["q1", "q2"])].copy()
    if df_work.empty:
        return pd.DataFrame(columns=columns)

    df_work = df_work[["TaxID", "contig", "query_id", "start", "end", "Tm", "identity"]]
    pivot = df_work.pivot_table(
        index=["TaxID", "contig"],
        columns="query_id",
        values=["start", "end", "Tm", "identity"],
        aggfunc="first",
    )
    pivot.columns = [f"{query}_{metric}" for metric, query in pivot.columns]
    acc_df = pivot.reset_index().rename(columns={"contig": "AccessionID"})

    for col in (
        "q1_start",
        "q1_end",
        "q1_Tm",
        "q1_identity",
        "q2_start",
        "q2_end",
        "q2_Tm",
        "q2_identity",
    ):
        if col not in acc_df.columns:
            acc_df[col] = pd.NA

    pos_cols = ["q1_start", "q1_end", "q2_start", "q2_end"]
    pos = acc_df[pos_cols].apply(pd.to_numeric, errors="coerce")
    acc_df["span_bp"] = pos.max(axis=1, skipna=False) - pos.min(axis=1, skipna=False)

    acc_df = acc_df[
        [
            "TaxID",
            "AccessionID",
            "span_bp",
            "q1_start",
            "q1_end",
            "q1_Tm",
            "q1_identity",
            "q2_start",
            "q2_end",
            "q2_Tm",
            "q2_identity",
        ]
    ]

    for col in ("span_bp", "q1_start", "q1_end", "q2_start", "q2_end"):
        acc_df[col] = pd.to_numeric(acc_df[col], errors="coerce").astype("Int64")
    acc_df["TaxID"] = acc_df["TaxID"].astype("string")
    acc_df.loc[acc_df["TaxID"].isin(["None", "nan", "<NA>"]), "TaxID"] = pd.NA
    acc_df = acc_df.dropna(subset=["TaxID"]).copy()
    acc_df["TaxID"] = acc_df["TaxID"].astype(str)
    return acc_df


def attach_taxonomy(df: pd.DataFrame, lineage: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    lineage_df = pd.DataFrame.from_dict(lineage, orient="index")
    if lineage_df.empty:
        lineage_df = pd.DataFrame(columns=["TaxID"] + TAXONOMY_RANKS)
    else:
        lineage_df.index.name = "TaxID"
        lineage_df = lineage_df.reset_index()
    lineage_df["TaxID"] = lineage_df.get("TaxID", pd.Series(dtype=str)).astype(str)
    lineage_df = normalize_taxonomy_frame(lineage_df, TAXONOMY_RANKS)

    merged = df.merge(lineage_df[["TaxID"] + TAXONOMY_RANKS], on="TaxID", how="left")
    merged = normalize_taxonomy_frame(merged, TAXONOMY_RANKS)
    return merged


def compute_taxid_summary(acc_df: pd.DataFrame, lineage: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    columns = [
        "TaxID",
        "accession_count",
        "q1_mean_Tm",
        "q1_mean_identity",
        "q2_mean_Tm",
        "q2_mean_identity",
    ] + SUMMARY_TAXONOMY_COLUMNS
    if acc_df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        acc_df.groupby("TaxID", dropna=False)
        .agg(
            accession_count=("AccessionID", "nunique"),
            q1_mean_Tm=("q1_Tm", "mean"),
            q1_mean_identity=("q1_identity", "mean"),
            q2_mean_Tm=("q2_Tm", "mean"),
            q2_mean_identity=("q2_identity", "mean"),
        )
        .reset_index()
    )
    summary["TaxID"] = summary["TaxID"].astype(str)

    lineage_df = pd.DataFrame.from_dict(lineage, orient="index")
    if lineage_df.empty:
        lineage_df = pd.DataFrame(columns=["TaxID"] + SUMMARY_TAXONOMY_COLUMNS)
    else:
        lineage_df.index.name = "TaxID"
        lineage_df = lineage_df.reset_index()
    lineage_df["TaxID"] = lineage_df.get("TaxID", pd.Series(dtype=str)).astype(str)
    lineage_df = normalize_taxonomy_frame(lineage_df, SUMMARY_TAXONOMY_COLUMNS)

    summary = summary.merge(lineage_df[["TaxID"] + SUMMARY_TAXONOMY_COLUMNS], on="TaxID", how="left")
    summary = normalize_taxonomy_frame(summary, SUMMARY_TAXONOMY_COLUMNS)
    return summary


def summarize_for_tree_map(df: pd.DataFrame, rank: str) -> pd.DataFrame:
    if rank not in GROUP_MAP:
        raise ValueError(f"Unsupported rank: {rank}")
    group_cols = GROUP_MAP[rank]
    if df.empty:
        return pd.DataFrame(columns=group_cols + SUMMARY_METRIC_COLUMNS)

    grouped = df.groupby(group_cols, dropna=False)
    summary = (
        grouped.agg(
            taxid_count=("TaxID", "nunique"),
            mean_accession_count=("accession_count", "mean"),
            mean_q1_Tm=("q1_mean_Tm", "mean"),
            mean_q1_identity=("q1_mean_identity", "mean"),
            mean_q2_Tm=("q2_mean_Tm", "mean"),
            mean_q2_identity=("q2_mean_identity", "mean"),
        )
        .reset_index()
    )
    return summary


def build_rank_summary(
    taxdump: Dict[str, Dict[str, str]],
    rank: str,
    metrics: pd.DataFrame,
    origin_counts: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if rank not in GROUP_MAP:
        raise ValueError(f"Unsupported rank: {rank}")
    group_cols = GROUP_MAP[rank]

    base = build_taxonomy_index(taxdump, rank)
    extra_groups: List[pd.DataFrame] = []
    if not metrics.empty:
        extra_groups.append(metrics[group_cols])
    if origin_counts is not None and not origin_counts.empty:
        extra_groups.append(origin_counts[group_cols])
    if extra_groups:
        extra = pd.concat(extra_groups, ignore_index=True).drop_duplicates()
        base = pd.concat([base, extra], ignore_index=True).drop_duplicates()

    merged = base.merge(metrics, on=group_cols, how="left")
    if origin_counts is not None and not origin_counts.empty:
        merged = merged.merge(origin_counts, on=group_cols, how="left")
    else:
        merged[ORIGIN_COUNT_COLUMN] = pd.NA

    for col in SUMMARY_TAXONOMY_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
        else:
            merged[col] = merged[col].fillna("")
    for col in SUMMARY_METRIC_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    if ORIGIN_COUNT_COLUMN not in merged.columns:
        merged[ORIGIN_COUNT_COLUMN] = pd.NA

    merged = merged[SUMMARY_OUTPUT_COLUMNS]
    merged.sort_values(group_cols, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate class/order/family summaries compatible with tree_map1.15.py",
    )
    parser.add_argument("input", help="Input TSV from TmBLAST")
    parser.add_argument("--fasta", required=True, help="Input GenBank FASTA for origin taxon counts")

    default_a2t = os.environ.get("ACC2TAXID_DIR", "accession2taxid")
    default_taxdump = os.environ.get("TAXDUMP_DIR", "taxdump")
    parser.add_argument(
        "--acc2taxid",
        nargs="?",
        const=default_a2t,
        default=default_a2t,
        help="Directory containing accession2taxid *.gz files",
    )
    parser.add_argument(
        "--taxdump",
        nargs="?",
        const=default_taxdump,
        default=default_taxdump,
        help="Directory containing taxdump files nodes.dmp/names.dmp",
    )
    parser.add_argument("--c", dest="class_out", help="Class summary output CSV path")
    parser.add_argument("--o", dest="order_out", help="Order summary output CSV path")
    parser.add_argument("--f", dest="family_out", help="Family summary output CSV path")
    parser.add_argument("--g", dest="genus_out", help="Genus summary output CSV path")
    parser.add_argument(
        "--a",
        "--accession-out",
        dest="accession_out",
        default=None,
        help="Optional per-accession output CSV path",
    )
    parser.add_argument(
        "--acc-cache",
        default=".organize_acc_cache.json",
        help="Optional cache file for accession -> TaxID mappings",
    )
    parser.add_argument(
        "--acc-workers",
        type=int,
        default=1,
        help="Number of threads to scan multiple accession2taxid files (default: 1)",
    )
    parser.add_argument(
        "--tax-cache",
        default=".organize_tax_cache.json",
        help="Optional cache file for TaxID -> taxonomy mappings",
    )
    args = parser.parse_args()

    if not any([args.class_out, args.order_out, args.family_out, args.genus_out, args.accession_out]):
        raise SystemExit("At least one output option is required: --c/--o/--f/--g/--a")

    df_raw = read_tm_results(args.input)
    print(f"[info] Loaded {len(df_raw):,} rows from {args.input}")

    df_best = filter_best_hits(df_raw)
    print(f"[info] Best hits retained: {len(df_best):,} rows (unique per query/accession)")

    fasta_accessions = read_fasta_accessions(args.fasta)
    print(f"[info] Loaded {len(fasta_accessions):,} FASTA headers from {args.fasta}")

    accessions_tsv = df_best["contig"].dropna().astype(str).tolist()
    unique_tsv = list(dict.fromkeys(accessions_tsv))
    unique_fasta = list(dict.fromkeys(fasta_accessions))
    all_accessions = list(dict.fromkeys(unique_tsv + unique_fasta))
    print(
        "[info] Resolving "
        f"{len(all_accessions):,} unique accessions (TSV {len(unique_tsv):,} + FASTA {len(unique_fasta):,}) "
        "to TaxIDs..."
    )
    acc2tax = map_accessions_to_taxids(
        all_accessions,
        cache_path=args.acc_cache,
        a2t_dir=args.acc2taxid,
        acc_workers=args.acc_workers,
    )
    resolved_all = sum(1 for acc in all_accessions if acc2tax.get(acc))
    unresolved_all = len(all_accessions) - resolved_all
    print(f"[info] Accession->TaxID resolved: {resolved_all:,} ok, {unresolved_all:,} unresolved")

    resolved_fasta = sum(1 for acc in unique_fasta if acc2tax.get(acc))
    unresolved_fasta = len(unique_fasta) - resolved_fasta
    print(f"[info] FASTA accessions resolved: {resolved_fasta:,} ok, {unresolved_fasta:,} unresolved")

    df_best = df_best.copy()
    df_best["TaxID"] = df_best["contig"].map(acc2tax)

    fasta_taxids = [str(acc2tax[acc]) for acc in unique_fasta if acc2tax.get(acc)]
    taxids = [str(t) for t in df_best["TaxID"].dropna().unique()]
    taxids_all = sorted(set(taxids) | set(fasta_taxids))
    print(f"[info] Loading taxonomy for {len(taxids_all):,} TaxIDs from {args.taxdump}...")
    taxdump = load_taxdump(args.taxdump)

    tax_cache_raw = load_cache(args.tax_cache)
    tax_cache = {str(k): v for k, v in tax_cache_raw.items() if isinstance(v, dict)}
    lineage_missing = [tid for tid in taxids_all if tid not in tax_cache]
    if lineage_missing:
        lineage_new = lineage_from_taxdump(lineage_missing, taxdump)
        tax_cache.update(lineage_new)
        save_cache(args.tax_cache, tax_cache)
    lineage = {tid: tax_cache.get(tid, {}) for tid in taxids_all}

    origin_counts = compute_origin_tax_counts(fasta_taxids, lineage)

    accession_metrics = compute_accession_metrics(df_best)
    if args.accession_out:
        accession_with_tax = attach_taxonomy(accession_metrics, lineage)
        accession_with_tax.sort_values(
            SUMMARY_TAXONOMY_COLUMNS + ["TaxID", "AccessionID"],
            inplace=True,
            na_position="last",
        )
        accession_with_tax.reset_index(drop=True, inplace=True)
        write_csv(accession_with_tax, args.accession_out, "per-accession metrics")

    if any([args.class_out, args.order_out, args.family_out, args.genus_out]):
        taxid_summary = compute_taxid_summary(accession_metrics, lineage)

        if args.class_out:
            class_metrics = summarize_for_tree_map(taxid_summary, "class")
            class_summary = build_rank_summary(taxdump, "class", class_metrics, origin_counts.get("class"))
            write_csv(class_summary, args.class_out, "class summary")

        if args.order_out:
            order_metrics = summarize_for_tree_map(taxid_summary, "order")
            order_summary = build_rank_summary(taxdump, "order", order_metrics, origin_counts.get("order"))
            write_csv(order_summary, args.order_out, "order summary")

        if args.family_out:
            family_metrics = summarize_for_tree_map(taxid_summary, "family")
            family_summary = build_rank_summary(taxdump, "family", family_metrics, origin_counts.get("family"))
            write_csv(family_summary, args.family_out, "family summary")

        if args.genus_out:
            genus_metrics = summarize_for_tree_map(taxid_summary, "genus")
            genus_summary = build_rank_summary(taxdump, "genus", genus_metrics, origin_counts.get("genus"))
            write_csv(genus_summary, args.genus_out, "genus summary")


if __name__ == "__main__":
    main()
