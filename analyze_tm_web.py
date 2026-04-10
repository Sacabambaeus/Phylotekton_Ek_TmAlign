from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import streamlit as st

from analyze_tm import (
    attach_taxonomy,
    build_rank_summary,
    compute_accession_metrics,
    compute_origin_tax_counts,
    compute_taxid_summary,
    filter_best_hits,
    lineage_from_taxdump,
    load_taxdump,
    map_accessions_to_taxids,
    read_fasta_accessions,
    read_tm_results,
    summarize_for_tree_map,
)


SESSION_KEY = "analyze_tm_web_state"
OUTPUT_LABELS = {
    "accession": "Per-accession metrics",
    "class": "Class summary",
    "order": "Order summary",
    "family": "Family summary",
    "genus": "Genus summary",
}
OUTPUT_FILE_NAMES = {
    "accession": "accession_metrics_web.csv",
    "class": "class_summary_web.csv",
    "order": "order_summary_web.csv",
    "family": "family_summary_web.csv",
    "genus": "genus_summary_web.csv",
}


def _save_uploaded_file(uploaded_file, destination: str) -> str:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return destination


def _resolve_taxdump_dir(
    temp_root: str,
    names_upload,
    nodes_upload,
    server_taxdump_dir: str,
) -> str:
    if names_upload is not None and nodes_upload is not None:
        taxdump_dir = os.path.join(temp_root, "taxdump")
        os.makedirs(taxdump_dir, exist_ok=True)
        _save_uploaded_file(names_upload, os.path.join(taxdump_dir, "names.dmp"))
        _save_uploaded_file(nodes_upload, os.path.join(taxdump_dir, "nodes.dmp"))
        return taxdump_dir

    server_taxdump_dir = server_taxdump_dir.strip()
    if server_taxdump_dir:
        names_path = os.path.join(server_taxdump_dir, "names.dmp")
        nodes_path = os.path.join(server_taxdump_dir, "nodes.dmp")
        if not (os.path.exists(names_path) and os.path.exists(nodes_path)):
            raise ValueError("The server-side taxdump directory must contain names.dmp and nodes.dmp.")
        return server_taxdump_dir

    raise ValueError("Upload both names.dmp and nodes.dmp, or provide a server-side taxdump directory.")


def _resolve_acc2taxid_dir(
    temp_root: str,
    accession2taxid_uploads: Iterable[object],
    server_acc2taxid_dir: str,
) -> str:
    uploads = [uploaded for uploaded in accession2taxid_uploads if uploaded is not None]
    if uploads:
        acc2taxid_dir = os.path.join(temp_root, "accession2taxid")
        os.makedirs(acc2taxid_dir, exist_ok=True)
        for uploaded in uploads:
            _save_uploaded_file(uploaded, os.path.join(acc2taxid_dir, uploaded.name))
        return acc2taxid_dir

    server_acc2taxid_dir = server_acc2taxid_dir.strip()
    if server_acc2taxid_dir:
        if not os.path.isdir(server_acc2taxid_dir):
            raise ValueError("The server-side accession2taxid directory does not exist.")
        return server_acc2taxid_dir

    raise ValueError(
        "Upload one or more accession2taxid .gz files, or provide a server-side accession2taxid directory."
    )


def _dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def run_analysis_job(
    tm_tsv_upload,
    fasta_upload,
    names_upload,
    nodes_upload,
    accession2taxid_uploads: Iterable[object],
    server_taxdump_dir: str,
    server_acc2taxid_dir: str,
    requested_outputs: List[str],
    acc_workers: int,
) -> tuple[Dict[str, pd.DataFrame], str]:
    if not requested_outputs:
        raise ValueError("Select at least one output.")

    logs = io.StringIO()
    with contextlib.redirect_stdout(logs), tempfile.TemporaryDirectory() as temp_root:
        tm_input_name = tm_tsv_upload.name or "tm_results.tsv"
        fasta_input_name = fasta_upload.name or "input.fasta"
        tm_input_path = _save_uploaded_file(tm_tsv_upload, os.path.join(temp_root, tm_input_name))
        fasta_path = _save_uploaded_file(fasta_upload, os.path.join(temp_root, fasta_input_name))

        taxdump_dir = _resolve_taxdump_dir(temp_root, names_upload, nodes_upload, server_taxdump_dir)
        acc2taxid_dir = _resolve_acc2taxid_dir(
            temp_root,
            accession2taxid_uploads,
            server_acc2taxid_dir,
        )

        acc_cache_path = os.path.join(temp_root, ".organize_acc_cache.json")

        df_raw = read_tm_results(tm_input_path)
        print(f"[info] Loaded {len(df_raw):,} rows from {tm_input_name}")

        df_best = filter_best_hits(df_raw)
        print(f"[info] Best hits retained: {len(df_best):,} rows (unique per query/accession)")

        fasta_accessions = read_fasta_accessions(fasta_path)
        print(f"[info] Loaded {len(fasta_accessions):,} FASTA headers from {fasta_input_name}")

        accessions_tsv = df_best["contig"].dropna().astype(str).tolist()
        unique_tsv = list(dict.fromkeys(accessions_tsv))
        unique_fasta = list(dict.fromkeys(fasta_accessions))
        all_accessions = list(dict.fromkeys(unique_tsv + unique_fasta))
        print(
            "[info] Resolving "
            f"{len(all_accessions):,} unique accessions "
            f"(TSV {len(unique_tsv):,} + FASTA {len(unique_fasta):,})..."
        )

        acc2tax = map_accessions_to_taxids(
            all_accessions,
            cache_path=acc_cache_path,
            a2t_dir=acc2taxid_dir,
            acc_workers=acc_workers,
        )
        resolved_all = sum(1 for accession in all_accessions if acc2tax.get(accession))
        print(
            f"[info] Accession->TaxID resolved: {resolved_all:,} ok, "
            f"{len(all_accessions) - resolved_all:,} unresolved"
        )

        resolved_fasta = sum(1 for accession in unique_fasta if acc2tax.get(accession))
        print(
            f"[info] FASTA accessions resolved: {resolved_fasta:,} ok, "
            f"{len(unique_fasta) - resolved_fasta:,} unresolved"
        )

        df_best = df_best.copy()
        df_best["TaxID"] = df_best["contig"].map(acc2tax)

        fasta_taxids = [str(acc2tax[accession]) for accession in unique_fasta if acc2tax.get(accession)]
        tm_taxids = [str(taxid) for taxid in df_best["TaxID"].dropna().unique()]
        taxids_all = sorted(set(tm_taxids) | set(fasta_taxids))
        print(f"[info] Loading taxonomy for {len(taxids_all):,} TaxIDs from {taxdump_dir}...")

        taxdump = load_taxdump(taxdump_dir)
        lineage = lineage_from_taxdump(taxids_all, taxdump)
        origin_counts = compute_origin_tax_counts(fasta_taxids, lineage)
        accession_metrics = compute_accession_metrics(df_best)

        results: Dict[str, pd.DataFrame] = {}

        if "accession" in requested_outputs:
            accession_with_taxonomy = attach_taxonomy(accession_metrics, lineage)
            accession_with_taxonomy.sort_values(
                ["phylum", "class", "order", "family", "genus", "TaxID", "AccessionID"],
                inplace=True,
                na_position="last",
            )
            accession_with_taxonomy.reset_index(drop=True, inplace=True)
            results["accession"] = accession_with_taxonomy

        summary_ranks = [rank for rank in ("class", "order", "family", "genus") if rank in requested_outputs]
        if summary_ranks:
            taxid_summary = compute_taxid_summary(accession_metrics, lineage)
            for rank in summary_ranks:
                metrics = summarize_for_tree_map(taxid_summary, rank)
                summary_df = build_rank_summary(taxdump, rank, metrics, origin_counts.get(rank))
                results[rank] = summary_df

        return results, logs.getvalue()


def _render_results() -> None:
    state = st.session_state.get(SESSION_KEY)
    if not state:
        return

    results: Dict[str, pd.DataFrame] = state["results"]
    logs: str = state["logs"]

    st.divider()
    st.subheader("Outputs")
    tabs = st.tabs([OUTPUT_LABELS[name] for name in results])
    for tab, output_name in zip(tabs, results):
        df = results[output_name]
        with tab:
            st.caption(f"{len(df):,} rows")
            st.dataframe(df.head(200), use_container_width=True, height=420)
            if len(df) > 200:
                st.caption("Preview limited to the first 200 rows.")
            st.download_button(
                label=f"Download {OUTPUT_LABELS[output_name]} CSV",
                data=_dataframe_to_csv_bytes(df),
                file_name=OUTPUT_FILE_NAMES[output_name],
                mime="text/csv",
                key=f"{SESSION_KEY}_{output_name}_download",
            )

    with st.expander("Execution log"):
        st.code(logs or "(no log output)")


def render_app(*, standalone: bool = False) -> None:
    if standalone:
        st.set_page_config(page_title="Analyze TM Web", layout="wide")

    st.title("Analyze TM Web")
    st.write(
        "Upload the TmBLAST TSV, FASTA, taxdump files, and accession2taxid files. "
        "The app runs the existing analysis pipeline and returns downloadable CSV outputs."
    )

    with st.form("analyze_tm_form"):
        tm_tsv_upload = st.file_uploader("TmBLAST TSV", type=["tsv", "txt"])
        fasta_upload = st.file_uploader("Origin FASTA", type=["fa", "fasta", "fna", "fas", "txt", "gz"])

        st.markdown("Reference data")
        taxdump_col1, taxdump_col2 = st.columns(2)
        with taxdump_col1:
            names_upload = st.file_uploader("names.dmp", key="analyze_names_dmp")
        with taxdump_col2:
            nodes_upload = st.file_uploader("nodes.dmp", key="analyze_nodes_dmp")

        server_taxdump_dir = st.text_input(
            "Server-side taxdump directory (optional)",
            value="",
            help="Used only when names.dmp and nodes.dmp are not uploaded.",
        )
        accession2taxid_uploads = st.file_uploader(
            "accession2taxid .gz files",
            accept_multiple_files=True,
            type=["gz"],
        )
        server_acc2taxid_dir = st.text_input(
            "Server-side accession2taxid directory (optional)",
            value="",
            help="Used only when accession2taxid files are not uploaded.",
        )

        st.markdown("Outputs")
        output_col1, output_col2, output_col3 = st.columns(3)
        with output_col1:
            include_accession = st.checkbox("Per-accession metrics", value=False)
            include_class = st.checkbox("Class summary", value=True)
        with output_col2:
            include_order = st.checkbox("Order summary", value=True)
            include_family = st.checkbox("Family summary", value=True)
        with output_col3:
            include_genus = st.checkbox("Genus summary", value=False)
            acc_workers = st.number_input(
                "accession2taxid workers",
                min_value=1,
                max_value=16,
                value=1,
                step=1,
            )

        submitted = st.form_submit_button("Run analysis")

    if submitted:
        requested_outputs: List[str] = []
        if include_accession:
            requested_outputs.append("accession")
        if include_class:
            requested_outputs.append("class")
        if include_order:
            requested_outputs.append("order")
        if include_family:
            requested_outputs.append("family")
        if include_genus:
            requested_outputs.append("genus")

        if tm_tsv_upload is None or fasta_upload is None:
            st.error("Upload both the TmBLAST TSV and the FASTA file.")
        else:
            try:
                with st.spinner("Running analyze_tm pipeline..."):
                    results, logs = run_analysis_job(
                        tm_tsv_upload=tm_tsv_upload,
                        fasta_upload=fasta_upload,
                        names_upload=names_upload,
                        nodes_upload=nodes_upload,
                        accession2taxid_uploads=accession2taxid_uploads or [],
                        server_taxdump_dir=server_taxdump_dir,
                        server_acc2taxid_dir=server_acc2taxid_dir,
                        requested_outputs=requested_outputs,
                        acc_workers=int(acc_workers),
                    )
                st.session_state[SESSION_KEY] = {"results": results, "logs": logs}
                st.success("Analysis completed.")
            except Exception as exc:
                st.session_state.pop(SESSION_KEY, None)
                st.error(str(exc))

    _render_results()


if __name__ == "__main__":
    render_app(standalone=True)
