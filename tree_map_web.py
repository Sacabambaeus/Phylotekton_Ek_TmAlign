from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import streamlit as st

TREE_MAP_IMPORT_ERROR: Exception | None = None

try:
    from tree_map import (
        DEPTH_CHOICES,
        RANK_ORDER,
        aggregate_by_depth_assignments,
        aggregate_to_depth,
        apply_taxon_depths,
        build_tree,
        draw_tree,
        fill_missing_ranks,
        filter_by_flags,
        filter_by_taxon_names,
        load_taxdump,
        normalize_taxon_depths,
        normalize_taxon_filters,
        read_input,
    )
except Exception as exc:  # pragma: no cover - only used when deployment is missing runtime deps
    TREE_MAP_IMPORT_ERROR = exc
    DEPTH_CHOICES = ["class", "order", "family", "genus"]
    RANK_ORDER = []


SESSION_KEY = "tree_map_web_state"
KINGDOM_FLAG_OPTIONS = {
    "Animalia / Metazoa": "a",
    "Plantae": "p",
    "Fungi": "f",
    "Protista / Chromista": "r",
    "Monera (Bacteria / Archaea)": "m",
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


def run_tree_map_job(
    summary_csv_upload,
    names_upload,
    nodes_upload,
    server_taxdump_dir: str,
    selected_kingdom_labels: Iterable[str],
    taxon_filter_text: str,
    taxon_depth_text: str,
    depth: str,
    output_format: str,
    show_branch_labels: bool,
) -> Dict[str, object]:
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs), tempfile.TemporaryDirectory() as temp_root:
        input_name = summary_csv_upload.name or "tree_map_input.csv"
        input_path = _save_uploaded_file(summary_csv_upload, os.path.join(temp_root, input_name))
        taxdump_dir = _resolve_taxdump_dir(temp_root, names_upload, nodes_upload, server_taxdump_dir)
        output_path = os.path.join(temp_root, f"tree_map_output.{output_format}")

        df = read_input(input_path)
        name_to_taxids, taxid_to_parent_rank, taxid_to_name = load_taxdump(taxdump_dir)
        df_enriched = fill_missing_ranks(df, name_to_taxids, taxid_to_parent_rank, taxid_to_name)

        flags = [KINGDOM_FLAG_OPTIONS[label] for label in selected_kingdom_labels]
        df_filtered = filter_by_flags(
            df_enriched,
            RANK_ORDER,
            flags,
            name_to_taxids,
            taxid_to_parent_rank,
            taxid_to_name,
        )

        normalized_taxon_filters = normalize_taxon_filters(
            [taxon_filter_text.replace("\n", ",")] if taxon_filter_text.strip() else []
        )
        df_filtered = filter_by_taxon_names(
            df_filtered,
            normalized_taxon_filters,
            name_to_taxids,
            taxid_to_parent_rank,
            taxid_to_name,
        )

        taxon_depth_specs = normalize_taxon_depths([taxon_depth_text] if taxon_depth_text.strip() else [])
        depth_limited_df, depth_series = apply_taxon_depths(
            df_filtered,
            taxon_depth_specs,
            name_to_taxids,
            taxid_to_parent_rank,
            taxid_to_name,
        )

        if depth_series is not None:
            depth_limited_df = aggregate_by_depth_assignments(depth_limited_df, depth_series)
        elif depth:
            depth_limited_df = aggregate_to_depth(depth_limited_df, depth)

        ranks_in_use = [
            rank
            for rank in RANK_ORDER
            if rank in depth_limited_df.columns and not depth_limited_df[rank].isna().all()
        ]
        if depth:
            allowed_ranks: List[str] = []
            for rank in RANK_ORDER:
                allowed_ranks.append(rank)
                if rank == depth:
                    break
            ranks_in_use = [rank for rank in ranks_in_use if rank in allowed_ranks]

        if not ranks_in_use:
            raise ValueError("No taxonomy columns remain after filtering. Check the selected filters.")

        root = build_tree(
            depth_limited_df,
            ranks_in_use,
            name_to_taxids,
            taxid_to_parent_rank,
            taxid_to_name,
        )
        draw_tree(
            root,
            ranks_in_use,
            taxid_to_name,
            output_path,
            show_class_labels=show_branch_labels or bool(normalized_taxon_filters),
            exclude_label_ranks={depth} if depth else set(),
        )

        with open(output_path, "rb") as handle:
            output_bytes = handle.read()

        output_name = f"{Path(input_name).stem}_web.{output_format}"
        return {
            "output_bytes": output_bytes,
            "output_name": output_name,
            "output_format": output_format,
            "preview_df": depth_limited_df.reset_index(drop=True),
            "ranks_in_use": ranks_in_use,
            "logs": logs.getvalue(),
        }


def _render_results() -> None:
    state = st.session_state.get(SESSION_KEY)
    if not state:
        return

    st.divider()
    st.subheader("Output")
    if state["output_format"] == "png":
        st.image(state["output_bytes"], use_container_width=True)
    else:
        st.info("PDF preview is not rendered in Streamlit. Use the download button below.")

    st.download_button(
        label=f"Download {state['output_name']}",
        data=state["output_bytes"],
        file_name=state["output_name"],
        mime="application/octet-stream",
        key=f"{SESSION_KEY}_download",
    )

    preview_df: pd.DataFrame = state["preview_df"]
    st.subheader("Processed data")
    st.caption(
        f"{len(preview_df):,} rows after filtering and aggregation. "
        f"Ranks in use: {', '.join(state['ranks_in_use'])}"
    )
    st.dataframe(preview_df.head(200), use_container_width=True, height=420)
    if len(preview_df) > 200:
        st.caption("Preview limited to the first 200 rows.")

    with st.expander("Execution log"):
        st.code(state["logs"] or "(no log output)")


def render_app(*, standalone: bool = False) -> None:
    if standalone:
        st.set_page_config(page_title="Tree Map Web", layout="wide")

    st.title("Tree Map Web")
    if TREE_MAP_IMPORT_ERROR is not None:
        st.error(
            "tree_map.py could not be imported. On Streamlit Community Cloud this usually means "
            "a runtime dependency such as matplotlib was not installed in the deployed environment."
        )
        st.code(repr(TREE_MAP_IMPORT_ERROR))
        st.info(
            "Confirm that `requirements.txt` is present at the repository root, includes `matplotlib`, "
            "then trigger a full redeploy or reboot of the app."
        )
        return

    st.write(
        "Upload a summary CSV and taxdump files to render the taxonomy tree in PNG or PDF format."
    )

    with st.form("tree_map_form"):
        summary_csv_upload = st.file_uploader("Summary CSV", type=["csv"])

        st.markdown("Reference data")
        taxdump_col1, taxdump_col2 = st.columns(2)
        with taxdump_col1:
            names_upload = st.file_uploader("names.dmp", key="tree_names_dmp")
        with taxdump_col2:
            nodes_upload = st.file_uploader("nodes.dmp", key="tree_nodes_dmp")

        server_taxdump_dir = st.text_input(
            "Server-side taxdump directory (optional)",
            value="",
            help="Used only when names.dmp and nodes.dmp are not uploaded.",
        )

        st.markdown("Filters")
        selected_kingdom_labels = st.multiselect(
            "Kingdom filters",
            options=list(KINGDOM_FLAG_OPTIONS.keys()),
            default=[],
        )
        taxon_filter_text = st.text_area(
            "Taxon filters (comma or newline separated)",
            value="",
            placeholder="Chordata, Arthropoda",
        )
        taxon_depth_text = st.text_area(
            "Taxon depth overrides",
            value="",
            placeholder="Chordata/order\nArthropoda/class",
            help="Use taxon/depth pairs. Example: Chordata/order",
        )

        options_col1, options_col2, options_col3 = st.columns(3)
        with options_col1:
            depth = st.selectbox(
                "Aggregate depth",
                options=[""] + DEPTH_CHOICES,
                format_func=lambda value: value or "None",
            )
        with options_col2:
            output_format = st.selectbox("Output format", options=["png", "pdf"])
        with options_col3:
            show_branch_labels = st.checkbox("Show internal branch labels", value=False)

        submitted = st.form_submit_button("Generate tree map")

    if submitted:
        if summary_csv_upload is None:
            st.error("Upload a summary CSV file.")
        else:
            try:
                with st.spinner("Generating tree map..."):
                    result = run_tree_map_job(
                        summary_csv_upload=summary_csv_upload,
                        names_upload=names_upload,
                        nodes_upload=nodes_upload,
                        server_taxdump_dir=server_taxdump_dir,
                        selected_kingdom_labels=selected_kingdom_labels,
                        taxon_filter_text=taxon_filter_text,
                        taxon_depth_text=taxon_depth_text,
                        depth=depth,
                        output_format=output_format,
                        show_branch_labels=show_branch_labels,
                    )
                st.session_state[SESSION_KEY] = result
                st.success("Tree map generated.")
            except Exception as exc:
                st.session_state.pop(SESSION_KEY, None)
                st.error(str(exc))

    _render_results()


if __name__ == "__main__":
    render_app(standalone=True)
