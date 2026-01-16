#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指定したCSVを系統樹上にマッピングしてPNGを出力するスクリプト。
例: python3 mapping_tree8.py --a taxonomy_class_summary3.csv out.png

入力: phylum, class, order, family, genus, taxid_count, mean_accession_count,
      mean_q1_Tm, mean_q1_identity, mean_q2_Tm, mean_q2_identity, origin_tax_count
出力: 系統樹 + 円/ひし形/長方形をマッピングしたPNG

フィルター:
    --a : Animalia(Metazoa) のみ
    --p : Plantae のみ
    --f : Fungi のみ
    --r : Protista/Chromista など (動物・植物・菌以外の真核生物)
    --m : Monera (Bacteria/Archaea) のみ
    --taxon <name>[,<name>...] : 指定した分類群名(phylum/class/order)のみを対象にする（複数回指定可）
    --depth <class|order|family> : どの分類階級まで描画するかを指定する
    何も指定しない場合は全て含める。

taxdump(names.dmp, nodes.dmp) を用いて名前→taxid解決および界判定を行う。
Accession2taxid/taxdump 以外には依存しない。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# --- 追加部分 ここから ---
import matplotlib
matplotlib.use('Agg')  # 画面表示せず、ファイル保存に特化させるおまじない
# --- 追加部分 ここまで ---

import matplotlib.pyplot as plt
import pandas as pd


# NCBI taxdump(nodes.dmp) に含まれる rank うち、描画対象とする階級順（上位→下位）。
# ※ draw_tree/assign_coordinates 側の「root直下は phylum」を前提とした描画を崩さないため、
#   domain/kingdom/superphylum 等の phylum より上位はあえて含めない。
# ※ "no rank" / "clade" は系統上で複数回現れ得るため、map_tree12.17 では taxdump の lineage を辿って
#   重複も含めて描画する（列として一意に扱う前提を捨てる）。
RANK_ORDER = [
    "phylum",
    "subphylum",
    "clade",
    "no rank",
    "superclass",
    "class",
    "subclass",
    "infraclass",
    "cohort",
    "subcohort",
    "superorder",
    "order",
    "suborder",
    "infraorder",
    "parvorder",
    "superfamily",
    "family",
    "subfamily",
    "tribe",
    "subtribe",
    "genus",
    "subgenus",
    "section",
    "subsection",
    "series",
    "species group",
    "species subgroup",
    "species",
    "subspecies",
    "varietas",
    "subvariety",
    "forma",
    "forma specialis",
    "strain",
    "isolate",
    "serogroup",
    "serotype",
    "pathogroup",
    "biotype",
    "genotype",
    "morph",
]

# フィルター用キーワード（全て小文字）
KINGDOM_KEYWORDS = {
    "a": {"animalia", "metazoa"},
    "p": {"plantae", "viridiplantae", "streptophyta", "chloroplastida", "archaeplastida"},
    "f": {"fungi"},
    "r": {
        "protista",
        "protozoa",
        "chromista",
        "stramenopiles",
        "alveolata",
        "rhizaria",
        "amoebozoa",
        "haptophyta",
        "cryptophyceae",
    },
    "m": {"monera", "bacteria", "eubacteria", "archaea"},
}


@dataclass
class TreeNode:
    name: str
    rank: str
    taxid: Optional[int] = None
    parent: Optional["TreeNode"] = None
    depth: int = 0
    children: Dict[str, "TreeNode"] = field(default_factory=dict)
    data: Optional[dict] = None
    x: float = 0.0
    y: float = 0.0
    is_zero: bool = False
    has_zero_subtree: bool = False
    display_name: Optional[str] = None

    def get_or_add_child(self, name: str, rank: str, taxid: Optional[int]) -> "TreeNode":
        if name not in self.children:
            self.children[name] = TreeNode(
                name=name,
                rank=rank,
                taxid=taxid,
                parent=self,
                depth=self.depth + 1,
            )
        return self.children[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSVの内容を系統樹上にマッピングして画像を作成する")
    parser.add_argument("input_csv", help="入力CSV (phylum, class, ...)")
    parser.add_argument("output_png", help="出力ファイル名(png/pdf)")
    parser.add_argument(
        "--taxdump",
        default="./taxdump",
        help="taxdumpディレクトリ(names.dmp, nodes.dmpを含む)",
    )
    parser.add_argument(
        "--a",
        dest="animalia",
        action="store_true",
        help="Animalia(Metazoa)のみを対象にする",
    )
    parser.add_argument("--p", dest="plantae", action="store_true", help="Plantaeのみを対象にする")
    parser.add_argument("--f", dest="fungi", action="store_true", help="Fungiのみを対象にする")
    parser.add_argument("--r", dest="protista", action="store_true", help="Protistaのみを対象にする")
    parser.add_argument("--m", dest="monera", action="store_true", help="Moneraのみを対象にする")
    parser.add_argument(
        "--taxon",
        "--t",
        dest="taxon_names",
        action="append",
        nargs="+",
        help="特定の分類群名(phylum/class/order)のみを対象にする（複数指定可、カンマ/空白区切り可）",
    )
    parser.add_argument(
        "--depth",
        "--d",
        dest="depth",
        choices=["class", "order", "family"],
        help="どの分類階級まで描画するかを指定する",
    )
    return parser.parse_args()


def load_taxdump(taxdump_dir: str) -> Tuple[Dict[str, List[int]], Dict[int, Tuple[int, str]], Dict[int, str]]:
    names_path = os.path.join(taxdump_dir, "names.dmp")
    nodes_path = os.path.join(taxdump_dir, "nodes.dmp")
    if not os.path.exists(names_path) or not os.path.exists(nodes_path):
        raise FileNotFoundError(f"taxdump内にnames.dmp/nodes.dmpが見つかりません: {taxdump_dir}")

    name_to_taxids: Dict[str, List[int]] = {}
    taxid_to_name: Dict[int, str] = {}
    with open(names_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 4:
                continue
            taxid_txt = row[0].strip()
            name_txt = row[1].strip()
            name_class = row[3].strip()
            if not taxid_txt or not name_txt:
                continue
            if name_class == "scientific name":
                tid = int(taxid_txt)
                lowered = name_txt.lower()
                name_to_taxids.setdefault(lowered, []).append(tid)
                if tid not in taxid_to_name:
                    taxid_to_name[tid] = name_txt

    taxid_to_parent_rank: Dict[int, Tuple[int, str]] = {}
    with open(nodes_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            taxid_txt = row[0].strip()
            parent_txt = row[1].strip()
            rank_txt = row[2].strip()
            if not taxid_txt or not parent_txt:
                continue
            taxid_to_parent_rank[int(taxid_txt)] = (int(parent_txt), rank_txt)

    return name_to_taxids, taxid_to_parent_rank, taxid_to_name


def choose_taxid(
    name: str,
    rank: str,
    name_to_taxids: Dict[str, List[int]],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
) -> Optional[int]:
    candidates = name_to_taxids.get(name.lower(), [])
    if not candidates:
        return None
    rank_matches = [tid for tid in candidates if taxid_to_parent_rank.get(tid, (None, None))[1] == rank]
    if rank_matches:
        return rank_matches[0]
    return candidates[0]


def read_input(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for rank in RANK_ORDER:
        if rank not in df.columns:
            df[rank] = pd.NA

    # 空文字は欠損扱いに統一する
    for rank in RANK_ORDER:
        df[rank] = df[rank].replace("", pd.NA)

    numeric_cols = [
        "taxid_count",
        "mean_accession_count",
        "mean_q1_Tm",
        "mean_q1_identity",
        "mean_q2_Tm",
        "mean_q2_identity",
        "origin_tax_count",
    ]
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        if missing == ["origin_tax_count"]:
            df["origin_tax_count"] = pd.NA
        else:
            raise ValueError(f"必要な列が不足しています: {', '.join(missing)}")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def detect_kingdom(
    taxid: Optional[int],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> Optional[str]:
    if taxid is None:
        return None
    names_in_lineage: List[str] = []
    current = taxid
    visited = set()
    while current in taxid_to_parent_rank and current not in visited:
        visited.add(current)
        sci_name = taxid_to_name.get(current, "")
        if sci_name:
            names_in_lineage.append(sci_name.lower())
        parent, _ = taxid_to_parent_rank[current]
        if parent == current:
            break
        current = parent

    for key, keywords in KINGDOM_KEYWORDS.items():
        if any(name in keywords for name in names_in_lineage):
            return key
    if any(name == "eukaryota" for name in names_in_lineage):
        return "r"
    return None


def determine_row_taxid(
    row: pd.Series,
    ranks_in_use: Iterable[str],
    name_to_taxids: Dict[str, List[int]],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
) -> Optional[int]:
    for rank in reversed(list(ranks_in_use)):
        val = row.get(rank)
        if pd.isna(val):
            continue
        name = str(val).strip()
        if not name:
            continue
        tid = choose_taxid(name, rank, name_to_taxids, taxid_to_parent_rank)
        if tid is not None:
            return tid
    return None


def get_lineage_ranks(
    taxid: Optional[int],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> Dict[str, str]:
    """
    taxidから遡ってphylum/class/order/family/genusを取得する。
    """
    if taxid is None:
        return {}
    lineage: Dict[str, str] = {}
    visited = set()
    current = taxid
    while current in taxid_to_parent_rank and current not in visited:
        visited.add(current)
        parent, rank = taxid_to_parent_rank[current]
        if rank in RANK_ORDER:
            name = taxid_to_name.get(current)
            if name:
                lineage.setdefault(rank, name)
        if parent == current:
            break
        current = parent
    return lineage


def is_row_lineage_consistent(
    row: pd.Series,
    taxid: int,
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> bool:
    lineage = get_lineage_ranks(taxid, taxid_to_parent_rank, taxid_to_name)
    for rank in RANK_ORDER:
        if rank not in row:
            continue
        val = row.get(rank)
        if pd.isna(val):
            continue
        name = str(val).strip()
        if not name or name.lower() == "unknown":
            continue
        lineage_name = lineage.get(rank)
        if not lineage_name:
            return False
        if lineage_name.lower() != name.lower():
            return False
    return True


def get_lineage_path(
    taxid: Optional[int],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
    allowed_ranks: Optional[set[str]] = None,
) -> List[Tuple[int, str, str]]:
    """
    taxid から root 方向へ遡り、rank の重複も保持したまま lineage を返す（root→taxid の順）。

    - clade/no rank が複数回出現しても全て含める
    - allowed_ranks を指定した場合、その rank のみを残す
    """
    if taxid is None:
        return []

    lineage_rev: List[Tuple[int, str, str]] = []
    visited = set()
    current = taxid
    while current in taxid_to_parent_rank and current not in visited:
        visited.add(current)
        parent, rank = taxid_to_parent_rank[current]
        if allowed_ranks is None or rank in allowed_ranks:
            name = taxid_to_name.get(current) or str(current)
            lineage_rev.append((current, rank, name))
        if parent == current:
            break
        current = parent

    lineage_rev.reverse()
    return lineage_rev


def trim_lineage_to_phylum(lineage: List[Tuple[int, str, str]]) -> List[Tuple[int, str, str]]:
    """
    root直下を phylum に揃えるため、最初に出現する phylum より上位を捨てる。
    （no rank/clade が phylum より上位に出るケースを抑制する）
    """
    for idx, (_, rank, _) in enumerate(lineage):
        if rank == "phylum":
            return lineage[idx:]
    return lineage


def fill_missing_ranks(
    df: pd.DataFrame,
    name_to_taxids: Dict[str, List[int]],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> pd.DataFrame:
    """
    入力に空欄の分類階級があっても、taxdumpの系統から補完する。
    """
    filled = df.copy()
    for idx, row in filled.iterrows():
        tid = determine_row_taxid(row, RANK_ORDER, name_to_taxids, taxid_to_parent_rank)
        if tid is None:
            continue
        lineage = get_lineage_ranks(tid, taxid_to_parent_rank, taxid_to_name)
        for rank, name in lineage.items():
            current_val = row.get(rank)
            if pd.isna(current_val) or (isinstance(current_val, str) and not current_val.strip()):
                filled.at[idx, rank] = name
    return filled


def normalize_taxon_filters(raw_list: Optional[List[object]]) -> List[str]:
    if not raw_list:
        return []
    normalized: List[str] = []
    for raw in raw_list:
        if isinstance(raw, (list, tuple)):
            items = raw
        else:
            items = [raw]
        for item in items:
            for part in str(item).split(","):
                name = part.strip()
                if name:
                    normalized.append(name.lower())
    return normalized


def filter_by_flags(
    df: pd.DataFrame,
    ranks_for_taxid: List[str],
    flags: List[str],
    name_to_taxids: Dict[str, List[int]],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> pd.DataFrame:
    if not flags:
        return df
    target_set = set(flags)
    kept_rows = []
    skipped_unknown = 0
    skipped_mismatch = 0

    for _, row in df.iterrows():
        tid = determine_row_taxid(row, ranks_for_taxid, name_to_taxids, taxid_to_parent_rank)
        group = detect_kingdom(tid, taxid_to_parent_rank, taxid_to_name)
        if group is None:
            skipped_unknown += 1
            continue
        if group in target_set:
            kept_rows.append(row)
        else:
            skipped_mismatch += 1

    if not kept_rows:
        raise ValueError("フィルター条件に合致する行がありません。taxdumpと入力を確認してください。")

    if skipped_unknown:
        print(f"[warn] 界を判定できず除外した行: {skipped_unknown}")
    if skipped_mismatch:
        print(f"[info] フィルター条件と合わず除外した行: {skipped_mismatch}")
    return pd.DataFrame(kept_rows)


def filter_by_taxon_names(
    df: pd.DataFrame,
    taxon_filters: List[str],
    name_to_taxids: Dict[str, List[int]],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> pd.DataFrame:
    """
    --taxon (--t) で指定された分類群名に対応する TaxID を特定し、
    その子孫（descendants）に含まれる行のみを抽出する。
    """
    if not taxon_filters:
        return df

    # 1. 指定された名前(文字列)を TaxID のセットに変換する
    target_taxids = set()
    for name in taxon_filters:
        # main側で小文字化(normalize)されている前提
        tids = name_to_taxids.get(name, [])
        if not tids:
            print(f"[warn] 指定された分類群名 '{name}' が names.dmp に見つかりません。")
            continue
        target_taxids.update(tids)

    if not target_taxids:
        # 指定した名前がすべて見つからなかった場合
        raise ValueError("指定された分類群名が taxdump 内に一つも見つかりませんでした。")

    kept_rows = []
    skipped_unknown = 0
    skipped_mismatch = 0
    skipped_inconsistent = 0
    
    # 高速化のためのメモ化 (TaxID -> Targetの子孫かどうか(bool))
    memo_verdict: Dict[int, bool] = {}

    for _, row in df.iterrows():
        # 行の TaxID を特定 (CSVのランク列などから解決)
        tid = determine_row_taxid(row, RANK_ORDER, name_to_taxids, taxid_to_parent_rank)
        
        if tid is None:
            skipped_unknown += 1
            continue
        if not is_row_lineage_consistent(row, tid, taxid_to_parent_rank, taxid_to_name):
            skipped_inconsistent += 1
            continue

        # 親を遡って target_taxids に含まれるかチェック
        current = tid
        match = False
        path = [] # 判定待ちの経路
        
        while current is not None:
            # すでに判定済みのIDなら結果を利用
            if current in memo_verdict:
                match = memo_verdict[current]
                break
            
            # ターゲット（指定された分類群）に到達したか
            if current in target_taxids:
                match = True
                break
            
            path.append(current)
            
            # 親へ移動
            if current not in taxid_to_parent_rank:
                 break
            parent, _ = taxid_to_parent_rank[current]
            if parent == current: # root到達
                break
            current = parent
        
        # 結果をキャッシュに保存 & 行の採用判定
        if match:
             # ヒットした場合、そこに至る経路(path)のIDはすべて「ターゲットの子孫」なのでTrue
             for p in path:
                 memo_verdict[p] = True
             kept_rows.append(row)
        else:
             # Rootまで辿ってダメだった場合、経路上のIDはすべてFalse
             for p in path:
                 memo_verdict[p] = False
             skipped_mismatch += 1

    if not kept_rows:
        raise ValueError("指定された分類群(--taxon)に含まれる行がCSVにありませんでした。")

    if skipped_unknown:
        print(f"[warn] 行のTaxIDを特定できず除外: {skipped_unknown} 件")
    if skipped_inconsistent:
        print(f"[warn] CSVの階級とTaxIDが矛盾するため除外: {skipped_inconsistent} 件")
    if skipped_mismatch:
        print(f"[info] フィルター条件外のため除外: {skipped_mismatch} 件")

    return pd.DataFrame(kept_rows)

def build_tree(
    df: pd.DataFrame,
    ranks_in_use: List[str],
    name_to_taxids: Dict[str, List[int]],
    taxid_to_parent_rank: Dict[int, Tuple[int, str]],
    taxid_to_name: Dict[int, str],
) -> TreeNode:
    root = TreeNode(name="root", rank="root", taxid=1, parent=None, depth=0)
    allowed_ranks = set(ranks_in_use)
    allowed_ranks.update({"clade", "no rank"})
    lineage_cache: Dict[int, List[Tuple[int, str, str]]] = {}

    for _, row in df.iterrows():
        tid = determine_row_taxid(row, ranks_in_use, name_to_taxids, taxid_to_parent_rank)
        if tid is None:
            continue

        lineage = lineage_cache.get(tid)
        if lineage is None:
            lineage = get_lineage_path(tid, taxid_to_parent_rank, taxid_to_name, allowed_ranks=allowed_ranks)
            lineage = trim_lineage_to_phylum(lineage)
            lineage_cache[tid] = lineage

        node = root
        leaf_node: Optional[TreeNode] = None
        for node_taxid, node_rank, node_name in lineage:
            node = node.get_or_add_child(node_name, node_rank, node_taxid)
            leaf_node = node
        if leaf_node is not None:
            leaf_node.data = row.to_dict()
    return root


def collect_leaves(node: TreeNode) -> List[TreeNode]:
    leaves: List[TreeNode] = []
    if not node.children:
        leaves.append(node)
    else:
        for child in sorted(node.children.values(), key=lambda c: c.name):
            leaves.extend(collect_leaves(child))
    return leaves


def row_has_zero(row: dict) -> bool:
    origin_val = row.get("origin_tax_count")
    try:
        origin_num = float(origin_val)
    except (TypeError, ValueError):
        return True
    if math.isnan(origin_num) or math.isclose(origin_num, 0.0):
        return True
    return False


def mark_has_zero(node: TreeNode) -> bool:
    node.is_zero = node.data is None or (node.data is not None and row_has_zero(node.data))
    child_zero = False
    for child in node.children.values():
        if mark_has_zero(child):
            child_zero = True
    node.has_zero_subtree = node.is_zero or child_zero
    return node.has_zero_subtree


def assign_display_names(node: TreeNode, taxid_to_name: Dict[int, str]) -> None:
    if node.taxid and node.taxid in taxid_to_name:
        node.display_name = taxid_to_name[node.taxid]
    else:
        node.display_name = node.name
    for child in node.children.values():
        assign_display_names(child, taxid_to_name)


def extract_label(row: dict, ranks_in_use: List[str]) -> str:
    for rank in reversed(ranks_in_use):
        val = row.get(rank)
        if pd.isna(val) or (isinstance(val, str) and not str(val).strip()):
            continue
        return str(val).strip()
    return ""


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def scale_value(value: float, lower: float, upper: float, size_min: float, size_max: float) -> float:
    if math.isnan(value):
        return size_min
    clipped = clamp(value, lower, upper)
    if math.isclose(upper, lower):
        return (size_min + size_max) / 2
    ratio = (clipped - lower) / (upper - lower)
    return size_min + ratio * (size_max - size_min)


def assign_coordinates(root: TreeNode, y_step: float) -> None:
    # 1. leaves の定義
    leaves = collect_leaves(root)
    if not leaves:
        return

    # --- ▼ 設定：ここで長さを調整します ▼ ---
    leaf_step = 24.0          # 末端（葉）への枝の長さ
    intermediate_step = 40.0  # 中間ノードへの枝の長さ（ここを大きくするとグイッと伸びます）
    # ------------------------------------

    base_x = 0.4 + 2.0
    
    # root_x の計算（初期位置）
    # 元のコードのロジックを流用しますが、係数は leaf_step を基準にします
    if len(leaves) < 100:
        root_x = base_x - leaf_step * 0.05 * len(leaves)
    else:
        root_x = base_x - leaf_step

    phylum_children = sorted(root.children.values(), key=lambda c: c.name)
    gap_mult = 12.0
    total_gaps = max(0, len(phylum_children) - 1) * gap_mult
    total_rows = len(leaves) + total_gaps
    start_y = (total_rows - 1) * y_step

    # ▼ 変更点: current_x 引数を追加しました
    def _assign(node: TreeNode, y_cursor: float, current_x: float) -> float:
        # 親から受け取った座標を自分のX座標にする
        node.x = current_x
        
        # 葉ノード（末端）の場合
        if not node.children:
            node.y = y_cursor
            return y_cursor - y_step
        
        # --- ループ処理の共通化関数 ---
        def process_children(children_list, current_y):
            child_ys = []
            for idx, child in enumerate(children_list):
                
                # ★ここで分岐: 子がさらに子供を持っている(=中間)なら長く、そうでなければ短く
                if child.children:
                    step = intermediate_step
                else:
                    step = leaf_step
                
                # 再帰呼び出し: 「今のX + 決めたステップ」を渡す
                current_y = _assign(child, current_y, current_x + step)
                
                child_ys.append(child.y)
                
                # Root直下のPhylum間の隙間処理
                if node.rank == "root" and idx < len(children_list) - 1:
                    current_y -= y_step * gap_mult
            return current_y, child_ys

        # --- Root と それ以外で処理対象リストを変える ---
        if node.rank == "root":
            targets = phylum_children
        else:
            targets = sorted(node.children.values(), key=lambda c: c.name)
            
        y_cursor, child_ys = process_children(targets, y_cursor)
        
        if child_ys:
            node.y = sum(child_ys) / len(child_ys)
            
        return y_cursor

    # 初回呼び出し: root_x を渡してスタート
    _assign(root, start_y, root_x)

def collect_phylum_ranges(root: TreeNode) -> List[Tuple[str, float, float]]:
    ranges: List[Tuple[str, float, float]] = []
    for child in sorted(root.children.values(), key=lambda c: c.y, reverse=True):
        if child.rank != "phylum":
            continue
        ys = [leaf.y for leaf in collect_leaves(child)]
        if not ys:
            continue
        ranges.append((child.name, min(ys), max(ys)))
    return ranges


def draw_diamond(ax, center_x: float, center_y: float, diag: float, color: str, edge: str) -> None:
    half = diag / 2.0
    verts = [
        (center_x, center_y + half),
        (center_x + half, center_y),
        (center_x, center_y - half),
        (center_x - half, center_y),
    ]
    diamond = plt.Polygon(verts, closed=True, facecolor=color, edgecolor=edge, lw=0.8, alpha=0.9)
    ax.add_patch(diamond)


def shrink_x(node: TreeNode, factor: float) -> None:
    node.x *= factor
    for child in node.children.values():
        shrink_x(child, factor)


def draw_tree(
    root: TreeNode,
    ranks_in_use: List[str],
    taxid_to_name: Dict[int, str],
    output_png: str,
    show_class_labels: bool = False,
    exclude_label_ranks: Optional[Iterable[str]] = None,
) -> None:
    def parse_numeric(value: object) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(num):
            return None
        return num

    leaves = collect_leaves(root)
    if not leaves:
        raise ValueError("葉ノードが存在しません。入力を確認してください。")

    mark_has_zero(root)
    assign_display_names(root, taxid_to_name)

    leaf_count = len(leaves)
    if leaf_count <= 30:
        y_step = 80.0
    elif leaf_count <= 80:
        y_step = 120.0
    else:
        y_step = 160.0

    assign_coordinates(root, y_step=y_step)
    shrink_x(root, 1.0)

    size_max = min(y_step * 0.5, 12.0)
    size_min = max(size_max * 0.1, 2.0)
    bar_height = clamp(y_step * 0.4, 10.0, 80.0)
    branch_thick = clamp(y_step * 0.84, 6.0, 21.0)
    shape_spacing = max(size_max * 1.0 + 1.0, 5.0) * 9.0

    tm_range = (30.0, 65.0)
    id_range = (50.0, 100.0)

    tree_end_x = max((leaf.x for leaf in leaves), default=2.0)
    red_cx = tree_end_x * 1.2
    blue_cx = red_cx + shape_spacing
    red_dx = blue_cx + shape_spacing
    blue_dx = red_dx + shape_spacing

    label_anchor_x = blue_dx + shape_spacing + 1.5
    label_bar_gap = 2.0
    base_bar_len = max(120.0, size_max * 20.0)
    #tax_count barの位置
    bar_shift = 200.0
    bar_start_x = label_anchor_x + label_bar_gap + bar_shift

    taxid_counts: List[float] = []
    origin_counts: List[float] = []
    for l in leaves:
        if not l.data:
            continue
        tax_num = parse_numeric(l.data.get("taxid_count"))
        if tax_num is not None:
            taxid_counts.append(tax_num)
        origin_num = parse_numeric(l.data.get("origin_tax_count"))
        if origin_num is not None:
            origin_counts.append(origin_num)

    max_tax_count = None
    if taxid_counts or origin_counts:
        max_tax_count = max(taxid_counts + origin_counts)

    max_bar_len = base_bar_len
    if max_tax_count is not None:
        max_bar_len = base_bar_len * max(1.0, math.log10(max_tax_count + 1.0))

    if max_tax_count is not None:
        tax_scale = (0.0, max_tax_count) if taxid_counts else None
        origin_scale = (0.0, max_tax_count) if origin_counts else None
    else:
        tax_scale = None
        origin_scale = None

    # 【設定】 0.5 = 平方根(ルート)。 1.0 = そのまま(リニア)。
    # 数値を大きくするほど(0.6, 0.7...)、差が強調されて棒の長さの差が激しくなります。
    # 対数より差をつけたい場合は 0.5 前後がおすすめです。
    power_factor = 0.7

    def calc_bar_len(val_num: float, scale: Tuple[float, float]) -> float:
        min_val, max_val = scale
        val_calc = max(val_num, 0.0) ** power_factor
        min_calc = max(min_val, 0.0) ** power_factor
        max_calc = max(max_val, 0.0) ** power_factor

        if math.isclose(max_calc, min_calc):
            percent = 1.0
        else:
            percent = 0.01 + (val_calc - min_calc) / (max_calc - min_calc) * 0.99

        percent = clamp(percent, 0.01, 1.0)
        return max_bar_len * percent

    max_y = max(leaf.y for leaf in leaves)
    min_y = min(leaf.y for leaf in leaves)
    # 葉っぱの枚数に応じた高さ + 上下の余白(4.0)
    # clamp ではなく max だけを使って、最小値だけ保証する
    raw_height = (max_y - min_y + 2 * y_step) * 0.06
    height = max(raw_height, 80.0)
    
    width = clamp(float(bar_start_x + max_bar_len + 4.0), 40.0, 160.0)
    # ---------------------------------------------------------
    # ピクセル数制限 (2^16 = 65536) を回避するためのDPI自動調整
    # ---------------------------------------------------------
    LIMIT_PIXELS = 60000  # 余裕を持って65536より少し小さく設定
    
    # まず希望のDPIを設定（最初は300など高めでもOK）
    target_dpi = 300
    
    # 高さが制限を超えそうなら、DPIを下げる
    if height * target_dpi > LIMIT_PIXELS:
        target_dpi = max(30, int(LIMIT_PIXELS / height))
        print(f"[info] 画像が巨大すぎるため、DPIを {target_dpi} に下げて出力します。")

    # 横幅が制限を超えそうなら、さらにDPIを下げる
    if width * target_dpi > LIMIT_PIXELS:
        target_dpi = max(30, int(LIMIT_PIXELS / width))
        print(f"[info] 横幅が巨大すぎるため、DPIを {target_dpi} に下げて出力します。")

    # DPIが低くなりすぎたら警告（文字が潰れる可能性があります）
    if target_dpi < 50:
        print("[warn] DPIが非常に低くなっています。PDF出力をお勧めします。")

    # ---------------------------------------------------------

    # ★修正1: ここで target_dpi を渡すことで、キャンバス作成時点から低解像度モードにする
    fig, ax = plt.subplots(figsize=(width, height), dpi=target_dpi)

    def draw_edges(node: TreeNode) -> None:
        for child in node.children.values():
            ax.plot([node.x, node.x], [node.y, child.y], color="#696969", lw=branch_thick, solid_capstyle="round")
            ax.plot([node.x, child.x], [child.y, child.y], color="#696969", lw=branch_thick, solid_capstyle="round")
            draw_edges(child)

    draw_edges(root)

    phylum_ranges = collect_phylum_ranges(root)

    for i in range(len(phylum_ranges) - 1):
        _, upper_min, _ = phylum_ranges[i]
        _, _, lower_max = phylum_ranges[i + 1]
        sep_y = (upper_min + lower_max) / 2.0
        ax.plot(
            [root.x - 0.4, bar_start_x + max_bar_len + 0.6],
            [sep_y, sep_y],
            linestyle="--",
            color="#888888",
            lw=1.2,
            dashes=(4, 4),
        )

    for leaf in leaves:
        y = leaf.y
        text_color = "red" if (leaf.rank != "phylum" and leaf.is_zero) else "black"
        branch_end_x = label_anchor_x - 0.45
        ax.plot([leaf.x, branch_end_x], [y, y], color="#696969", lw=branch_thick, solid_capstyle="round")

        if leaf.data:
            row = leaf.data
            scatter_scale = 18.0

            val_q1_tm = parse_numeric(row.get("mean_q1_Tm"))
            if val_q1_tm is not None:
                red_diam = scale_value(val_q1_tm, tm_range[0], tm_range[1], size_min, size_max)
                # 円 (marker='o')
                ax.scatter(
                    red_cx,
                    y,
                    s=(red_diam * scatter_scale) ** 2.1,
                    marker="o",
                    color="#e74c3c",
                    alpha=1.0,
                    edgecolors="#696969",
                    linewidths=0.8,
                    zorder=10,
                )

            val_q1_identity = parse_numeric(row.get("mean_q1_identity"))
            if val_q1_identity is not None:
                blue_diam = scale_value(val_q1_identity, id_range[0], id_range[1], size_min, size_max)
                ax.scatter(
                    blue_cx,
                    y,
                    s=(blue_diam * scatter_scale) ** 2.1,
                    marker="o",
                    color="cyan",
                    alpha=1.0,
                    edgecolors="#696969",
                    linewidths=0.8,
                    zorder=10,
                )

            val_q2_tm = parse_numeric(row.get("mean_q2_Tm"))
            if val_q2_tm is not None:
                red_diag = scale_value(val_q2_tm, tm_range[0], tm_range[1], size_min, size_max)
                # ひし形 (marker='D')
                ax.scatter(
                    red_dx,
                    y,
                    s=(red_diag * scatter_scale) ** 2,
                    marker="D",
                    color="#e74c3c",
                    alpha=1.0,
                    edgecolors="#696969",
                    linewidths=0.8,
                    zorder=10,
                )

            val_q2_identity = parse_numeric(row.get("mean_q2_identity"))
            if val_q2_identity is not None:
                blue_diag = scale_value(val_q2_identity, id_range[0], id_range[1], size_min, size_max)
                ax.scatter(
                    blue_dx,
                    y,
                    s=(blue_diag * scatter_scale) ** 2,
                    marker="D",
                    color="cyan",
                    alpha=1.0,
                    edgecolors="#696969",
                    linewidths=0.8,
                    zorder=10,
                )

            text_kwargs = dict(ha="center", va="center", fontsize=80, color="black", fontweight="bold", zorder=20)
            # ▼ 円・ひし形の数値：ずらしたい量をここで設定（数字を大きくするともっと下に行きます）
            
            if leaf_count <= 30:
               text_offset_y = y_step * 0.36
            elif leaf_count <= 80:
               text_offset_y = y_step * 0.45
            else:
               text_offset_y = y_step * 0.54

            if val_q1_tm is not None:
                ax.text(red_cx, y - text_offset_y, f"{val_q1_tm:.2f}", **text_kwargs)
            if val_q1_identity is not None:
                ax.text(blue_cx, y - text_offset_y, f"{val_q1_identity:.2f}", **text_kwargs)
            if val_q2_tm is not None:
                ax.text(red_dx, y - text_offset_y, f"{val_q2_tm:.2f}", **text_kwargs)
            if val_q2_identity is not None:
                ax.text(blue_dx, y - text_offset_y, f"{val_q2_identity:.2f}", **text_kwargs)

#tax_count長方形の設定
            origin_val = parse_numeric(row.get("origin_tax_count"))
            origin_draw = origin_val is not None and not math.isclose(origin_val, 0.0)
            if origin_draw and origin_scale is not None:
                bar_len = calc_bar_len(origin_val, origin_scale)
                bar_rect = plt.Rectangle(
                    (bar_start_x * 1.13, y - bar_height / 2),
                    bar_len,
                    bar_height,
                    facecolor="#b0b0b0",
                    edgecolor="#696969",
                    lw=0.8,
                    zorder=7,
                )
                ax.add_patch(bar_rect)

            tax_val = parse_numeric(row.get("taxid_count"))
            tax_draw = tax_val is not None and not math.isclose(tax_val, 0.0)
            if tax_draw and tax_scale is not None:
                bar_len = calc_bar_len(tax_val, tax_scale)
                bar_rect = plt.Rectangle(
                    (bar_start_x * 1.13, y - bar_height / 2),
                    bar_len,
                    bar_height,
                    facecolor="#05a54b",
                    edgecolor="#696969",
                    lw=0.8,
                    zorder=8,
                )
                ax.add_patch(bar_rect)

            label_text = None
            if tax_draw:
                label_text = f"{tax_val:.0f}"
            if origin_draw:
                origin_label = f"{origin_val:.0f}"
                if label_text:
                    label_text = f"{label_text} ({origin_label})"
                else:
                    label_text = f"({origin_label})"
            if label_text:
                ax.text(
                    bar_start_x * 1.12 - 4.0,
                    y,
                    label_text,
                    ha="right",
                    va="center",
                    fontsize=80,
                    color="black",
                    fontweight="bold",
                    zorder=25,
                )
            if leaf.taxid and leaf.taxid in taxid_to_name:
                label = taxid_to_name[leaf.taxid]
            else:
                label = extract_label(row, ranks_in_use)
        else:
            if leaf.taxid and leaf.taxid in taxid_to_name:
                label = taxid_to_name[leaf.taxid]
            else:
                label = leaf.name

        # (前略) for leaf in leaves: ループの最後の方
        
        # --- 背景色の設定 (葉ノード用) ---
        # is_zero (データ欠損など) の場合は背景を薄い赤、通常は白にする例
        if leaf.rank != "phylum" and leaf.is_zero:
            bg_color = "#ffeaea"  # 薄い赤
        else:
            bg_color = "white"    # 白 (透明にしたい場合は alphaを下げる)

        leaf_bbox = dict(
            boxstyle="square,pad=0.4",  # 四角形、余白少なめ
            fc=bg_color,                # 背景色
            ec="none",                  # 枠線なし
            alpha=0.9                   # 少し透過させる
        )

        ax.text(
            label_anchor_x + 10.0,
            y,
            label,
            ha="left",
            va="center",
            fontsize=85,
            fontstyle="italic",
            color=text_color,  # 元の文字色設定(赤/黒)はそのまま維持
            bbox=leaf_bbox     # ★ここに追加
        )


        ax.plot(
            [branch_end_x, label_anchor_x],
            [y, y],
            color="#696969",
            lw=branch_thick,
            solid_capstyle="round",
        )

    # (前略) 葉のループが終わった後

    for child in sorted(root.children.values(), key=lambda c: c.y, reverse=True):
        if child.rank != "phylum":
            continue
        disp = child.display_name or child.name

        # --- 背景色の設定 (Phylum用) ---
        phylum_bbox = dict(
            boxstyle="round,pad=0.4",   # 角丸、余白広め
            fc=bg_color,               
            ec="none",          
            alpha=0.9                   # あまり透けさせない
        )

        ax.text(
            root.x - 0.2,
            child.y,
            disp,
            ha="right",
            va="center",
            fontsize=85,
            fontstyle="italic",
            color="black",
            bbox=phylum_bbox  # ★ここに追加
        )

    # (後略) if show_class_labels: ...

    # adjustText の import は削除してください（ファイルの先頭にある場合）

    # ---------------------------------------------------------
    if show_class_labels:
        branch_label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.2)
        
        # 表示対象とするランク
        excluded_ranks = {rank for rank in (exclude_label_ranks or []) if rank}
        target_ranks = set(ranks_in_use) - {"phylum"} - excluded_ranks
        
        # 1. まずは表示候補のノードをすべて集める
        raw_candidates: List[TreeNode] = []
        candidate_ids = set()  # TreeNodeはハッシュ不可のためidで管理

        def collect_branch_nodes(node: TreeNode) -> None:
            if node.rank in target_ranks:
                raw_candidates.append(node)
                candidate_ids.add(id(node))
            for ch in node.children.values():
                collect_branch_nodes(ch)

        collect_branch_nodes(root)

        # 2. フィルタリング処理
        # 「自分より下位（具体的）なラベルがすぐ近くにある場合、自分（上位）は表示しない」
        # 実装としては「各ノードから親を辿り、近くに親ラベルがあったらその親を消す」方式で行います。
        
        suppressed_ids = set() # 表示しないことに決めたノード
        check_dist_y = 60.0      # この距離(Y)以内に親がいたら「重なってる」とみなして親を消す
        check_dist_x = 300.0     # この距離(X)以内に親がいたら「重なってる」とみなす

        for node in raw_candidates:
            # 親を遡ってチェック
            curr = node.parent
            while curr:
                # 距離が離れすぎたら、もう重ならないのでチェック終了
                if abs(curr.y - node.y) > check_dist_y:
                    break
                
                # もし親も「表示候補」に入っていたら
                if id(curr) in candidate_ids:
                    # さらにX座標（横方向）も近いか確認（全然違う枝でないか）
                    if abs(curr.x - node.x) < check_dist_x:
                        # 親(curr)は「自分(node)より上位」かつ「近い」ので消す
                        suppressed_ids.add(id(curr))
                
                curr = curr.parent

        # 除外リストに入っていないものだけを残す
        final_nodes = [n for n in raw_candidates if id(n) not in suppressed_ids]

        # 3. 描画処理（残ったノードだけを描画）
        sorted_nodes = sorted(final_nodes, key=lambda c: c.y, reverse=True)
        placed_positions: List[Tuple[float, float]] = []
        
        # 判定基準（この距離内なら重なっているとみなす）
        min_y_dist = y_step * 0.2  
        min_x_range = 400.0 

        for bnode in sorted_nodes:
            disp = bnode.display_name or bnode.name
            parent_x = bnode.parent.x if bnode.parent else bnode.x
            
            # ラベルの基準X位置（お好みで調整可）
            offset_x = (shape_spacing * 0.7 + 0.8) * 0.4
            base_label_x = min(bnode.x, parent_x) - offset_x
            
            # --- 重なり回避ロジック（改良版：上下交互サーチ） ---
            original_y = bnode.y
            final_y = original_y
            
            # 探索するオフセット順序: 0, +50, -50, +100, -100 ...
            # これにより「なるべく元の高さに近い場所」を探します
            search_offsets = [0.0]
            for i in range(1, 10):  # 最大9段階(上下合わせ18箇所)まで探す
                search_offsets.append(min_y_dist * 0.3 * i)      # 上へ
                search_offsets.append(min_y_dist * 0.3 * -i)     # 下へ
            
            found_pos = False
            for offset in search_offsets:
                test_y = original_y + offset - 10.0
                
                # 他のラベルと重なるかチェック
                collision = False
                for (px, py) in placed_positions:
                    if abs(test_y - py) < min_y_dist:       # Yが近い
                        if abs(base_label_x - px) < min_x_range: # かつXも近い
                            collision = True
                            break
                
                if not collision:
                    final_y = test_y 
                    found_pos = True
                    break
            
            # どうしても空きがない場合は、仕方ないので元の位置(最背面)に置く
            # (または search_offsets の最後を採用するなら final_y はそのまま)
            # ------------------------

            placed_positions.append((base_label_x, final_y))

            # 色の設定
            label_color = "#FF00FF"
            #if bnode.rank in ["order", "class"]:
                 #label_color = "#006400"

            ax.text(
                base_label_x,
                final_y,
                disp,
                ha="left",
                va="top",
                fontsize=80,
                fontstyle="italic",
                color=label_color,
                bbox=branch_label_bbox,
                zorder=30
            )

    ax.set_xlim(root.x - 100.0, bar_start_x + max_bar_len + 300.0)
    ax.set_ylim(min_y - y_step, max_y + y_step)
    ax.axis("off")
    # ★修正2: tight_layout は巨大画像ではエラーの原因になるので削除する
    # plt.tight_layout() 
    
    # PDF出力時は bbox_inches='tight' を使うと余白を自動調整してくれます
    fig.savefig(output_png, dpi=target_dpi, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = parse_args()

    df = read_input(args.input_csv)

    name_to_taxids, taxid_to_parent_rank, taxid_to_name = load_taxdump(args.taxdump)

    df_enriched = fill_missing_ranks(df, name_to_taxids, taxid_to_parent_rank, taxid_to_name)

    flags: List[str] = []
    if args.animalia:
        flags.append("a")
    if args.plantae:
        flags.append("p")
    if args.fungi:
        flags.append("f")
    if args.protista:
        flags.append("r")
    if args.monera:
        flags.append("m")

    df_filtered = filter_by_flags(
        df_enriched,
        RANK_ORDER,
        flags,
        name_to_taxids,
        taxid_to_parent_rank,
        taxid_to_name,
    )

    taxon_filters = normalize_taxon_filters(args.taxon_names)
    df_filtered = filter_by_taxon_names(
        df_filtered,
        taxon_filters,
        name_to_taxids,
        taxid_to_parent_rank,
        taxid_to_name,
    )

    depth_limited_df = df_filtered
    if args.depth:
        depth_idx = RANK_ORDER.index(args.depth)
        deeper_ranks = RANK_ORDER[depth_idx + 1 :]
        if deeper_ranks:
            depth_limited_df = df_filtered.copy()
            for r in deeper_ranks:
                depth_limited_df[r] = pd.NA

    ranks_in_use = [rank for rank in RANK_ORDER if not depth_limited_df[rank].isna().all()]
    if args.depth:
        allowed: List[str] = []
        for r in RANK_ORDER:
            allowed.append(r)
            if r == args.depth:
                break
        ranks_in_use = [r for r in ranks_in_use if r in allowed]

    if not ranks_in_use:
        raise ValueError("分類情報が残りませんでした。フィルター条件を確認してください。")

    show_class_labels = bool(taxon_filters)
    exclude_label_ranks = {args.depth} if args.depth else set()

    root = build_tree(depth_limited_df, ranks_in_use, name_to_taxids, taxid_to_parent_rank, taxid_to_name)
    draw_tree(
        root,
        ranks_in_use,
        taxid_to_name,
        args.output_png,
        show_class_labels=show_class_labels,
        exclude_label_ranks=exclude_label_ranks,
    )
    print(f"出力しました: {args.output_png}")


if __name__ == "__main__":
    main()
