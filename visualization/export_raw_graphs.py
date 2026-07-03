#!/usr/bin/env python3
"""Export raw ALD material graphs as PNG files."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/font-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "qwen_extracted_info"
OUTPUT_DIR = ROOT / "raw_graph_pngs"
BIN_SIZE = 50
CHART_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#9D755D",
]
TEXT_COLOR = "#1f2933"
MUTED_TEXT = "#52616b"
AXIS_COLOR = "#9aa5b1"
GRID_COLOR = "#d9e2ec"
FIGURE_BG = "#ffffff"
PLOT_BG = "#ffffff"
BAR_COLOR = "#4C78A8"
BAR_EDGE = "#2F5F8F"
SANKEY_COLORS = {"precursor": "#2F6F73", "coreactant": "#B76E00", "phase": "#5B5F97"}
SANKEY_FILLS = {"precursor": "#E2F0F1", "coreactant": "#FFF0D6", "phase": "#ECEBFA"}
DIGIT_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

plt.rcParams.update(
    {
        "axes.labelsize": 18,
        "axes.titlesize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    }
)

CHEMICAL_ALIASES = [
    ("H2O", ["h2o", "h20", "water", "deionized water", "di water", "distilled water", "h2o vapor", "water vapor"]),
    ("O3", ["o3", "ozone"]),
    ("O2", ["o2", "oxygen", "molecular oxygen"]),
    ("O2 plasma", ["o2 plasma", "oxygen plasma", "o2-plasma"]),
    ("H2O2", ["h2o2", "hydrogen peroxide"]),
    ("NH3", ["nh3", "ammonia"]),
    ("N2", ["n2", "nitrogen"]),
    ("Ar", ["ar", "argon"]),
    ("TMA", ["tma", "al(ch3)3", "trimethylaluminum", "trimethyl aluminium", "aluminum trimethyl", "aluminium trimethyl"]),
    ("TDMAT", ["tdmat", "tetrakis(dimethylamido)titanium", "tetrakis(dimethylamino)titanium"]),
    ("TiCl4", ["ticl4", "titanium tetrachloride"]),
    ("TTIP", ["ttip", "titanium isopropoxide", "titanium(iv) isopropoxide", "titanium tetraisopropoxide"]),
    ("TDMAH", ["tdmah", "tdmahf", "tetrakis(dimethylamido)hafnium", "tetrakis(dimethylamino)hafnium"]),
    ("TEMAH", ["temah", "temahf", "tetrakis(ethylmethylamido)hafnium", "tetrakis(ethylmethylamino)hafnium"]),
    ("HfCl4", ["hfcl4", "hafnium chloride", "hafnium tetrachloride"]),
    ("HfI4", ["hfi4", "hafnium iodide", "hafnium tetraiodide"]),
    ("Hf(NO3)4", ["hf(no3)4", "hafnium nitrate"]),
    ("DEZ", ["dez", "diethylzinc", "diethyl zinc"]),
    ("ZnO", ["zno", "zinc oxide"]),
]


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def is_reported(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "n/a", "na", "null", "not reported", "unknown", "none"}


def normalize_chemical_name(value: str) -> str:
    subscript_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return (
        value.translate(subscript_digits)
        .lower()
        .replace("\n", " ")
        .strip()
    )


def chemical_label(item: Any) -> str:
    if isinstance(item, str):
        candidates = [item]
    elif isinstance(item, dict):
        candidates = [item.get("abbreviation", ""), item.get("full_name", "")]
    else:
        candidates = []

    normalized: set[str] = set()
    for candidate in candidates:
        if not is_reported(candidate):
            continue
        clean = re.sub(r"\s+", " ", normalize_chemical_name(candidate))
        compact = clean.replace(" (", "(").replace("( ", "(").replace(" )", ")")
        normalized.add(compact)
        normalized.add(re.sub(r"(?<=[a-z])\s+(?=\d)|(?<=\d)\s+(?=[a-z])", "", compact))
        normalized.add(re.sub(r"\s*\([^)]*\)", "", clean).strip())
        normalized.update(part.strip() for part in re.findall(r"\(([^)]+)\)", clean) if is_reported(part))
    for label, aliases in CHEMICAL_ALIASES:
        if any(alias in normalized for alias in aliases):
            return label
    for candidate in candidates:
        if is_reported(candidate):
            return str(candidate).strip()
    return ""


def subscript_formula(value: str) -> str:
    return value.translate(DIGIT_SUBSCRIPTS)


def phase_label(value: Any) -> str:
    if not is_reported(value):
        return "Not reported"
    normalized = re.sub(r"\s+", " ", str(value).strip())
    lower = normalized.lower()

    phase_checks = [
        ("Amorphous", ("amorphous",)),
        ("Anatase", ("anatase",)),
        ("Rutile", ("rutile",)),
        ("Brookite", ("brookite",)),
        ("Monoclinic", ("monoclinic", "m-", "m phase", "m-phase")),
        ("Tetragonal", ("tetragonal", "t-", "t phase", "t-phase")),
        ("Orthorhombic", ("orthorhombic", "o-", "o phase", "o-phase")),
        ("Cubic", ("cubic", " c ", "c-phase", "c phase")),
        ("Gamma", ("gamma", "γ")),
        ("Alpha", ("alpha", "α")),
    ]
    matches = [
        label
        for label, tokens in phase_checks
        if any(token in f" {lower} " for token in tokens)
    ]
    if matches:
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            return unique_matches[0]
        return " + ".join(unique_matches[:3])

    if "amorphous" in lower:
        return "Amorphous"
    if "crystalline" in lower or "polycrystalline" in lower:
        return "Crystalline"
    return normalized


def parse_temperature(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if not isinstance(value, str) or not is_reported(value):
        return None
    normalized = value.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", normalized)
    numbers = [float(match) for match in re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)]
    numbers = [number for number in numbers if number >= 0]
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    return sum(numbers[:2]) / 2


def load_papers() -> list[dict[str, Any]]:
    papers = []
    for paper_dir in sorted(DATA_DIR.iterdir(), key=lambda item: item.name):
        if not paper_dir.is_dir():
            continue
        target = read_json(paper_dir / "target_material.json", {})
        target_material = target.get("target_material", {}) if isinstance(target, dict) else {}
        formula = target_material.get("chemical_formula", "")
        if not is_reported(formula):
            continue

        deposition = read_json(paper_dir / "deposition_conditions.json", {})
        film = read_json(paper_dir / "film_properties.json", {})
        chemistry = read_json(paper_dir / "precursor_coreactant.json", {})
        papers.append(
            {
                "id": paper_dir.name,
                "formula": str(formula).strip(),
                "temperature": parse_temperature(deposition.get("deposition_temperature_C") if isinstance(deposition, dict) else None),
                "phase": phase_label(film.get("crystal_phase") if isinstance(film, dict) else None),
                "precursors": chemistry.get("precursors", []) if isinstance(chemistry, dict) else [],
                "coreactants": chemistry.get("coreactants", []) if isinstance(chemistry, dict) else [],
            }
        )
    return papers


def material_papers(papers: list[dict[str, Any]], formula: str) -> list[dict[str, Any]]:
    return [paper for paper in papers if paper["formula"].lower() == formula.lower()]


def temperature_bins(papers: list[dict[str, Any]]) -> list[tuple[int, int]]:
    counts: Counter[int] = Counter()
    for paper in papers:
        temperature = paper["temperature"]
        if temperature is None:
            continue
        start = int(math.floor(temperature / BIN_SIZE) * BIN_SIZE)
        counts[start] += 1
    return sorted(counts.items())


def temperature_phase_bins(papers: list[dict[str, Any]]) -> tuple[list[int], list[str], dict[int, Counter[str]]]:
    phase_counts: Counter[str] = Counter()
    records = []
    for paper in papers:
        if paper["temperature"] is None or paper["phase"] == "Not reported":
            continue
        records.append((paper["temperature"], paper["phase"]))
        phase_counts[paper["phase"]] += 1

    top_phases = [phase for phase, _ in sorted(phase_counts.items(), key=lambda item: (-item[1], item[0]))[:6]]
    top_set = set(top_phases)
    phases = top_phases + (["Other"] if len(phase_counts) > len(top_phases) else [])
    bins: dict[int, Counter[str]] = defaultdict(Counter)
    for temperature, phase in records:
        start = int(math.floor(temperature / BIN_SIZE) * BIN_SIZE)
        bins[start][phase if phase in top_set else "Other"] += 1
    return sorted(bins), phases, bins


def build_chemistry_flow(papers: list[dict[str, Any]]) -> dict[str, Any]:
    precursor_totals: Counter[str] = Counter()
    coreactant_totals: Counter[str] = Counter()
    phase_totals: Counter[str] = Counter()
    precursor_coreactant_edges: Counter[tuple[str, str]] = Counter()
    coreactant_phase_edges: Counter[tuple[str, str]] = Counter()

    for paper in papers:
        precursors = sorted({chemical_label(item) for item in paper["precursors"] if is_reported(chemical_label(item))})
        coreactants = sorted({chemical_label(item) for item in paper["coreactants"] if is_reported(chemical_label(item))})
        phase = paper["phase"]
        if not precursors or not coreactants:
            continue
        for precursor in precursors:
            precursor_totals[precursor] += 1
            for coreactant in coreactants:
                precursor_coreactant_edges[(precursor, coreactant)] += 1
        for coreactant in coreactants:
            coreactant_totals[coreactant] += 1
            if phase != "Not reported":
                phase_totals[phase] += 1
                coreactant_phase_edges[(coreactant, phase)] += 1

    def top(counter: Counter[str], limit: int = 6) -> list[str]:
        return [name for name, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]]

    precursors = top(precursor_totals)
    coreactants = top(coreactant_totals)
    phases = top(phase_totals)
    edges = []
    for (source, target), count in precursor_coreactant_edges.items():
        if source in precursors and target in coreactants:
            edges.append(("precursor", source, "coreactant", target, count))
    for (source, target), count in coreactant_phase_edges.items():
        if source in coreactants and target in phases:
            edges.append(("coreactant", source, "phase", target, count))
    return {
        "columns": {"precursor": precursors, "coreactant": coreactants, "phase": phases},
        "totals": {"precursor": precursor_totals, "coreactant": coreactant_totals, "phase": phase_totals},
        "edges": edges,
    }


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PLOT_BG)
    ax.figure.set_facecolor(FIGURE_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=16)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.9, linewidth=0.7)
    ax.set_axisbelow(True)


def save_temperature_distribution(formula: str, papers: list[dict[str, Any]], filename: str) -> None:
    data = temperature_bins(papers)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    style_axes(ax)
    if data:
        labels = [f"{start}-{start + BIN_SIZE - 1}" for start, _ in data]
        counts = [count for _, count in data]
        ax.bar(labels, counts, color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.6)
        ax.set_ylabel("Number of studies")
        ax.set_xlabel("Distribution temperature (°C)")
        ax.tick_params(axis="x", rotation=35)
        for index, count in enumerate(counts):
            ax.text(index, count + max(counts) * 0.02, str(count), ha="center", va="bottom", color=TEXT_COLOR, fontsize=18)
    else:
        ax.text(0.5, 0.5, "No reported temperatures", ha="center", va="center", color=MUTED_TEXT, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_temperature_phase(formula: str, papers: list[dict[str, Any]], filename: str) -> None:
    starts, phases, bins = temperature_phase_bins(papers)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    style_axes(ax)
    if starts and phases:
        labels = [f"{start}-{start + BIN_SIZE - 1}" for start in starts]
        bottoms = [0] * len(starts)
        for index, phase in enumerate(phases):
            values = [bins[start][phase] for start in starts]
            ax.bar(
                labels,
                values,
                bottom=bottoms,
                color=CHART_COLORS[index % len(CHART_COLORS)],
                edgecolor=FIGURE_BG,
                linewidth=0.35,
                label=phase,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_ylabel("Number of studies")
        ax.set_xlabel("Distribution temperature (°C)")
        ax.tick_params(axis="x", rotation=35)
        legend = ax.legend(
            facecolor=FIGURE_BG,
            edgecolor=AXIS_COLOR,
            labelcolor=TEXT_COLOR,
            ncols=min(3, len(phases)),
            frameon=True,
            fontsize=11,
        )
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
    else:
        ax.text(0.5, 0.5, "No paired temperature and crystal phase data", ha="center", va="center", color=MUTED_TEXT, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def truncate(label: str, limit: int = 20) -> str:
    return label if len(label) <= limit else f"{label[: limit - 1]}..."


def save_chemistry_flow(formula: str, papers: list[dict[str, Any]], filename: str) -> None:
    flow = build_chemistry_flow(papers)
    columns = flow["columns"]
    totals = flow["totals"]
    edges = flow["edges"]
    max_rows = max(*(len(values) for values in columns.values()), 1)
    height = max(6.0, max_rows * 0.90 + 2.2)
    fig, ax = plt.subplots(figsize=(16, height), dpi=300)
    ax.set_facecolor(FIGURE_BG)
    fig.set_facecolor(FIGURE_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = {"precursor": 0.04, "coreactant": 0.38, "phase": 0.72}
    titles = {"precursor": "Precursor", "coreactant": "Coreactant", "phase": "Crystal Phase"}
    node_w = 0.21
    node_h = 0.08
    node_positions: dict[tuple[str, str], tuple[float, float]] = {}

    for column, names in columns.items():
        ax.text(x_positions[column], 0.95, titles[column], color=MUTED_TEXT, fontsize=28, fontweight="bold")
        rows = max(len(names), 1)
        for index, name in enumerate(names):
            y = 0.82 if rows == 1 else 0.82 - index * (0.68 / (rows - 1))
            node_positions[(column, name)] = (x_positions[column], y)

    if edges:
        max_count = max(edge[4] for edge in edges)
        for source_column, source, target_column, target, count in edges:
            if (source_column, source) not in node_positions or (target_column, target) not in node_positions:
                continue
            sx, sy = node_positions[(source_column, source)]
            tx, ty = node_positions[(target_column, target)]
            x1 = sx + node_w
            x2 = tx
            y1 = sy
            y2 = ty
            control = max(0.12, (x2 - x1) * 0.45)
            path = MplPath(
                [(x1, y1), (x1 + control, y1), (x2 - control, y2), (x2, y2)],
                [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
            )
            edge_color = SANKEY_COLORS["precursor"] if source_column == "precursor" else SANKEY_COLORS["coreactant"]
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=edge_color,
                    linewidth=max(1.5, 9 * count / max_count),
                    alpha=0.34,
                    capstyle="round",
                )
            )

    for column, names in columns.items():
        for name in names:
            x, y = node_positions[(column, name)]
            ax.add_patch(
                Rectangle(
                    (x, y - node_h / 2),
                    node_w,
                    node_h,
                    facecolor=SANKEY_FILLS[column],
                    edgecolor=SANKEY_COLORS[column],
                    linewidth=1.0,
                )
            )
            ax.text(x + 0.015, y, truncate(subscript_formula(name)), color=TEXT_COLOR, fontsize=18, fontweight="bold", va="center")
            ax.text(
                x + node_w - 0.015,
                y,
                str(totals[column][name]),
                color=SANKEY_COLORS[column],
                fontsize=18,
                fontweight="bold",
                va="center",
                ha="right",
            )

    if not edges:
        ax.text(0.5, 0.5, "Not enough precursor-coreactant-phase data", ha="center", va="center", color=MUTED_TEXT)
    ax.set_title(f"{subscript_formula(formula)} ALD Chemistry Flow", color=TEXT_COLOR, fontsize=28, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    papers = load_papers()
    requested = [
        ("Al2O3", "chemistry", "al2o3_ald_chemistry_flow.png"),
        ("Al2O3", "phase", "al2o3_temperature_vs_crystal_phase.png"),
        ("TiO2", "chemistry", "tio2_ald_chemistry_flow.png"),
        ("TiO2", "phase", "tio2_temperature_vs_crystal_phase.png"),
        ("HfO2", "chemistry", "hfo2_ald_chemistry_flow.png"),
        ("HfO2", "phase", "hfo2_temperature_vs_crystal_phase.png"),
    ]

    for formula, graph_type, filename in requested:
        selected = material_papers(papers, formula)
        if graph_type == "chemistry":
            save_chemistry_flow(formula, selected, filename)
        elif graph_type == "temperature":
            save_temperature_distribution(formula, selected, filename)
        elif graph_type == "phase":
            save_temperature_phase(formula, selected, filename)
        print(f"{filename}: {len(selected)} {formula} papers")

    print(f"Saved {len(requested)} PNG files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
