"""Restyle four saved NTK panels and optionally insert them into the editable loss schematic."""

import argparse
import json
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from lerobot.scripts.plot_pi05_ntk_with_loss import load_ntk

PALETTE = {"vlm": "#3778A8", "action": "#D77C4D"}
MARKERS = {"vlm": "o", "action": "D"}
LABELS = {"vlm": "VLM", "action": "Action expert"}
TITLES = ("Before training", "After priming", "After stage 2", "After adaptation")
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def panel_values(stages, seeds, lookup, scope):
    return {
        module: np.asarray(
            [
                [
                    [
                        lookup[(stage["name"], seed)]["groups"][f"{scope}/{module}"][metric]
                        for metric in ("effective_rank", "parameter_normalized_energy")
                    ]
                    for seed in seeds
                ]
                for stage in stages
            ]
        )
        for module in PALETTE
    }


def common_limits(values):
    points = np.concatenate(list(values.values())).reshape(-1, 2)
    x, y = points.T
    xpad = max(float(np.ptp(x)) * 0.13, float(abs(x.mean())) * 0.015, 0.05)
    xlim = (max(0, x.min() - xpad), x.max() + xpad)
    logarithmic = bool(np.all(y > 0))
    if logarithmic:
        logs = np.log10(y)
        pad = max(float(np.ptp(logs)) * 0.18, 0.15)
        ylim = (10 ** (logs.min() - pad), 10 ** (logs.max() + pad))
    else:
        pad = max(float(np.ptp(y)) * 0.18, 1e-12)
        ylim = (max(0, y.min() - pad), y.max() + pad)
    return xlim, ylim, logarithmic


def draw_panel(axis, index, values, limits, show_y=True):
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator, NullFormatter

    xlim, ylim, logarithmic = limits
    axis.set(xlim=xlim, ylim=ylim)
    if logarithmic:
        axis.set_yscale("log")
        subs = (1, 2, 5) if np.log10(ylim[1] / ylim[0]) < 1.5 else (1,)
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=5))
        axis.yaxis.set_major_formatter(LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf)))
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#9DA8B0")
        axis.spines[side].set_linewidth(0.65)
    axis.grid(axis="y", which="major", color="#E7EBEE", linewidth=0.65, zorder=0)
    axis.tick_params(
        axis="both", which="major", labelsize=11, length=3, width=0.65, color="#9DA8B0", labelcolor="#56616B"
    )
    axis.tick_params(axis="y", which="minor", length=0)
    axis.tick_params(labelleft=show_y)
    for module, color in PALETTE.items():
        x, y = values[module][index].T
        axis.scatter(x, y, s=34, marker=MARKERS[module], color=color, alpha=0.3, edgecolors="none", zorder=2)
        xq, yq = np.quantile(x, [0.25, 0.5, 0.75]), np.quantile(y, [0.25, 0.5, 0.75])
        axis.errorbar(
            xq[1],
            yq[1],
            xerr=[[xq[1] - xq[0]], [xq[2] - xq[1]]],
            yerr=[[yq[1] - yq[0]], [yq[2] - yq[1]]],
            fmt=MARKERS[module],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=color,
            elinewidth=1.3,
            capsize=3,
            capthick=1,
            label=LABELS[module],
            zorder=4,
        )


def fill_pptx(template, destination, panels, steps):
    """Replace the template's named picture slots, retaining its native editable diagram.

    Only picture relationships/media and pending-data labels change. The original
    template and all native curves, connectors, text and axes remain intact.
    """
    if steps != [0, 750, 1000, 4000]:
        raise ValueError(
            "The supplied loss-schematic template marks 0/750/1000/4000. Results with other steps need a matching template; do not relabel a checkpoint."
        )
    template, destination = Path(template), Path(destination)
    if template.resolve() == destination.resolve():
        raise ValueError("Write the filled PPTX to a different file from its template")
    changes, filled = {}, set()
    with zipfile.ZipFile(template) as source:
        for filename in source.namelist():
            if not re.fullmatch(r"ppt/slides/slide\d+\.xml", filename):
                continue
            root = ET.fromstring(source.read(filename))
            relfile = posixpath.join(
                posixpath.dirname(filename), "_rels", posixpath.basename(filename) + ".rels"
            )
            relationships = ET.fromstring(source.read(relfile))
            existing = {element.get("Id") for element in relationships}
            changed = False
            for picture in root.findall(".//p:pic", NS):
                props = picture.find("p:nvPicPr/p:cNvPr", NS)
                marker = props.get("descr", "") if props is not None else ""
                if not marker.startswith("NTK_SLOT:"):
                    # Some PPTX exporters omit image alt text. The template also
                    # has a named pending-data label inside each picture frame.
                    frame = picture.find("p:spPr/a:xfrm", NS)
                    matches = []
                    if frame is not None:
                        offset, size = frame.find("a:off", NS), frame.find("a:ext", NS)
                        px, py = int(offset.get("x")), int(offset.get("y"))
                        pw, ph = int(size.get("cx")), int(size.get("cy"))
                        for label in root.findall(".//p:sp", NS):
                            label_props = label.find("p:nvSpPr/p:cNvPr", NS)
                            match = re.fullmatch(
                                r"NTK_PENDING_(backbone|prompts)_(before|priming|stage2|final)-note",
                                label_props.get("name", "") if label_props is not None else "",
                            )
                            box = label.find("p:spPr/a:xfrm", NS)
                            if match and box is not None:
                                start, span = box.find("a:off", NS), box.find("a:ext", NS)
                                cx = int(start.get("x")) + int(span.get("cx")) / 2
                                cy = int(start.get("y")) + int(span.get("cy")) / 2
                                if px <= cx <= px + pw and py <= cy <= py + ph:
                                    matches.append(match.groups())
                    if not matches:
                        continue
                    if len(matches) != 1:
                        raise ValueError("Ambiguous NTK picture slot; use the original template layout")
                    key = matches[0]
                else:
                    key = tuple(marker.split(":")[1:])
                if key not in panels:
                    raise ValueError(f"Template requires {key}; export both scopes with --scope=both")
                relationship_id = f"rIdNtkPanel{len(filled) + 1}"
                while relationship_id in existing:
                    relationship_id += "x"
                existing.add(relationship_id)
                media = f"ntk_{key[0]}_{key[1]}.png"
                ET.SubElement(
                    relationships,
                    f"{{{REL_NS}}}Relationship",
                    {"Id": relationship_id, "Type": NS["r"] + "/image", "Target": "../media/" + media},
                )
                picture.find("p:blipFill/a:blip", NS).set(f"{{{NS['r']}}}embed", relationship_id)
                props.set("descr", f"Measured {key[0]} NTK at {key[1]}; paired-seed median and IQR")
                changes[f"ppt/media/{media}"] = Path(panels[key]).read_bytes()
                filled.add(key)
                changed = True
            if changed:
                tree = root.find("p:cSld/p:spTree", NS)
                for shape in list(tree):
                    props = shape.find("p:nvSpPr/p:cNvPr", NS)
                    if props is not None and props.get("name", "").startswith("NTK_PENDING_"):
                        tree.remove(shape)
                changes[filename] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                changes[relfile] = ET.tostring(relationships, encoding="utf-8", xml_declaration=True)
        if not filled:
            raise ValueError(
                "No NTK_SLOT picture placeholders found; use the provided editable schematic template"
            )
        temporary = destination.with_suffix(".pptx.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                target.writestr(info, changes.pop(info.filename, source.read(info.filename)))
            for filename, content in changes.items():
                target.writestr(filename, content)
        temporary.replace(destination)
    return filled


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--scope", choices=("backbone", "prompts", "both"), default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--pptx-template",
        type=Path,
        help="Optional editable loss-schematic PPTX to fill with these real panels",
    )
    args = parser.parse_args()
    stages, seeds, lookup, scopes = load_ntk(args.results, args.scope)
    out = args.output_dir or args.results.parent / "panels"
    out.mkdir(parents=True, exist_ok=True)
    panels = {}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": "#46525D",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    ):
        for scope in scopes:
            values = panel_values(stages, seeds, lookup, scope)
            limits = common_limits(values)
            for index, stage in enumerate(stages):
                fig, axis = plt.subplots(figsize=(4.4, 3.7))
                fig.subplots_adjust(left=0.23, right=0.975, top=0.94, bottom=0.20)
                draw_panel(axis, index, values, limits)
                axis.set_xlabel("Spectral effective rank", fontsize=13, labelpad=8)
                axis.set_ylabel("Parameter-normalized\ntangent energy", fontsize=13, labelpad=8)
                for extension in ("png", "pdf", "svg"):
                    path = out / f"{scope}_{stage['name']}.{extension}"
                    fig.savefig(path, dpi=300)
                    if extension == "png":
                        panels[(scope, stage["name"])] = path
                plt.close(fig)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharex=True, sharey=True)
            fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.73, wspace=0.15)
            for index, (stage, axis) in enumerate(zip(stages, axes, strict=True)):
                draw_panel(axis, index, values, limits, show_y=index == 0)
                axis.set_title(
                    f"{TITLES[index]}\nStep {stage['total_step']}", fontsize=12, color="#25333F", pad=14
                )
            axes[0].set_ylabel("Parameter-normalized\ntangent energy", fontsize=12, labelpad=10)
            fig.supxlabel("Spectral effective rank", fontsize=12, y=0.06, color="#46525D")
            handles = [
                Line2D(
                    [],
                    [],
                    color=color,
                    marker=MARKERS[module],
                    markersize=7,
                    linestyle="none",
                    label=LABELS[module],
                )
                for module, color in PALETTE.items()
            ]
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.52, 1.015),
                ncol=2,
                frameon=False,
                handletextpad=0.4,
                columnspacing=2,
            )
            for extension in ("png", "pdf", "svg"):
                fig.savefig(out / f"{scope}_four_stages.{extension}", dpi=300)
            plt.close(fig)
    steps = [stage["total_step"] for stage in stages]
    if args.pptx_template:
        destination = out / "PrimingVLA-NTK-Loss-Filled.pptx"
        filled = fill_pptx(args.pptx_template, destination, panels, steps)
        print(f"Inserted {len(filled)} measured panels into {destination}")
    (out / "panel_manifest.json").write_text(
        json.dumps(
            {
                "source_results": str(args.results.resolve()),
                "cumulative_steps": steps,
                "seeds": seeds,
                "panels": {"/".join(key): path.name for key, path in panels.items()},
                "loss_is_schematic": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved single-panel and one-row PNG/PDF/SVG figures to {out}; no model, GPU, or loss CSV needed.")


if __name__ == "__main__":
    main()
