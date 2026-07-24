#!/usr/bin/env python3
"""Generate a PlotNeuralNet diagram of the casmvsnet_fpn encoder."""

from __future__ import annotations

import sys
from pathlib import Path

# Same layout and import convention as the standard PlotNeuralNet examples:
# this file is expected to live in a subdirectory of the PlotNeuralNet checkout.
sys.path.append("../")

from pycore.tikzeng import (  # noqa: E402
    to_Conv,
    to_ConvConvRelu,
    to_Pool,
    to_Sum,
    to_UnPool,
    to_begin,
    to_connection,
    to_cor,
    to_end,
    to_generate,
    to_head,
    to_skip,
)


def to_title(text: str) -> str:
    return rf"\node[font=\Large\bfseries] at (13,0,13) {{{text}}};"


def to_stage_label(node: str, text: str, yshift: float = -3.5) -> str:
    return rf"\node[align=center,font=\small] at ([yshift={yshift}cm]{node}-south) {{{text}}};"


def to_branch_connection(source: str, target: str) -> str:
    """Route a branch without implying that adjacent sequential blocks are used."""
    return rf"\draw [connection] ({source}-east) -- node {{\midarrow}} ({target}-west);"


def to_vertical_connection(source: str, target: str) -> str:
    return rf"\draw [connection] ({source}-north) -- node {{\midarrow}} ({target}-south);"


def to_output_arrow(source: str, name: str, text: str) -> str:
    return rf"""
    \draw [connection] ({source}-north) -- ++(0,0,1.4)
      node[above,align=center,font=\small] ({name}) {{{text}}};
    """


# Diagram dimensions are visual rather than literal. Spatial resolution is
# encoded by box height/depth and channels by box width and labels.
arch = [
    to_head(".."),
    to_cor(),
    to_begin(),
    to_title("CasMVSNet-style Feature Pyramid (current model)"),

    # Bottom-up encoder.
    to_ConvConvRelu(
        name="stem",
        s_filer=240,
        n_filer=(16, 16),
        offset="(0,0,0)",
        to="(0,0,0)",
        width=(1.2, 1.2),
        height=48,
        depth=64,
        caption="Stem",
    ),
    to_stage_label("stem", r"$6\!\times\!240\!\times\!320$\\$\rightarrow16\!\times\!240\!\times\!320$"),
    to_Pool(
        name="down_h2",
        offset="(0.8,0,0)",
        to="(stem-east)",
        width=0.7,
        height=40,
        depth=52,
        opacity=0.35,
        caption="stride 2",
    ),
    to_connection("stem", "down_h2"),
    to_ConvConvRelu(
        name="h2",
        s_filer=120,
        n_filer=(32, 32),
        offset="(1.0,0,0)",
        to="(down_h2-east)",
        width=(1.5, 1.5),
        height=36,
        depth=48,
        caption="H/2 block",
    ),
    to_connection("down_h2", "h2"),
    to_stage_label("h2", r"$32\!\times\!120\!\times\!160$"),
    to_Pool(
        name="down_h4",
        offset="(0.8,0,0)",
        to="(h2-east)",
        width=0.7,
        height=29,
        depth=38,
        opacity=0.35,
        caption="stride 2",
    ),
    to_connection("h2", "down_h4"),
    to_ConvConvRelu(
        name="h4",
        s_filer=60,
        n_filer=(64, 64),
        offset="(1.0,0,0)",
        to="(down_h4-east)",
        width=(2.0, 2.0),
        height=26,
        depth=34,
        caption="H/4 block",
    ),
    to_connection("down_h4", "h4"),
    to_stage_label("h4", r"$64\!\times\!60\!\times\!80$"),
    to_Pool(
        name="down_h8",
        offset="(0.8,0,0)",
        to="(h4-east)",
        width=0.7,
        height=20,
        depth=26,
        opacity=0.35,
        caption="stride 2",
    ),
    to_connection("h4", "down_h8"),
    to_ConvConvRelu(
        name="h8",
        s_filer=30,
        n_filer=(128, 128),
        offset="(1.0,0,0)",
        to="(down_h8-east)",
        width=(2.8, 2.8),
        height=18,
        depth=24,
        caption="H/8 block",
    ),
    to_connection("down_h8", "h8"),
    to_stage_label("h8", r"$128\!\times\!30\!\times\!40$"),

    # Coarse output is a projection of H/8. It is shown vertically so the
    # horizontal path remains available for top-down fusion.
    to_Conv(
        name="coarse",
        s_filer=30,
        n_filer=128,
        offset="(0,0,5.0)",
        to="(h8-north)",
        width=2.4,
        height=18,
        depth=24,
        caption="Coarse output",
    ),
    to_vertical_connection("h8", "coarse"),
    to_output_arrow("coarse", "coarse_mvs", r"H/8 MVS\\32 global depths"),

    # H/8 -> H/4 fusion. The skip from h4 represents its learned lateral 1x1.
    to_UnPool(
        name="up_h4",
        offset="(1.7,0,0)",
        to="(h8-east)",
        width=1.0,
        height=26,
        depth=34,
        opacity=0.55,
        caption=r"upsample $\times2$",
    ),
    to_branch_connection("h8", "up_h4"),
    to_Sum(
        name="sum_h4",
        offset="(1.5,0,0)",
        to="(up_h4-east)",
        radius=2.0,
        opacity=0.75,
    ),
    to_connection("up_h4", "sum_h4"),
    to_skip("h4", "sum_h4", pos=1.35),
    to_stage_label("sum_h4", r"lateral $1\!\times\!1$: $64\rightarrow128$", yshift=-2.2),
    to_Conv(
        name="middle",
        s_filer=60,
        n_filer=64,
        offset="(0,0,5.0)",
        to="(sum_h4-north)",
        width=2.0,
        height=26,
        depth=34,
        caption="Middle output",
    ),
    to_vertical_connection("sum_h4", "middle"),
    to_stage_label("middle", r"$64\!\times\!60\!\times\!80$"),
    to_output_arrow("middle", "middle_mvs", r"H/4 MVS\\8 local depths"),

    # Fused H/4 -> H/2 fusion. This is the second upsampling that is easy to
    # miss in a compact FPN drawing.
    to_Conv(
        name="reduce_h4",
        s_filer=60,
        n_filer=64,
        offset="(1.4,0,0)",
        to="(sum_h4-east)",
        width=1.4,
        height=26,
        depth=34,
        caption=r"$1\times1$ reduce",
    ),
    to_connection("sum_h4", "reduce_h4"),
    to_UnPool(
        name="up_h2",
        offset="(1.0,0,0)",
        to="(reduce_h4-east)",
        width=1.0,
        height=36,
        depth=48,
        opacity=0.55,
        caption=r"upsample $\times2$",
    ),
    to_connection("reduce_h4", "up_h2"),
    to_Sum(
        name="sum_h2",
        offset="(1.5,0,0)",
        to="(up_h2-east)",
        radius=2.0,
        opacity=0.75,
    ),
    to_connection("up_h2", "sum_h2"),
    to_skip("h2", "sum_h2", pos=1.55),
    to_stage_label("sum_h2", r"lateral $1\!\times\!1$: $32\rightarrow64$", yshift=-2.2),
    to_Conv(
        name="fine",
        s_filer=120,
        n_filer=32,
        offset="(1.5,0,0)",
        to="(sum_h2-east)",
        width=1.8,
        height=36,
        depth=48,
        caption="Fine output",
    ),
    to_connection("sum_h2", "fine"),
    to_stage_label("fine", r"$32\!\times\!120\!\times\!160$"),
    to_output_arrow("fine", "fine_mvs", r"H/2 MVS\\5 adaptive depths"),
    to_end(),
]


def main() -> None:
    output = Path(sys.argv[0]).with_suffix(".tex")
    to_generate(arch, str(output))
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
