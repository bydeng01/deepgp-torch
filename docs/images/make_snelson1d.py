# Copyright 2026 Boyuan Deng.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draw the README figure: ExactGP, SVGP and DeepGP predictions on snelson1d.

Run ``python docs/images/make_snelson1d.py`` after ``pip install -e ".[docs]"``.
It writes ``snelson1d_light.png`` and ``snelson1d_dark.png`` next to this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import deepgp  # noqa: E402
from deepgp import SVGP, DeepGP, ExactGP, fit, predict  # noqa: E402
from deepgp.data import load_snelson1d  # noqa: E402

OUT = Path(__file__).parent

# Colors follow GitHub's light and dark page themes.
THEMES = {
    "light": dict(
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        alpha=0.20,
        colors=["#2a78d6", "#eb6834", "#1baf7a"],
    ),
    "dark": dict(
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        alpha=0.28,
        colors=["#3987e5", "#d95926", "#199e70"],
    ),
}


def run():
    deepgp.seed_everything(0)
    data = load_snelson1d(test_fraction=0.2, seed=0)
    X, Y = data.X_train, data.Y_train
    grid = torch.linspace(X.min() - 0.6, X.max() + 0.6, 400).unsqueeze(-1)

    runs = [
        ("ExactGP", ExactGP(X, Y), dict(epochs=300, lr=0.1)),
        ("SVGP, M = 32", SVGP(X[:32].clone()), dict(epochs=2000, lr=0.02)),
        (
            "DeepGP, 2 layers",
            DeepGP([1, 1], num_inducing=32),
            dict(epochs=3000, lr=0.02),
        ),
    ]
    preds = []
    for title, model, kw in runs:
        fit(model, X, Y, **kw)
        # Many samples keep the Monte Carlo noise of the DeepGP band invisible.
        mean, var = predict(model, grid, k=10000)
        preds.append((title, mean, var.sqrt()))
    return X.squeeze(-1), Y, grid.squeeze(-1), preds


def draw(X, Y, grid, preds, theme, t):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharex=True, sharey=True)
    for ax, (title, mean, sd), color in zip(axes, preds, t["colors"]):
        ax.scatter(X, Y, s=7, color=t["muted"], alpha=0.6, linewidths=0, zorder=2)
        ax.fill_between(
            grid,
            mean - 2 * sd,
            mean + 2 * sd,
            color=color,
            alpha=t["alpha"],
            linewidth=0,
            zorder=1,
        )
        ax.plot(grid, mean, color=color, linewidth=1.6, zorder=3)
        ax.set_title(
            title, loc="left", color=t["ink"], fontsize=10.5, fontweight="bold"
        )
        ax.set_xlabel("x", color=t["ink2"])
        ax.set_xlim(grid[0], grid[-1])
        ax.grid(axis="y", color=t["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], length=3)
    axes[0].set_ylabel("y", color=t["ink2"])
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / f"snelson1d_{theme}.png", dpi=200, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9})
    X, Y, grid, preds = run()
    for theme, t in THEMES.items():
        draw(X, Y, grid, preds, theme, t)
