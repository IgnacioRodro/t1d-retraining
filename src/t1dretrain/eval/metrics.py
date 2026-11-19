"""Forecast accuracy (rule R6): RMSE, MAE, MARD, bias and the Clarke Error Grid.

Every evaluation of a model in this project reports, for each group of
forecasts (model x horizon x dataset x patient group, or per patient):

- ``rmse``, ``mae``, ``mard_pct`` (mean |forecast - reference| / reference,
  in %), ``bias`` (mean forecast - reference) and ``n``;
- the share of forecasts in each Clarke Error Grid zone (``clarke_A_pct`` to
  ``clarke_E_pct``) and ``clarke_AB_pct``.

The Clarke grid is a paper figure, so what is needed to redraw it is saved as
well (``ClarkeAccumulator``):

- a sparse 2-D histogram of (reference, forecast) pairs in 5 mg/dL bins with
  the zone of each pair as part of the key (``ref_bin``, ``pred_bin``,
  ``zone``, ``count``): summing ``count`` by zone gives the exact zone counts,
  and summing over zones gives the density to draw;
- optionally a random sample of pairs (bottom-k sample on random priorities,
  so samples from parallel workers merge into a sample of the union).

Zones follow Clarke et al., Diabetes Care 1987;10:622-628, in the usual
implementation (reference and forecast in mg/dL, checked in this order):

- A: both <= 70, or the forecast within 20 % of the reference;
- E: reference >= 180 and forecast <= 70, or reference <= 70 and forecast >= 180;
- C: 70 <= reference <= 290 and forecast >= reference + 110, or
  130 <= reference <= 180 and forecast <= 7/5 reference - 182;
- D: reference >= 240 and 70 <= forecast <= 180, or reference <= 175/3 and
  70 <= forecast <= 180, or 175/3 <= reference <= 70 and forecast >= 6/5 reference;
- B: everything else.

Forecasts are scored as given (no clipping) for the metrics and zones; the
histogram clips values to [0, ``HIST_MAX_MGDL``) so that every pair is stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ZONES = ["A", "B", "C", "D", "E"]
HIST_MAX_MGDL = 600.0
PLOT_MAX_MGDL = 400.0
METRIC_COLUMNS = ["n", "rmse", "mae", "mard_pct", "bias"] + [f"clarke_{z}_pct" for z in ZONES] + ["clarke_AB_pct"]

# Boundary segments of the grid, ((x0, y0), (x1, y1)) with x = reference, y = forecast, mg/dL.
CLARKE_LINES = [
    ((0, 70), (175 / 3, 70)),
    ((175 / 3, 70), (400 / 1.2, 400)),
    ((70, 84), (70, 400)),
    ((0, 180), (70, 180)),
    ((70, 180), (290, 400)),
    ((70, 0), (70, 56)),
    ((70, 56), (400, 320)),
    ((180, 0), (180, 70)),
    ((180, 70), (400, 70)),
    ((240, 70), (240, 180)),
    ((240, 180), (400, 180)),
    ((130, 0), (180, 70)),
]
ZONE_LABELS = [("A", 30, 15), ("B", 370, 260), ("B", 280, 370), ("C", 160, 370), ("C", 160, 15), ("D", 30, 140),
               ("D", 370, 120), ("E", 30, 370), ("E", 370, 15)]


# ----------------------------------------------------------------------------- zones and metrics


def clarke_zone_codes(reference: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Zone of each pair as an integer 0-4 (A-E)."""
    r = np.asarray(reference, dtype=float)
    p = np.asarray(forecast, dtype=float)
    zone_a = ((r <= 70) & (p <= 70)) | ((p <= 1.2 * r) & (p >= 0.8 * r))
    zone_e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    zone_c = ((r >= 70) & (r <= 290) & (p >= r + 110)) | ((r >= 130) & (r <= 180) & (p <= 7 / 5 * r - 182))
    zone_d = ((r >= 240) & (p >= 70) & (p <= 180)) | ((r <= 175 / 3) & (p >= 70) & (p <= 180)) | \
        ((r >= 175 / 3) & (r <= 70) & (p >= 6 / 5 * r))
    return np.select([zone_a, zone_e, zone_c, zone_d], [0, 4, 2, 3], default=1).astype(np.int8)


def clarke_zones(reference: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Zone letter of each pair."""
    return np.asarray(ZONES)[clarke_zone_codes(reference, forecast)]


def _finite_pairs(reference, forecast) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(reference, dtype=float)
    p = np.asarray(forecast, dtype=float)
    ok = np.isfinite(r) & np.isfinite(p)
    return r[ok], p[ok]


def point_metrics(reference, forecast) -> dict:
    """All metrics of rule R6 for one group of forecasts (pairs with a missing value are ignored)."""
    r, p = _finite_pairs(reference, forecast)
    out = {"n": int(len(r))}
    if len(r) == 0:
        out.update({c: np.nan for c in METRIC_COLUMNS if c != "n"})
        return out
    error = p - r
    out.update({
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "mard_pct": float(100 * np.mean(np.abs(error) / r)),
        "bias": float(np.mean(error)),
    })
    counts = np.bincount(clarke_zone_codes(r, p), minlength=5)
    for zone, count in zip(ZONES, counts):
        out[f"clarke_{zone}_pct"] = float(100 * count / len(r))
    out["clarke_AB_pct"] = out["clarke_A_pct"] + out["clarke_B_pct"]
    return out


def metrics_table(frame: pd.DataFrame, by: list[str], reference: str = "reference",
                  forecast: str = "forecast") -> pd.DataFrame:
    """``point_metrics`` for every group of ``frame`` (one row per group)."""
    rows = []
    for key, group in frame.groupby(by, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **point_metrics(group[reference], group[forecast])})
    return pd.DataFrame(rows, columns=by + METRIC_COLUMNS)


def zone_table(counts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Zone counts and shares per group from a (sparse) histogram with columns ``by + [zone, count]``."""
    wide = counts.pivot_table(index=by, columns="zone", values="count", aggfunc="sum", fill_value=0)
    wide = wide.reindex(columns=ZONES, fill_value=0)
    out = wide.rename(columns={z: f"clarke_{z}_n" for z in ZONES})
    total = wide.sum(axis=1)
    out.insert(0, "n", total)
    for zone in ZONES:
        out[f"clarke_{zone}_pct"] = 100 * wide[zone] / total
    out["clarke_AB_pct"] = out["clarke_A_pct"] + out["clarke_B_pct"]
    out.columns.name = None
    return out.reset_index()


# ----------------------------------------------------------------------------- what is saved to redraw the grid


def clarke_histogram(reference, forecast, bin_mgdl: float = 5.0) -> pd.DataFrame:
    """Sparse 2-D histogram: lower bin edges (mg/dL), zone and count of every non-empty cell."""
    r, p = _finite_pairs(reference, forecast)
    if len(r) == 0:
        return pd.DataFrame({"ref_bin": pd.Series(dtype=float), "pred_bin": pd.Series(dtype=float),
                             "zone": pd.Series(dtype=str), "count": pd.Series(dtype=np.int64)})
    n_bins = int(np.ceil(HIST_MAX_MGDL / bin_mgdl))
    i = np.clip(np.floor(r / bin_mgdl), 0, n_bins - 1).astype(np.int64)
    j = np.clip(np.floor(p / bin_mgdl), 0, n_bins - 1).astype(np.int64)
    z = clarke_zone_codes(r, p).astype(np.int64)
    key = (i * n_bins + j) * 5 + z
    cells, counts = np.unique(key, return_counts=True)
    z, ij = cells % 5, cells // 5
    return pd.DataFrame({"ref_bin": (ij // n_bins) * bin_mgdl, "pred_bin": (ij % n_bins) * bin_mgdl,
                         "zone": np.asarray(ZONES)[z], "count": counts.astype(np.int64)})


@dataclass
class ClarkeAccumulator:
    """Histogram and point sample per figure panel, mergeable across workers.

    ``keys`` names the columns that identify a panel (e.g. ["model",
    "horizon_min", "dataset", "group"]). ``add`` takes the key values as a
    tuple and the pairs of that call; ``merge`` adds another accumulator (from
    another worker). ``histogram()``, ``zones()`` and ``sample()`` return the
    tables to save (CSV): ``clarke_histogram.csv``, ``clarke_zones.csv`` and
    ``clarke_points_sample.csv``.
    """

    keys: list[str]
    bin_mgdl: float = 5.0
    sample_size: int = 20000
    seed: int = 20261026
    stream: int = 0  # give each worker or patient its own stream so that merged samples stay independent
    _hist: dict = field(default_factory=dict)
    _sample: dict = field(default_factory=dict)
    _calls: int = 0

    def add(self, key: tuple, reference, forecast) -> None:
        key = tuple(key)
        r, p = _finite_pairs(reference, forecast)
        if len(r) == 0:
            return
        h = clarke_histogram(r, p, self.bin_mgdl)
        if key in self._hist:
            h = pd.concat([self._hist[key], h]).groupby(["ref_bin", "pred_bin", "zone"], as_index=False)["count"].sum()
        self._hist[key] = h
        if self.sample_size > 0:
            rng = np.random.default_rng([self.seed, self.stream, self._calls])
            self._calls += 1
            self._add_sample(key, r, p, rng.random(len(r)))

    def _add_sample(self, key: tuple, r: np.ndarray, p: np.ndarray, priority: np.ndarray) -> None:
        if key in self._sample:
            old = self._sample[key]
            r, p, priority = (np.concatenate([old[0], r]), np.concatenate([old[1], p]),
                              np.concatenate([old[2], priority]))
        if len(r) > self.sample_size:
            keep = np.argpartition(priority, self.sample_size)[:self.sample_size]
            r, p, priority = r[keep], p[keep], priority[keep]
        self._sample[key] = (r, p, priority)

    def merge(self, other: "ClarkeAccumulator") -> None:
        for key, h in other._hist.items():
            if key in self._hist:
                h = pd.concat([self._hist[key], h]).groupby(["ref_bin", "pred_bin", "zone"], as_index=False)[
                    "count"].sum()
            self._hist[key] = h
        for key, (r, p, priority) in other._sample.items():
            self._add_sample(key, r, p, priority)

    def subset(self, keep) -> "ClarkeAccumulator":
        """A new accumulator with the panels whose key satisfies ``keep(dict of key values)``."""
        out = ClarkeAccumulator(list(self.keys), self.bin_mgdl, self.sample_size, self.seed, self.stream)
        for key, h in self._hist.items():
            if keep(dict(zip(self.keys, key))):
                out._hist[key] = h
                if key in self._sample:
                    out._sample[key] = self._sample[key]
        return out

    def tagged(self, **values) -> "ClarkeAccumulator":
        """A copy whose panels carry more key columns, placed first (e.g. ``cohort=...`` to merge cohorts)."""
        names = list(values)
        out = ClarkeAccumulator(names + list(self.keys), self.bin_mgdl, self.sample_size, self.seed, self.stream)
        prefix = tuple(values[n] for n in names)
        out._hist = {prefix + key: h for key, h in self._hist.items()}
        out._sample = {prefix + key: s for key, s in self._sample.items()}
        return out

    def histogram(self) -> pd.DataFrame:
        frames = [h.assign(**dict(zip(self.keys, key))) for key, h in self._hist.items()]
        if not frames:
            return pd.DataFrame(columns=self.keys + ["ref_bin", "pred_bin", "zone", "count"])
        out = pd.concat(frames, ignore_index=True)
        return out[self.keys + ["ref_bin", "pred_bin", "zone", "count"]].sort_values(
            self.keys + ["ref_bin", "pred_bin"]).reset_index(drop=True)

    def zones(self) -> pd.DataFrame:
        return zone_table(self.histogram(), self.keys)

    def sample(self) -> pd.DataFrame:
        frames = []
        for key, (r, p, priority) in self._sample.items():
            order = np.argsort(priority)
            frames.append(pd.DataFrame({**dict(zip(self.keys, key)), "reference": r[order], "forecast": p[order],
                                        "zone": clarke_zones(r[order], p[order])}))
        if not frames:
            return pd.DataFrame(columns=self.keys + ["reference", "forecast", "zone"])
        return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- figure


def plot_clarke(ax, histogram: pd.DataFrame | None = None, sample: pd.DataFrame | None = None,
                title: str | None = None, bin_mgdl: float = 5.0, max_mgdl: float = PLOT_MAX_MGDL,
                annotate: bool = True):
    """Draw the Clarke Error Grid on ``ax`` from a saved histogram (density) or a sample (scatter).

    ``histogram`` has the columns ``ref_bin``, ``pred_bin``, ``count`` (and
    ``zone``) of one panel; ``sample`` the columns ``reference`` and
    ``forecast``. Pairs beyond ``max_mgdl`` are counted in the zone shares but
    fall outside the drawn square. Returns the image (for a colour bar) or None.
    """
    import matplotlib.colors as mcolors

    from ..plotting import SEQUENTIAL_BLUE, TEXT_PRIMARY, TEXT_SECONDARY

    image = None
    if histogram is not None and not histogram.empty:
        cells = histogram.groupby(["ref_bin", "pred_bin"])["count"].sum()
        n = int(np.ceil(max_mgdl / bin_mgdl))
        grid = np.zeros((n, n))
        i = (cells.index.get_level_values(0).to_numpy() / bin_mgdl).astype(int)
        j = (cells.index.get_level_values(1).to_numpy() / bin_mgdl).astype(int)
        inside = (i < n) & (j < n)
        np.add.at(grid, (j[inside], i[inside]), cells.to_numpy()[inside])
        grid = np.ma.masked_equal(grid, 0)
        cmap = mcolors.LinearSegmentedColormap.from_list("density", SEQUENTIAL_BLUE)
        edges = np.arange(n + 1) * bin_mgdl
        image = ax.pcolormesh(edges, edges, grid, cmap=cmap, norm=mcolors.LogNorm(vmin=1, vmax=max(grid.max(), 1)),
                              shading="flat", rasterized=True)
        shares = histogram.groupby("zone")["count"].sum().reindex(ZONES, fill_value=0)
    elif sample is not None and not sample.empty:
        ax.scatter(sample["reference"], sample["forecast"], s=2, alpha=0.25, color=SEQUENTIAL_BLUE[4], linewidths=0,
                   rasterized=True)
        shares = pd.Series(clarke_zones(sample["reference"], sample["forecast"])).value_counts().reindex(
            ZONES, fill_value=0)
    else:
        shares = None
    for (x0, y0), (x1, y1) in CLARKE_LINES:
        ax.plot([x0, x1], [y0, y1], color=TEXT_PRIMARY, lw=0.9)
    ax.plot([0, max_mgdl], [0, max_mgdl], color=TEXT_SECONDARY, lw=0.6, ls=":")
    if annotate:
        for label, x, y in ZONE_LABELS:
            ax.text(x, y, label, fontsize=11, fontweight="bold", color=TEXT_PRIMARY, ha="center", va="center")
    ax.set_xlim(0, max_mgdl)
    ax.set_ylim(0, max_mgdl)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("reference glucose (mg/dL)")
    ax.set_ylabel("forecast glucose (mg/dL)")
    if shares is not None and shares.sum():
        pct = 100 * shares / shares.sum()
        text = "  ".join(f"{z} {pct[z]:.1f}%" for z in ZONES)
        ax.text(0.0, 1.01, text, transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY, va="bottom")
    if title:
        ax.set_title(title, loc="left", fontsize=10, pad=16)
    return image
