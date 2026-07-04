"""Resolve data file locations for nested repo layout or flat Kaggle datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def has_international_data(root: Path) -> bool:
    return (
        (root / "results.csv").exists()
        or (root / "raw" / "results.csv").exists()
    )


def _flat_data_score(root: Path) -> int:
    names = (
        "results.csv",
        "wc2026_results.csv",
        "former_names.csv",
        "fixtures.csv",
        "match_stats.csv",
        "understat_matches.parquet",
    )
    return sum(1 for name in names if (root / name).exists())


def _kaggle_input_candidates(kaggle_input: Path) -> list[Path]:
    """Collect dataset roots that may hold flat or nested match files."""
    candidates: list[Path] = []
    if not kaggle_input.is_dir():
        return candidates

    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    for results_csv in kaggle_input.rglob("results.csv"):
        add(results_csv.parent)
    for results_csv in kaggle_input.rglob("raw/results.csv"):
        add(results_csv.parent.parent)

    for item in sorted(kaggle_input.iterdir()):
        if item.is_dir():
            add(item)
            for sub in sorted(item.iterdir()):
                if sub.is_dir():
                    add(sub)
                    for nested in sorted(sub.iterdir()):
                        if nested.is_dir():
                            add(nested)

    return candidates


def find_kaggle_data_root(project_root: Path) -> Path | None:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return None

    preferred = (
        kaggle_input / "soccer-data",
        kaggle_input / "soccer-data-dataset",
    )
    for path in preferred:
        if path.is_dir() and _flat_data_score(path) >= 3:
            return path

    ranked = sorted(
        (( _flat_data_score(path), path) for path in _kaggle_input_candidates(kaggle_input)),
        key=lambda item: (-item[0], str(item[1])),
    )
    preferred_suffixes = ("soccer-data", "soccer-data-dataset")
    for score, path in ranked:
        if score >= 3:
            if path.name in preferred_suffixes or any(
                part in preferred_suffixes for part in path.parts
            ):
                return path
    for score, path in ranked:
        if score >= 3 or has_international_data(path):
            return path

    nested = project_root / "data"
    if has_international_data(nested):
        return nested
    return None


def describe_kaggle_input() -> str:
    """Human-readable listing of /kaggle/input for error messages."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return "/kaggle/input does not exist (not running on Kaggle?)"
    lines = ["/kaggle/input contents:"]
    if not any(kaggle_input.iterdir()):
        lines.append("  (empty — add the soccer-data dataset via Add Data)")
        return "\n".join(lines)
    for item in sorted(kaggle_input.iterdir()):
        if item.is_file():
            lines.append(f"  {item.name}")
            continue
        files = sorted(p.name for p in item.iterdir() if p.is_file())
        lines.append(f"  {item.name}/ ({len(files)} files)")
        for name in files[:12]:
            lines.append(f"    - {name}")
        if len(files) > 12:
            lines.append(f"    ... +{len(files) - 12} more")
    return "\n".join(lines)


@dataclass(frozen=True)
class DataLayout:
    """Maps logical data files to paths for flat or nested layouts."""

    data_root: Path
    project_root: Path

    @classmethod
    def resolve(cls, project_root: Path, data_root_override: Path | None = None) -> DataLayout:
        if data_root_override is not None:
            return cls(data_root=data_root_override, project_root=project_root)
        kaggle_root = find_kaggle_data_root(project_root)
        if kaggle_root is not None:
            return cls(data_root=kaggle_root, project_root=project_root)
        return cls(data_root=project_root / "data", project_root=project_root)

    def results_csv(self) -> Path:
        path = _first_existing(
            self.data_root / "results.csv",
            self.data_root / "raw" / "results.csv",
            self.project_root / "data" / "raw" / "results.csv",
        )
        if path is None:
            raise FileNotFoundError(
                f"results.csv not found under {self.data_root} or {self.project_root / 'data'}"
            )
        return path

    def wc2026_results_csv(self) -> Path:
        path = _first_existing(
            self.data_root / "wc2026_results.csv",
            self.data_root / "raw" / "wc2026_results.csv",
            self.project_root / "data" / "raw" / "wc2026_results.csv",
        )
        if path is None:
            raise FileNotFoundError(
                f"wc2026_results.csv not found under {self.data_root} or project data/"
            )
        return path

    def former_names_csv(self) -> Path:
        path = _first_existing(
            self.data_root / "former_names.csv",
            self.data_root / "raw" / "former_names.csv",
            self.project_root / "data" / "raw" / "former_names.csv",
        )
        if path is None:
            raise FileNotFoundError(
                f"former_names.csv not found under {self.data_root} or project data/"
            )
        return path

    def understat_parquet(self) -> Path | None:
        return _first_existing(
            self.data_root / "understat_matches.parquet",
            self.data_root / "processed" / "understat_matches.parquet",
            self.project_root / "data" / "processed" / "understat_matches.parquet",
        )

    def club_dir(self) -> Path | None:
        flat_dir = self.data_root
        if (flat_dir / "fixtures.csv").exists() and (flat_dir / "match_stats.csv").exists():
            return flat_dir
        nested = self.data_root / "raw" / "club"
        if (nested / "fixtures.csv").exists() and (nested / "match_stats.csv").exists():
            return nested
        project_club = self.project_root / "data" / "raw" / "club"
        if (project_club / "fixtures.csv").exists() and (
            project_club / "match_stats.csv"
        ).exists():
            return project_club
        return None

    def has_club_data(self) -> bool:
        return self.understat_parquet() is not None and self.club_dir() is not None
