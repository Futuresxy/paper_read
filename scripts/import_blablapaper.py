#!/usr/bin/env python3
"""Validate and publish one BlaBlaPaper output bundle into Quartz content."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Any


REPORTS = (
    "paper_notes.md",
    "ELI5_notes.md",
    "figs_notes.md",
    "translation_notes.md",
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class BundleError(ValueError):
    """Raised when generated content is incomplete or unsafe to publish."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleError(f"缺少元数据文件: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BundleError(f"元数据不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"元数据根节点必须是对象: {path}")
    return data


def _safe_collection(value: str) -> Path:
    raw = value.strip().strip("/")
    if not raw or "\\" in raw:
        raise BundleError("collection 不能为空，也不能包含反斜杠")
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"不安全的 collection 路径: {value!r}")
    return path


def _prepare_reports(source: Path) -> dict[str, str]:
    available_images = sorted(
        path.relative_to(source).as_posix()
        for path in (source / "images").rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    available_set = set(available_images)
    missing: list[str] = []
    prepared: dict[str, str] = {}

    for report_name in REPORTS:
        report_path = source / report_name
        text = report_path.read_text(encoding="utf-8")
        if not text.strip():
            raise BundleError(f"报告为空: {report_path}")

        def repair_link(match: re.Match[str]) -> str:
            raw_target = match.group(1)
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "data:")):
                return match.group(0)
            target = target.split("#", 1)[0].split("?", 1)[0]
            if target in available_set:
                return match.group(0)

            candidates = get_close_matches(target, available_images, n=2, cutoff=0.96)
            if len(candidates) == 1:
                raw_token = raw_target.strip().split(maxsplit=1)[0]
                corrected = candidates[0]
                if raw_token.startswith("<") and raw_token.endswith(">"):
                    corrected = f"<{corrected}>"
                corrected_target = raw_target.replace(raw_token, corrected, 1)
                return match.group(0).replace(raw_target, corrected_target, 1)

            missing.append(f"{report_name}: {raw_target}")
            return match.group(0)

        prepared[report_name] = IMAGE_PATTERN.sub(repair_link, text)

    if missing:
        details = "\n  - ".join(missing[:20])
        raise BundleError(f"存在找不到的本地图片引用:\n  - {details}")
    return prepared

def import_bundle(
    source: Path,
    content_root: Path,
    collection: str,
    tags_value: str,
    source_sha256: str = "",
) -> Path:
    source = source.resolve()
    content_root = content_root.resolve()
    if not source.is_dir():
        raise BundleError(f"BlaBlaPaper 输出目录不存在: {source}")

    info = _load_json(source / "info.json")
    slug = str(info.get("index") or source.name).strip()
    if not SAFE_SLUG.fullmatch(slug):
        raise BundleError(f"info.json 中的 index 不是安全 URL slug: {slug!r}")

    missing_reports = [name for name in REPORTS if not (source / name).is_file()]
    if missing_reports:
        raise BundleError("论文流水线没有完整生成四份报告: " + ", ".join(missing_reports))
    if not (source / "images").is_dir():
        raise BundleError(f"缺少图片目录: {source / 'images'}")
    prepared_reports = _prepare_reports(source)

    destination = content_root / _safe_collection(collection) / slug
    destination.mkdir(parents=True, exist_ok=True)

    for report_name in REPORTS:
        (destination / report_name).write_text(
            prepared_reports[report_name],
            encoding="utf-8",
        )

    destination_images = destination / "images"
    destination_images.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((source / "images").rglob("*")):
        if image_path.is_symlink():
            raise BundleError(f"图片目录不允许包含符号链接: {image_path}")
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_path = image_path.relative_to(source / "images")
        target_path = destination_images / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_path)

    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 BlaBlaPaper 输出校验并发布到 paper-notes/content",
    )
    parser.add_argument("--source", required=True, type=Path, help="BlaBlaPaper 的单篇输出目录")
    parser.add_argument("--content-root", type=Path, default=Path("content"))
    parser.add_argument("--collection", default="misc", help="站点内分类，例如 ISCA26")
    parser.add_argument("--tags", default="paper", help="逗号分隔的标签")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument(
        "--github-output",
        type=Path,
        help="可选：将 slug 和 destination 追加到 GitHub Actions 输出文件",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        destination = import_bundle(
            source=args.source,
            content_root=args.content_root,
            collection=args.collection,
            tags_value=args.tags,
            source_sha256=args.source_sha256,
        )
    except BundleError as exc:
        print(f"[publish error] {exc}", file=sys.stderr)
        return 1

    print(f"论文内容已发布到: {destination}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"slug={destination.name}\n")
            handle.write(f"destination={destination.as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
