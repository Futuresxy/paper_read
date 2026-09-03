#!/usr/bin/env python3
"""Validate and publish one BlaBlaPaper output bundle into Quartz content."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORTS = (
    ("paper_notes.md", "技术解析"),
    ("ELI5_notes.md", "通俗讲解"),
    ("figs_notes.md", "图表详解"),
    ("translation_notes.md", "原文翻译"),
)

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_tags(value: str) -> list[str]:
    tags: list[str] = []
    for item in (part.strip() for part in value.split(",")):
        if item and item not in tags:
            tags.append(item)
    if "paper" not in tags:
        tags.insert(0, "paper")
    return tags


def _yaml_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _validate_image_links(source: Path) -> None:
    missing: list[str] = []
    for report_name, _ in REPORTS:
        report_path = source / report_name
        text = report_path.read_text(encoding="utf-8")
        if not text.strip():
            raise BundleError(f"报告为空: {report_path}")
        for raw_target in IMAGE_PATTERN.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "data:")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if target and not (source / target).is_file():
                missing.append(f"{report_name}: {raw_target}")
    if missing:
        details = "\n  - ".join(missing[:20])
        raise BundleError(f"存在找不到的本地图片引用:\n  - {details}")


def _render_index(info: dict[str, Any], tags: list[str]) -> str:
    title = info.get("paper_title") or info.get("title") or info.get("index")
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    description = info.get("description") or ""

    lines = ["---", f"title: {_yaml_string(title)}"]
    if description:
        lines.append(f"description: {_yaml_string(description)}")
    lines.append("tags:")
    lines.extend(f"  - {_yaml_string(tag)}" for tag in tags)

    authors = metadata.get("authors")
    if isinstance(authors, list) and authors:
        lines.append("authors:")
        lines.extend(f"  - {_yaml_string(author)}" for author in authors)
    for key in ("venue", "year", "doi"):
        value = metadata.get(key)
        if value not in (None, ""):
            lines.append(f"{key}: {_yaml_string(value)}")

    lines.extend(["---", "", f"# {title}", ""])
    if description:
        lines.extend([description, ""])
    lines.append("## 阅读入口")
    lines.append("")
    for report_name, label in REPORTS:
        lines.append(f"- [{label}]({report_name})")
    lines.append("")
    return "\n".join(lines)


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

    missing_reports = [name for name, _ in REPORTS if not (source / name).is_file()]
    if missing_reports:
        raise BundleError("论文流水线没有完整生成四份报告: " + ", ".join(missing_reports))
    if not (source / "images").is_dir():
        raise BundleError(f"缺少图片目录: {source / 'images'}")
    _validate_image_links(source)

    destination = content_root / _safe_collection(collection) / slug
    destination.mkdir(parents=True, exist_ok=True)

    for report_name, _ in REPORTS:
        shutil.copy2(source / report_name, destination / report_name)
    shutil.copytree(source / "images", destination / "images", dirs_exist_ok=True)

    tags = _parse_tags(tags_value)
    (destination / "index.md").write_text(_render_index(info, tags), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "slug": slug,
        "paper_title": info.get("paper_title") or info.get("title") or slug,
        "metadata": info.get("metadata") or {},
        "description": info.get("description") or "",
        "collection": collection,
        "tags": tags,
        "source_sha256": source_sha256,
        "generator": "BlaBlaPaper",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "reports": [name for name, _ in REPORTS],
        "report_sha256": {
            name: _sha256(source / name)
            for name, _ in REPORTS
        },
    }
    (destination / "paper.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
