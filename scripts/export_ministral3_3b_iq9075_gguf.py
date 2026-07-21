#!/usr/bin/env python3
"""Compile Ministral 3B Q4 GGUF for IQ9075 HTP and export a Genie bundle."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("~/qairt_build/ministral3_3b_q4"),
    )
    parser.add_argument(
        "--target",
        default="dsp_arch:v73;soc_model:77;cores:1",
        help="QAIRT HTP target for Dragonwing IQ9075",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=2,
        help="Number of QNN context binaries in the Genie export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gguf = args.gguf.expanduser().resolve()
    build_root = args.build_root.expanduser().resolve()
    cache_root = build_root / "cache"
    container_dir = build_root / "container"
    export_dir = build_root / "genie"

    if not gguf.is_file():
        raise FileNotFoundError(gguf)
    if container_dir.exists() or export_dir.exists():
        raise FileExistsError(
            f"{container_dir} or {export_dir} already exists; use a new --build-root"
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print(f"CREATE_BUILDER gguf={gguf}", flush=True)
    builder = GenAIBuilderFactory.create(
        str(gguf),
        "HTP",
        cache_root=str(cache_root),
    )
    builder.set_targets([args.target])
    builder.set_transformation_options(
        options={"split.num_splits": args.num_splits}
    )

    build_started = time.monotonic()
    print(
        f"BUILD_START target={args.target} num_splits={args.num_splits}",
        flush=True,
    )
    container = builder.build()
    build_elapsed = time.monotonic() - build_started
    print(f"BUILD_DONE elapsed_s={build_elapsed:.3f}", flush=True)

    save_started = time.monotonic()
    container.save(container_dir)
    save_elapsed = time.monotonic() - save_started
    print(f"SAVE_DONE elapsed_s={save_elapsed:.3f}", flush=True)

    export_started = time.monotonic()
    container.export(export_dir)
    export_elapsed = time.monotonic() - export_started
    print(f"EXPORT_DONE elapsed_s={export_elapsed:.3f}", flush=True)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "gguf": str(gguf),
        "target": args.target,
        "num_splits": args.num_splits,
        "build_elapsed_s": round(build_elapsed, 3),
        "save_elapsed_s": round(save_elapsed, 3),
        "export_elapsed_s": round(export_elapsed, 3),
        "total_elapsed_s": round(time.monotonic() - started, 3),
        "max_rss_kb": usage.ru_maxrss,
        "cache_root": str(cache_root),
        "container_dir": str(container_dir),
        "export_dir": str(export_dir),
    }
    print("EXPORT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
