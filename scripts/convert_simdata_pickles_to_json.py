#!/usr/bin/env python3
"""Convert legacy simdata pickle files to test_shape_w_arm JSON format."""

import argparse

from adapteddlo_muj.envs.test_shape_w_arm.base import (
    TEST_SHAPE_DATA_DIR,
    convert_all_legacy_simdata_pickles,
    iter_legacy_simdata_pickles,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy simdata/*.pickle files to test_shape_w_arm JSON."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite JSON files that already exist.",
    )
    args = parser.parse_args()

    saved_paths = convert_all_legacy_simdata_pickles(skip_existing=args.skip_existing)
    print(f"Converted {len(saved_paths)} pickle(s) -> {TEST_SHAPE_DATA_DIR}")
    for path in saved_paths:
        print(f"  {path}")

    legacy_count = len(iter_legacy_simdata_pickles())
    if len(saved_paths) < legacy_count and not args.skip_existing:
        print(
            f"Note: {legacy_count} legacy pickle file(s) found; "
            f"{legacy_count - len(saved_paths)} duplicate case(s) used plugin over normal."
        )


if __name__ == "__main__":
    main()
