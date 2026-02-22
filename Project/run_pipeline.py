#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RadiovLLMjection - Automated Pipeline Orchestrator

Runs the full experiment pipeline:
  1. Image creation  (raw, injection, stealth)
  2. Baseline test   (naive prompt on raw images)
  3. Injection test   (naive prompt on injected images)
  4. Stealth test     (naive prompt + OCR on stealth images)
  5. Immune test      (immune prompt on injected images)
  6. Stealth immune   (immune prompt on stealth images)

Usage:
  python run_pipeline.py              # Run everything
  python run_pipeline.py --step 2     # Run only step 2 (baseline)
  python run_pipeline.py --step 3-5   # Run steps 3 through 5
  python run_pipeline.py --skip-images # Skip image creation
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="RadiovLLMjection Pipeline")
    parser.add_argument(
        "--step", type=str, default=None,
        help="Run specific step(s): '2' or '3-5'",
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip image creation steps (1)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test mode (10 images only)",
    )
    return parser.parse_args()


def parse_steps(step_str: str):
    """Parse step argument like '2' or '3-5' into a set of ints."""
    if "-" in step_str:
        start, end = step_str.split("-", 1)
        return set(range(int(start), int(end) + 1))
    return {int(step_str)}


def step_1_images():
    """Create all image variants."""
    print("\n" + "=" * 70)
    print("STEP 1: IMAGE CREATION")
    print("=" * 70)

    print("\n--- 1a. Raw images ---")
    from image_creators.create_raw import create_raw_images
    create_raw_images()

    print("\n--- 1b. Visible injection images ---")
    from image_creators.create_injections import create_all_injections
    create_all_injections()

    print("\n--- 1c. Stealth injection images ---")
    from image_creators.create_stealth import create_all_stealth
    create_all_stealth()


def step_2_baseline():
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE (Ground Truth)")
    print("=" * 70)
    from experiments.run_baseline import main as run_baseline
    run_baseline()


def step_3_injection_naive():
    print("\n" + "=" * 70)
    print("STEP 3: INJECTION NAIVE")
    print("=" * 70)
    from experiments.run_injection_naive import main as run_inj
    run_inj()


def step_4_stealth_naive():
    print("\n" + "=" * 70)
    print("STEP 4: STEALTH INJECTION NAIVE (+ OCR)")
    print("=" * 70)
    from experiments.run_stealth_naive import main as run_stealth
    run_stealth()


def step_5_immune():
    print("\n" + "=" * 70)
    print("STEP 5: INJECTION IMMUNE")
    print("=" * 70)
    from experiments.run_immune import main as run_immune
    run_immune()


def step_6_stealth_immune():
    print("\n" + "=" * 70)
    print("STEP 6: STEALTH INJECTION IMMUNE")
    print("=" * 70)
    from experiments.run_stealth_immune import main as run_si
    run_si()


STEPS = {
    1: ("Image Creation", step_1_images),
    2: ("Baseline (GT)", step_2_baseline),
    3: ("Injection Naive", step_3_injection_naive),
    4: ("Stealth Naive + OCR", step_4_stealth_naive),
    5: ("Injection Immune", step_5_immune),
    6: ("Stealth Immune", step_6_stealth_immune),
}


def main():
    args = parse_args()

    # Apply smoke test flag
    if args.smoke:
        import config
        config.SMOKE_TEST = True

    # Determine which steps to run
    if args.step:
        steps_to_run = parse_steps(args.step)
    elif args.skip_images:
        steps_to_run = {2, 3, 4, 5, 6}
    else:
        steps_to_run = set(STEPS.keys())

    print("=" * 70)
    print("RadiovLLMjection - Automated Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Steps  : {sorted(steps_to_run)}")
    if args.smoke:
        print("Mode   : SMOKE TEST (10 images)")
    print("=" * 70)

    for step_num in sorted(steps_to_run):
        if step_num not in STEPS:
            print(f"Unknown step: {step_num}")
            continue
        name, func = STEPS[step_num]
        try:
            func()
        except Exception as e:
            print(f"\nERROR in step {step_num} ({name}): {e}")
            print("Continuing with next step...")
            continue

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE - {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
