#!/usr/bin/env python3
"""Reserve VRAM from the SPLADE indexer for other GPU tasks.

The SPLADE indexer checks a control file between batches and dynamically
reduces its batch size + clears GPU cache when a reservation is active.

Usage:
    # Reserve 4 GB of VRAM
    python scripts/vram_reserve.py 4

    # Reserve 2.5 GB
    python scripts/vram_reserve.py 2.5

    # Clear reservation (indexer resumes full speed)
    python scripts/vram_reserve.py --clear

    # Show current reservation and GPU status
    python scripts/vram_reserve.py --status
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

VRAM_RESERVE_FILE = Path(__file__).parent.parent / "data" / "vram_reserve.json"


def set_reservation(reserve_gb: float) -> None:
    VRAM_RESERVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "active": True,
        "reserve_gb": reserve_gb,
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
    }
    VRAM_RESERVE_FILE.write_text(json.dumps(data, indent=2))
    print(f"Reserved {reserve_gb:.1f} GB VRAM. Indexer will reduce batch size on next batch.")


def clear_reservation() -> None:
    if VRAM_RESERVE_FILE.exists():
        VRAM_RESERVE_FILE.unlink()
        print("VRAM reservation cleared. Indexer will restore full batch size.")
    else:
        print("No active reservation.")


def show_status() -> None:
    if VRAM_RESERVE_FILE.exists():
        data = json.loads(VRAM_RESERVE_FILE.read_text())
        if data.get("active"):
            print(f"Active reservation: {data['reserve_gb']:.1f} GB (since {data.get('requested_at', '?')})")
        else:
            print("Reservation file exists but is inactive.")
    else:
        print("No active reservation.")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_mem / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\nGPU: {props.name}")
            print(f"  Total:     {total:.1f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Free:      {total - allocated:.1f} GB")
        else:
            print("\nNo CUDA GPU available.")
    except ImportError:
        print("\nPyTorch not available — cannot show GPU stats.")


def main():
    parser = argparse.ArgumentParser(
        description="Reserve VRAM from the SPLADE indexer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "gb", type=float, nargs="?",
        help="Amount of VRAM to reserve in GB (e.g., 4, 2.5)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear the VRAM reservation",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current reservation and GPU status",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.clear:
        clear_reservation()
    elif args.gb is not None:
        if args.gb <= 0:
            print("Error: reservation must be > 0 GB", file=sys.stderr)
            sys.exit(1)
        set_reservation(args.gb)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
