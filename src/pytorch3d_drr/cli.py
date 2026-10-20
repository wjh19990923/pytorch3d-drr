"""Headless commands with JSON stdout and nonzero failure codes."""
import argparse
import contextlib
import importlib
import importlib.metadata
import json
import platform
import sys


def doctor(device="cpu"):
    report = {"schema_version": 1, "python": platform.python_version(),
              "requested_device": device, "dependencies": {}, "errors": []}
    for distribution, module in (("torch", "torch"), ("pytorch3d", "pytorch3d"),
                                 ("numpy", "numpy"), ("nibabel", "nibabel"), ("numpy-stl", "stl")):
        try:
            importlib.import_module(module)
            report["dependencies"][distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            report["errors"].append(f"{distribution}: {type(exc).__name__}: {exc}")
    try:
        importlib.import_module("pytorch3d._C")
    except Exception as exc:
        report["errors"].append(f"PyTorch3D compiled extension: {exc}")
    try:
        import torch
        from .utilities import determine_device
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_build"] = torch.version.cuda
        selected = determine_device(device)
        torch.empty(1, device=selected)
        report["actual_device"] = str(selected)
    except Exception as exc:
        report["errors"].append(f"device: {exc}")
    report["ok"] = not report["errors"]
    report["note"] = "Import/allocation checks only; run demo to test rendering."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pytorch3d-drr")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Check imports, compiled extension and device")
    check.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    check.add_argument("--json", action="store_true", help="JSON stdout (also default)")
    demo = commands.add_parser("demo", help="Render synthetic CT and mesh without downloads")
    demo.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    demo.add_argument("--resolution", type=int, default=32)
    demo.add_argument("--output", default="outputs/demo")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "doctor":
                result = doctor(args.device)
            else:
                from .demo import run_demo
                result = run_demo(args.output, args.device, args.resolution)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    except Exception as exc:
        print(json.dumps({"schema_version": 1, "ok": False,
                          "error": f"{type(exc).__name__}: {exc}"}))
        return 1
