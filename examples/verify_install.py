#!/usr/bin/env python3
"""Validate the DexTacHil Python environment without touching robot hardware.

Run this from the repo root with the intended virtualenv, for example:

    /home/ealin/workspaces/DexTacHil/hil_env/bin/python examples/verify_install.py --run-update

The script checks imports and lightweight CUDA/JAX/Torch paths used by the
learner/actor. It deliberately avoids instantiating the real Franka env, opening
RealSense cameras, or connecting to ROS services.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import Callable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
SERL_LAUNCHER_DIR = REPO_ROOT / "serl_launcher"
SERL_ROBOT_INFRA_DIR = REPO_ROOT / "serl_robot_infra"


def configure_paths() -> None:
    for path in (REPO_ROOT, EXAMPLES_DIR, SERL_LAUNCHER_DIR, SERL_ROBOT_INFRA_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


class Checker:
    def __init__(self, traceback_enabled: bool = False) -> None:
        self.traceback_enabled = traceback_enabled
        self.results: list[CheckResult] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, "OK", detail))
        print(f"[OK]   {name}{': ' + detail if detail else ''}")

    def warn(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, "WARN", detail))
        print(f"[WARN] {name}{': ' + detail if detail else ''}")

    def fail(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, "FAIL", detail))
        print(f"[FAIL] {name}{': ' + detail if detail else ''}")

    def run(self, name: str, fn: Callable[[], str | None], warn_only: bool = False) -> None:
        try:
            detail = fn() or ""
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic script
            if self.traceback_enabled:
                traceback.print_exc()
            detail = f"{type(exc).__name__}: {exc}"
            if warn_only:
                self.warn(name, detail)
            else:
                self.fail(name, detail)
            return
        self.ok(name, detail)

    def has_failures(self) -> bool:
        return any(result.status == "FAIL" for result in self.results)

    def summary(self) -> str:
        counts = {
            "OK": sum(result.status == "OK" for result in self.results),
            "WARN": sum(result.status == "WARN" for result in self.results),
            "FAIL": sum(result.status == "FAIL" for result in self.results),
        }
        return f"OK={counts['OK']} WARN={counts['WARN']} FAIL={counts['FAIL']}"


def import_module(module: str) -> str:
    mod = importlib.import_module(module)
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            version = metadata.version(module.split(".")[0])
        except metadata.PackageNotFoundError:
            version = "imported"
    return str(version)


def package_version(distribution: str) -> str:
    return metadata.version(distribution)


def check_python() -> str:
    return f"{sys.version.split()[0]} at {sys.executable}; {platform.platform()}"


def check_nvidia_smi() -> str:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi not found on PATH")
    proc = subprocess.run(
        [
            exe,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip()


def check_jax(require_gpu: bool, run_update: bool) -> str:
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    backends = sorted({device.platform for device in devices})
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if require_gpu and not gpu_devices:
        raise RuntimeError(f"JAX sees no GPU devices; devices={devices}")
    x = jnp.arange(8.0)
    y = jax.jit(lambda value: value * 2.0 + 1.0)(x).block_until_ready()
    if float(y[-1]) != 15.0:
        raise RuntimeError("unexpected JAX compute result")
    if run_update:
        _run_agent_smoke()
    return f"jax={jax.__version__}; jaxlib={package_version('jaxlib')}; backends={backends}; gpu_count={len(gpu_devices)}"


def check_torch(require_gpu: bool) -> str:
    import torch

    cuda_available = torch.cuda.is_available()
    if require_gpu and not cuda_available:
        raise RuntimeError("torch.cuda.is_available() is False")
    device_name = torch.cuda.get_device_name(0) if cuda_available else "no cuda"
    cuda_version = getattr(torch.version, "cuda", None)
    x = torch.ones((4,), device="cuda" if cuda_available else "cpu")
    y = (x + 2).sum().item()
    if y != 12.0:
        raise RuntimeError("unexpected Torch compute result")
    return f"torch={torch.__version__}; torch_cuda={cuda_version}; gpu={device_name}"


def _run_agent_smoke() -> None:
    """Create the gaze SAC agent with synthetic observations.

    This exercises the Flax/JAX model path used by learner startup, including
    the front_camera/front_camera_mask pairing. ResNet weight download is
    bypassed because this is an install check, not a network fetch.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from serl_launcher.utils import launcher
    from serl_launcher.utils import train_utils

    def no_op_load_resnet10_params(agent, image_keys=("image",), public=True):
        return agent

    train_utils.load_resnet10_params = no_op_load_resnet10_params

    sample_obs = {
        "state": np.zeros((32,), dtype=np.float32),
        "front_camera": np.zeros((128, 128, 3), dtype=np.uint8),
        "tactile_data": np.zeros((128, 128, 3), dtype=np.uint8),
        "front_camera_mask": np.zeros((128, 128, 3), dtype=np.uint8),
        "gaze_heatmap": np.zeros((128, 128), dtype=np.float32),
    }
    sample_action = np.zeros((17,), dtype=np.float32)

    agent = launcher.make_gaze_sac_pixel_agent_hybrid_single_arm(
        seed=0,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=("front_camera", "tactile_data", "front_camera_mask"),
        encoder_type="resnet-pretrained",
        discount=0.97,
    )
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    _ = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent),
        sharding.replicate(),
    )


def check_sam_numpy_metadata() -> str:
    import numpy as np
    import sam3

    sam_version = getattr(sam3, "__version__", metadata.version("sam3"))
    requirements = metadata.requires("sam3") or []
    numpy_reqs = [req for req in requirements if req.lower().startswith("numpy")]
    if numpy_reqs and "<2" in " ".join(numpy_reqs) and np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        raise RuntimeError(
            f"sam3 runtime import works, but package metadata still says {numpy_reqs}; "
            "relax sam3 pyproject.toml if you want pip check to be clean with NumPy 2"
        )
    return f"sam3={sam_version}; numpy={np.__version__}"


def check_pip_check() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        return output or "pip check clean"
    lower = output.lower()
    sam_numpy_only = (
        "sam3" in lower
        and "numpy" in lower
        and "<2" in lower
        and len([line for line in output.splitlines() if line.strip()]) == 1
    )
    if sam_numpy_only:
        raise RuntimeError(output)
    raise RuntimeError(output or f"pip check exited {proc.returncode}")


def check_path(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        count = sum(1 for _ in path.iterdir())
        return f"{path} ({count} entries)"
    return f"{path} ({path.stat().st_size} bytes)"


def check_core_imports(checker: Checker) -> None:
    required = [
        "absl",
        "agentlace",
        "cv2",
        "distrax",
        "flax",
        "gym",
        "gymnasium",
        "imageio",
        "jax",
        "msgpack",
        "natsort",
        "numpy",
        "optax",
        "pyrealsense2",
        "pyspacemouse",
        "scipy",
        "tensorflow",
        "tqdm",
        "wandb",
        "zmq",
    ]
    for module in required:
        checker.run(f"import {module}", lambda module=module: import_module(module))

    repo_modules = [
        "experiments.mappings",
        "serl_launcher.utils.launcher",
        "serl_launcher.utils.gaze_utils",
        "serl_launcher.utils.gaze_mask_utils",
        "serl_launcher.wrappers.gaze_derived_observation",
        "serl_robot_infra.franka_env.envs.franka_env",
    ]
    for module in repo_modules:
        checker.run(f"import {module}", lambda module=module: import_module(module))


def check_actor_hardware_imports(checker: Checker, strict_actor: bool) -> None:
    warn_only = not strict_actor
    modules = [
        "rclpy",
        "geometry_msgs.msg",
        "sensor_msgs.msg",
        "std_msgs.msg",
        "leap_hand.srv",
        "dmrobotics",
        "pynput",
    ]
    for module in modules:
        checker.run(
            f"actor/hardware import {module}",
            lambda module=module: import_module(module),
            warn_only=warn_only,
        )


def check_optional_model_artifacts(checker: Checker, strict_actor: bool) -> None:
    warn_only = not strict_actor
    paths = [
        EXAMPLES_DIR / "gaze_data_process" / "gaze_heatmap_ckpt",
        EXAMPLES_DIR / "gaze_data_process" / "SAM_process" / "mask_predictor_ckpt" / "best.pt",
        EXAMPLES_DIR / "reward_classifier" / "classifier_ckpt_ball_pick",
        EXAMPLES_DIR / "reward_classifier" / "classifier_ckpt_ball_pick_no_tactile",
        EXAMPLES_DIR / "reward_classifier" / "classifier_ckpt_ball_place",
        EXAMPLES_DIR / "reward_classifier" / "classifier_ckpt_ball_place_no_tactile",
    ]
    for path in paths:
        checker.run(
            f"artifact {path.relative_to(REPO_ROOT)}",
            lambda path=path: check_path(path),
            warn_only=warn_only,
        )


def print_external_dependency_notes() -> None:
    print("\nExternal repos/artifacts that are not guaranteed by pip:")
    print("- sam3: install editable from /home/ealin/workspaces/sam3; current Torch wheel should be CUDA 12.x/12.8 compatible.")
    print("- SAM2 is only needed for examples/gaze_data_process/SAM_process/propagate_recorded_sam_masks.py.")
    print("- ROS2 + leap_hand messages are needed by actor/hardware startup; source the ROS workspace before running actor.")
    print("- dmrobotics is needed for DM-TAC tactile sensors when tactile input is enabled.")
    print("- ResNet-10 SERL weights are downloaded to ~/.serl/resnet10_params.pkl unless already cached.")
    print("- Gaze/mask/reward classifier checkpoints listed above are runtime artifacts, not pip packages.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="Fail if JAX or Torch cannot see a CUDA GPU.")
    parser.add_argument("--run-update", action="store_true", help="Also instantiate the gaze SAC agent and device_put it.")
    parser.add_argument("--strict-actor", action="store_true", help="Treat ROS/Leap/DMTac/checkpoint actor items as failures.")
    parser.add_argument("--traceback", action="store_true", help="Print Python tracebacks for failed checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_paths()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hil-serl-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hil-serl-cache")
    os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/hil-serl-cupy-cache")

    checker = Checker(traceback_enabled=args.traceback)
    checker.run("python", check_python)
    checker.run("nvidia-smi", check_nvidia_smi, warn_only=not args.require_gpu)
    checker.run("JAX CUDA smoke", lambda: check_jax(args.require_gpu, args.run_update))
    checker.run("Torch CUDA smoke", lambda: check_torch(args.require_gpu))
    checker.run("SAM3 + NumPy metadata", check_sam_numpy_metadata, warn_only=True)
    checker.run("pip check", check_pip_check, warn_only=True)

    check_core_imports(checker)
    check_actor_hardware_imports(checker, strict_actor=args.strict_actor)
    check_optional_model_artifacts(checker, strict_actor=args.strict_actor)
    print_external_dependency_notes()

    print(f"\nSummary: {checker.summary()}")
    if checker.has_failures():
        print("Environment verification failed. See [FAIL] lines above.")
        return 1
    print("Environment verification completed. Review [WARN] lines before running actor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
