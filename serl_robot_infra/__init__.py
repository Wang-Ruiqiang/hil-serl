import sys as _sys, importlib as _importlib

try:
    _pkg = _importlib.import_module(".franka_env", __name__)
    # 顶级别名
    _sys.modules.setdefault("franka_env", _pkg)
    # 常见子包别名（可按需增减）
    for _sub in ("envs", "camera", "utils", "ros", "robot", "planning"):
        try:
            _sys.modules.setdefault(f"franka_env.{_sub}",
                                    _importlib.import_module(f".franka_env.{_sub}", __name__))
        except Exception:
            pass
    del _pkg, _sub, _sys, _importlib
except Exception:
    # 允许在没有 franka_env 子包时正常导入 serl_robot_infra
    pass