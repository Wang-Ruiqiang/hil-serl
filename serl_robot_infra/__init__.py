import sys as _sys, importlib as _importlib

try:
    _pkg = _importlib.import_module(".denso_env", __name__)
    # 顶级别名
    _sys.modules.setdefault("denso_env", _pkg)
    # 常见子包别名（可按需增减）
    for _sub in ("envs", "camera", "utils", "ros", "robot", "planning"):
        try:
            _sys.modules.setdefault(f"denso_env.{_sub}",
                                    _importlib.import_module(f".denso_env.{_sub}", __name__))
        except Exception:
            pass
    del _pkg, _sub, _sys, _importlib
except Exception:
    # 允许在没有 denso_env 子包时正常导入 serl_robot_infra
    pass