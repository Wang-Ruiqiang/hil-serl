import ctypes

libpath = "/home/wrq/workspaces/virtual_env/serl_env/lib/python3.11/site-packages/nvidia/cu13/lib/libcublasLt.so.13"
lib = ctypes.CDLL(libpath)

cublasGetVersion = lib.cublasGetVersion
cublasGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
cublasGetVersion.restype = ctypes.c_int

v = ctypes.c_int(-1)
ret = cublasGetVersion(ctypes.byref(v))
print("cublasGetVersion ret =", ret, "version =", v.value)
