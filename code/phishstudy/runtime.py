"""Process-level runtime settings for long CPU jobs on Windows laptops.

Windows may apply execution-speed power throttling (EcoQoS) to long-running
background processes, which moves them to low-power cores. A process can opt
itself out through SetProcessInformation(ProcessPowerThrottling); nothing
system-wide is changed.
"""
from __future__ import annotations

import ctypes
import os
import sys


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def disable_power_throttling(pid: int | None = None) -> bool:
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_SET_INFORMATION = 0x0200
    ProcessPowerThrottling = 4
    handle = kernel32.OpenProcess(PROCESS_SET_INFORMATION, False, pid or os.getpid())
    if not handle:
        return False
    try:
        state = _PowerThrottlingState(1, 0x1, 0x0)  # control execution speed, state: not throttled
        ok = kernel32.SetProcessInformation(handle, ProcessPowerThrottling, ctypes.byref(state),
                                            ctypes.sizeof(state))
        return bool(ok)
    finally:
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, disable_power_throttling(int(arg)))
