# pip install psutil
import asyncio
import psutil
import time
import contextlib 
import gc 
import sys

def is_debugging() -> bool:
    """
    Checks if the current Python script is running under a debugger.
    """
    # Check for an active trace function (set by debuggers)
    has_trace = hasattr(sys, 'gettrace') and sys.gettrace() is not None

    # Check for a modified breakpoint hook (often done by debuggers)
    has_breakpoint_hook = sys.breakpointhook.__module__ != "sys"

    return has_trace or has_breakpoint_hook


def _gb(bytes_): 
    return bytes_ / (1024 ** 3)

async def monitor_memory(interval: float = 5.0, pid: int = None) -> None:
    """
    Print memory usage of a process (RSS) in GB every `interval` seconds.
    - interval: seconds between prints
    - pid: process ID to watch (default: current process)
    """
    proc = psutil.Process(pid) if pid is not None else psutil.Process()
    try:
        while True:
            rss = proc.memory_info().rss  # resident set size (bytes)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] RSS: {_gb(rss):.3f} GB")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # graceful shutdown
        pass

def run_with_monitoring(func, *args, **kwargs):
    # check if in debug mode
    if is_debugging():
        return func(*args, **kwargs)
    async def wrapper():
        task = asyncio.create_task(monitor_memory(interval=5.0))  # current process
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
            gc.collect()
            return result
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    asyncio.run(wrapper())
