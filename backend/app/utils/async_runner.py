"""
Shared persistent asyncio event loop for Graphiti calls from sync Flask context.

asyncio.run() creates+closes a loop per call. Graphiti internally spawns tasks
(semaphore_gather, etc.) that need the loop to stay alive across the coroutine.
Using a single persistent loop + run_coroutine_threadsafe solves this.
"""
import asyncio
import threading

_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True, name="graphiti-loop")
_loop_thread.start()


def run_async(coro, timeout: float = 120):
    """Submit a coroutine to the shared persistent loop and block until done."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)
