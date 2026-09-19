"""A pool of warm worker processes that run and score CadQuery code with a hard per-task timeout.

Model-written code runs outside the caller's process, so a hang or an OCC segfault costs one worker, which is
replaced. Workers import CadQuery once at startup (1–2 s warm; the very first import on a machine can take ~90 s).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Any


GEOMETRY_MODULE = __name__.rsplit(".", 1)[0] + ".geometry"  # works as cadabra.cad.* and as training/cadcheck.*


def _worker_main(conn, module_name: str) -> None:
    import importlib
    import tempfile

    os.chdir(tempfile.mkdtemp(prefix="cadworker-"))  # any stray file write lands here, not in the repo
    geometry = importlib.import_module(module_name)
    verify = None

    conn.send("ready")
    while True:
        task = conn.recv()
        if task is None:
            break
        try:
            if task.pop("_fn", "evaluate") == "render_score":
                if verify is None:
                    verify = importlib.import_module(module_name.rsplit(".", 1)[0] + ".verify")
                result = verify.render_score(**task)
            else:
                result = geometry.evaluate(**task)
        except Exception as e:  # noqa: BLE001
            result = {"runs": False, "error": f"worker error: {type(e).__name__}: {e}"}
        conn.send(result)


class CadPool:
    def __init__(self, workers: int | None = None, timeout: float = 30.0, ready_timeout: float = 600.0):
        self.workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
        self.timeout = timeout
        self.ready_timeout = ready_timeout
        self._ctx = mp.get_context("spawn")
        self._idle: queue.Queue = queue.Queue()
        starting = [self._start() for _ in range(self.workers)]
        for proc, conn in starting:
            self._idle.put(self._await_ready(proc, conn))

    def _start(self):
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_main, args=(child, GEOMETRY_MODULE), daemon=True)
        proc.start()
        return proc, parent

    def _await_ready(self, proc, conn):
        if not conn.poll(self.ready_timeout) or conn.recv() != "ready":
            proc.kill()
            raise RuntimeError("CAD worker failed to start")
        return proc, conn

    def _replace(self, proc) -> None:
        proc.kill()
        proc.join(timeout=5)
        self._idle.put(self._await_ready(*self._start()))

    def run(self, **task: Any) -> dict:
        """Evaluate one task: run(code=..., gold_code=..., want_mesh=False, want_chamfer=True)."""
        proc, conn = self._idle.get()
        try:
            conn.send(task)
            if conn.poll(self.timeout):
                result = conn.recv()
                self._idle.put((proc, conn))
                return result
            self._replace(proc)
            return {"runs": False, "error": f"timed out after {self.timeout:.0f}s"}
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as e:
            self._replace(proc)
            return {"runs": False, "error": f"worker crashed: {type(e).__name__}"}

    def map(self, tasks: list[dict]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda t: self.run(**t), tasks))

    def close(self) -> None:
        while not self._idle.empty():
            proc, conn = self._idle.get()
            try:
                conn.send(None)
            except OSError:
                pass
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()

    def __enter__(self) -> CadPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
