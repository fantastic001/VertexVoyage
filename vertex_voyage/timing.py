"""
Lightweight timing instrumentation for measuring individual, named parts of code.

A single object, :class:`TimeMetric`, works three ways off the same name:

    with TimeMetric("partitioning"):
        ...

    @TimeMetric("reconstruct")
    def reconstruct(...):
        ...

    tm = TimeMetric("walks")
    tm.start()
    ...
    tm.stop()

Metrics that share a name share one accumulator, so a decorated function that
runs many times aggregates (count, total, mean, stddev, min, max).

When a metric is entered while another is already active on the same thread, its
name is prefixed with the parent's ("parent/child"). Parents then report both
*total* time and *self* time (total minus the time spent inside nested metrics).

Reporting:

    TimeMetric.print_report()   # logs + prints an aggregated table
    TimeMetric.dump()           # JSON-serializable dict, e.g. for PersistedRun

Disable globally with ``VERTEX_VOYAGE_TIME_METRICS=false`` (near-zero overhead:
enter/exit become a flag check and a list append/pop).
"""

import contextlib
import math
import threading
import time
import logging

from vertex_voyage.config import get_config_bool

logger = logging.getLogger("timing")

_SEP = "/"

# --- global state -----------------------------------------------------------

_ENABLED = get_config_bool(
    "time_metrics", True,
    "Whether to collect TimeMetric timings (set false to disable with ~zero overhead)",
)

_records = {}                 # path -> _Record
_records_lock = threading.RLock()
_local = threading.local()    # per-thread stack of _ActiveSpan


def _stack():
    s = getattr(_local, "stack", None)
    if s is None:
        s = []
        _local.stack = s
    return s


class _ActiveSpan:
    __slots__ = ("path", "start", "child_time")

    def __init__(self, path, start):
        self.path = path
        self.start = start
        self.child_time = 0.0


class _Record:
    """Streaming aggregate for one metric path. Guarded by ``_records_lock``."""

    __slots__ = (
        "path", "count", "total", "total_sq", "min", "max", "last",
        "self_total", "self_total_sq",
    )

    def __init__(self, path):
        self.path = path
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min = math.inf
        self.max = 0.0
        self.last = 0.0
        self.self_total = 0.0
        self.self_total_sq = 0.0

    def add(self, elapsed, self_time):
        self.count += 1
        self.total += elapsed
        self.total_sq += elapsed * elapsed
        self.last = elapsed
        if elapsed < self.min:
            self.min = elapsed
        if elapsed > self.max:
            self.max = elapsed
        self.self_total += self_time
        self.self_total_sq += self_time * self_time

    @staticmethod
    def _stddev(n, total, total_sq):
        if n <= 0:
            return 0.0
        mean = total / n
        var = max(0.0, total_sq / n - mean * mean)
        return math.sqrt(var)

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    @property
    def stddev(self):
        return self._stddev(self.count, self.total, self.total_sq)

    @property
    def self_mean(self):
        return self.self_total / self.count if self.count else 0.0

    @property
    def self_stddev(self):
        return self._stddev(self.count, self.self_total, self.self_total_sq)

    def as_dict(self):
        return {
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.min if self.count else 0.0,
            "max": self.max,
            "last": self.last,
            "self_total": self.self_total,
            "self_mean": self.self_mean,
            "self_stddev": self.self_stddev,
        }


def _commit(path, elapsed, self_time):
    with _records_lock:
        rec = _records.get(path)
        if rec is None:
            rec = _Record(path)
            _records[path] = rec
        rec.add(elapsed, self_time)


def _push(name):
    stack = _stack()
    parent = stack[-1].path if stack else None
    path = parent + _SEP + name if parent else name
    stack.append(_ActiveSpan(path, time.perf_counter()))


def _pop():
    stack = _stack()
    if not stack:
        logger.warning("TimeMetric stop/exit with no active metric on this thread")
        return
    span = stack.pop()
    elapsed = time.perf_counter() - span.start
    self_time = elapsed - span.child_time
    if stack:
        stack[-1].child_time += elapsed
    _commit(span.path, elapsed, self_time)


# --- public API ------------------------------------------------------------

class TimeMetric(contextlib.ContextDecorator):
    """A named timing probe usable as a context manager, decorator, or start/stop pair.

    ``name`` is required and identifies the accumulator. Instances are cheap and
    stateless; the same name may be constructed anywhere and will aggregate into
    one record. Nesting on a single thread builds ``"parent/child"`` paths.
    """

    def __init__(self, name):
        if not name or not isinstance(name, str):
            raise ValueError("TimeMetric requires a non-empty string name")
        if _SEP in name:
            raise ValueError(f"TimeMetric name must not contain {_SEP!r}: {name!r}")
        self.name = name

    # context manager / decorator (ContextDecorator turns this into @-usable)
    def __enter__(self):
        if _ENABLED:
            _push(self.name)
        return self

    def __exit__(self, *exc):
        if _ENABLED:
            _pop()
        return False

    # manual, for code that is not lexically scoped
    def start(self):
        if _ENABLED:
            _push(self.name)
        return self

    def stop(self):
        if _ENABLED:
            _pop()

    # -- collection-wide operations ----------------------------------------

    @classmethod
    def dump(cls):
        """Return a JSON-serializable ``{path: {count, total, mean, stddev, ...}}`` dict."""
        with _records_lock:
            return {path: _records[path].as_dict() for path in sorted(_records)}

    @classmethod
    def report_lines(cls):
        data = cls.dump()
        if not data:
            return ["No timing metrics collected."]
        name_w = max(len("metric"), max(len(p) for p in data))
        header = (
            f"{'metric':<{name_w}}  {'count':>7}  {'total(s)':>10}  "
            f"{'mean(s)':>10}  {'stddev(s)':>10}  {'self(s)':>10}"
        )
        lines = ["Timing metrics:", header, "-" * len(header)]
        for path, d in data.items():
            lines.append(
                f"{path:<{name_w}}  {d['count']:>7d}  {d['total']:>10.4f}  "
                f"{d['mean']:>10.6f}  {d['stddev']:>10.6f}  {d['self_total']:>10.4f}"
            )
        return lines

    @classmethod
    def print_report(cls):
        """Log (INFO) and print the aggregated timing table."""
        for line in cls.report_lines():
            logger.info(line)
            print(line, flush=True)

    @classmethod
    def reset(cls):
        """Drop all collected records (and any dangling per-thread state on this thread)."""
        with _records_lock:
            _records.clear()
        _local.stack = []

    @classmethod
    def set_enabled(cls, enabled):
        """Enable/disable collection at runtime (mainly for tests)."""
        global _ENABLED
        _ENABLED = bool(enabled)

    @classmethod
    def is_enabled(cls):
        return _ENABLED


# convenient lowercase alias
time_metric = TimeMetric
