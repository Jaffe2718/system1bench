"""Bounded CPU producers, one CUDA consumer, and memory-driven microbatches."""
import collections
import copy
import gc
import importlib.metadata
import math
import queue
import threading
import time
import traceback


DEFAULTS = dict(initial_batch_rows=8, max_batch_rows=128, max_padded_tokens=32768,
                producer_workers=2, prefetch_cases=64, lookahead_cases=64,
                sample_interval=.25, low_samples=64, grow_below=.70,
                shrink_above=.85, reserve_gib=2.)


def validate_options(options):
    result = dict(DEFAULTS, **options)
    unknown = set(result) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown adaptive settings: " + str(sorted(unknown)))
    for key in ("initial_batch_rows", "max_batch_rows", "max_padded_tokens",
                "producer_workers", "prefetch_cases", "lookahead_cases", "low_samples"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if result["initial_batch_rows"] > result["max_batch_rows"]:
        raise ValueError("initial_batch_rows must not exceed max_batch_rows")
    if not 0 < result["grow_below"] < result["shrink_above"] < 1:
        raise ValueError("Require 0 < grow_below < shrink_above < 1")
    if not math.isfinite(result["sample_interval"]) or result["sample_interval"] <= 0:
        raise ValueError("sample_interval must be finite and positive")
    if not math.isfinite(result["reserve_gib"]) or result["reserve_gib"] < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    return result


class CapturedError(Exception):
    """Preserve diagnostics without retaining CUDA tensors in traceback frames."""
    def __init__(self, exc):
        super().__init__(str(exc))
        self.name = type(exc).__name__
        self.trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.oom = self.name == "OutOfMemoryError" or "cuda out of memory" in str(exc).lower()
        self.fatal_cuda = any(s in str(exc).lower() for s in
                              ("device-side assert", "illegal memory access", "unspecified launch failure"))


def error_name(exc):
    return exc.name if isinstance(exc, CapturedError) else type(exc).__name__


def error_trace(exc):
    return exc.trace if isinstance(exc, CapturedError) else "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__))


class Controller:
    def __init__(self, options, event=lambda *args, **kwargs: None):
        self.options = options
        self.capacity = options["initial_batch_rows"]
        self.low_count = 0
        self.cooldown = 0
        self.lock = threading.Lock()
        self.event = event

    def limit(self):
        with self.lock:
            return self.capacity

    def pressure(self, reason):
        with self.lock:
            old = self.capacity
            self.capacity = max(1, old // 2)
            self.low_count = 0
            self.cooldown = self.options["low_samples"]
            new = self.capacity
        self.event("shrink", reason=reason, previous=old, batch_rows=new)

    def observe(self, sample):
        o = self.options
        if (sample["ram_ratio"] >= o["shrink_above"] or sample["gpu_ratio"] >= o["shrink_above"]
                or sample["gpu_free"] < o["reserve_gib"] * 2**30):
            self.pressure("memory_pressure")
            return
        with self.lock:
            if self.cooldown:
                self.cooldown -= 1
                self.low_count = 0
                return
            low = (sample["ram_ratio"] < o["grow_below"] and sample["gpu_ratio"] < o["grow_below"]
                   and (sample["completed_batches"] > 0 or sample.get("forward_inflight", False)) and sample["backlog"] > 0)
            self.low_count = self.low_count + 1 if low else 0
            if self.low_count < o["low_samples"]:
                return
            old = self.capacity
            self.capacity = min(o["max_batch_rows"], max(old + 1, math.ceil(old * 1.5)))
            self.low_count = 0
            new = self.capacity
        if new != old:
            self.event("grow", previous=old, batch_rows=new)


class CudaMeter:
    def __init__(self, device):
        import psutil
        import torch
        self.torch, self.psutil, self.device = torch, psutil, device

    def sample(self):
        free, total = self.torch.cuda.mem_get_info(self.device)
        return dict(ram_ratio=self.psutil.virtual_memory().percent / 100,
                    gpu_free=free, gpu_total=total, gpu_used=total - free,
                    gpu_allocated=self.torch.cuda.memory_allocated(self.device),
                    gpu_reserved=self.torch.cuda.memory_reserved(self.device))


class Monitor:
    def __init__(self, meter, controller, options, backlog, event):
        self.meter, self.controller, self.options = meter, controller, options
        self.backlog, self.event = backlog, event
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peak = 0
        self.batches = 0
        self.requests = 0
        self.inflight = False
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def begin_forward(self):
        with self.lock:
            self.inflight = True

    def end_forward(self):
        with self.lock:
            self.inflight = False

    def completed(self, peak, requests):
        with self.lock:
            self.peak = max(self.peak, peak)
            self.batches += 1
            self.requests += requests

    def poll(self):
        sample = self.meter.sample()
        with self.lock:
            peak, batches, requests, inflight = self.peak, self.batches, self.requests, self.inflight
            self.peak = self.batches = self.requests = 0
        external = max(0, sample["gpu_used"] - sample["gpu_reserved"])
        sample.update(gpu_ratio=min(1., max(sample["gpu_used"], external + peak) / sample["gpu_total"]),
                      peak_reserved=peak, completed_batches=batches, completed_requests=requests, forward_inflight=inflight,
                      backlog=self.backlog(), batch_rows=self.controller.limit(),
                      elapsed=time.monotonic() - self.started)
        self.event("resource_sample", **sample)
        self.controller.observe(sample)
        return sample

    def _loop(self):
        interval = self.options["sample_interval"]
        deadline = time.monotonic() + interval
        while not self.stop.wait(max(0., deadline - time.monotonic())):
            try:
                self.poll()
            except Exception as exc:
                # A broken monitor must never authorize blind growth.
                self.controller.pressure("monitor_error")
                self.event("monitor_error", error=type(exc).__name__, message=str(exc))
            deadline += interval
            if deadline < time.monotonic():
                # Do not burst several stale samples after a delayed CUDA query.
                deadline = time.monotonic() + interval
                with self.controller.lock:
                    self.controller.low_count = 0

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.options["sample_interval"] + .5)


class LayaEngine:
    """Pinned Laya encoder/decode semantics with strict CUDA OOM propagation."""
    def __init__(self, adapter, budget):
        import torch
        from laya.common import collate_items
        if importlib.metadata.version("laya") != "0.3.20":
            raise RuntimeError("Adaptive Laya path requires the pinned laya==0.3.20")
        self.adapter, self.agent, self.budget = adapter, adapter.agent, budget
        if self.agent.device.type != "cuda" or self.agent._fast is not None:
            raise RuntimeError("Adaptive path requires CUDA and the standard (non-fast) Laya backend")
        self.torch, self.collate = torch, collate_items
        self.local = threading.local()
        self.device = self.agent.device
        self.meter = CudaMeter(self.device)

    def prepare(self, row, worker):
        if not hasattr(self.local, "adapter"):
            # Model weights are shared, but each CPU producer owns its tokenizer/cache.
            adapter = copy.copy(self.adapter)
            adapter.agent = copy.copy(self.agent)
            adapter.agent.tok = copy.deepcopy(self.agent.tok)
            adapter._head_audit_cache = {}
            self.local.adapter = adapter
        adapter = self.local.adapter
        started = time.perf_counter()
        audit = adapter.audit(row["state"], row["questions"], self.budget)
        ids = list(row["questions"])
        for qid in ids:
            adapter.agent._check_question(qid, row["questions"][qid])
        internal = {qid: adapter.agent._to_internal(row["questions"][qid]) for qid in ids}
        items = adapter.agent._encode_state(row["state"], ids, internal, **self.budget)
        return dict(row=row, audit=audit, ids=ids, internal=internal, items=items,
                    rows=len(items), length=max(len(i["ids"]) for i in items), worker=worker,
                    prepare_seconds=time.perf_counter() - started)

    def forward(self, prepared):
        t, agent = self.torch, self.agent
        batch = self.collate([p["items"] for p in prepared], agent.tok.pad_token_id)
        t.cuda.synchronize(self.device)
        t.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        # Match the pinned Agent._infer forward, but let OOM reach our splitter.
        # The upstream method catches OOM and moves the model to CPU permanently.
        with t.inference_mode(), t.autocast(device_type="cuda", dtype=agent.dtype,
                                           enabled=agent._amp_enabled_for(batch["input_ids"].shape[0])):
            logits, act = agent.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
                                      batch["marker_pos"].to(self.device), batch["marker_mask"].to(self.device),
                                      batch["qtype"].to(self.device))
        logits = logits.float().cpu().numpy()
        act = t.softmax(act.float(), -1).cpu().numpy()
        t.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        outputs, offset = [], 0
        for p in prepared:
            answers = agent._decode_answers(logits, act, p["items"], p["ids"], p["internal"], offset,
                                            lang=p["row"]["language"])
            outputs.append(dict(answers=answers))
            offset += p["rows"]
        return outputs, dict(batch_seconds=seconds, gpu_peak_allocated=t.cuda.max_memory_allocated(self.device),
                             gpu_peak_reserved=t.cuda.max_memory_reserved(self.device),
                             gpu_device=str(self.device))

    def recover_oom(self):
        gc.collect()
        self.torch.cuda.empty_cache()


class Producers:
    DONE = object()

    def __init__(self, rows, engine, options):
        self.iterator, self.engine, self.options = iter(rows), engine, options
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.queue = queue.Queue(maxsize=options["prefetch_cases"])
        self.threads = [threading.Thread(target=self._produce, args=(i,), daemon=True)
                        for i in range(options["producer_workers"])]

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _produce(self, worker):
        try:
            while not self.stop.is_set():
                with self.lock:
                    row = next(self.iterator, None)
                if row is None:
                    break
                try:
                    prepared = self.engine.prepare(row, worker)
                except Exception as exc:
                    prepared = dict(row=row, error=CapturedError(exc), stage="prepare", worker=worker)
                self._put(prepared)
        finally:
            self._put(self.DONE)

    def start(self):
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)


def bucket(prepared):
    row = prepared["row"]
    length_bin = 1 << max(0, (prepared["length"] - 1).bit_length())
    return row["panel"], row["k"], row["language"], length_bin


def take_batch(buffer, capacity, token_budget):
    first = buffer.popleft()
    if "error" in first:
        return [first]
    selected, remaining = [first], collections.deque()
    rows, length, key = first["rows"], first["length"], bucket(first)
    while buffer:
        candidate = buffer.popleft()
        fits = ("error" not in candidate and bucket(candidate) == key
                and rows + candidate["rows"] <= capacity
                and (rows + candidate["rows"]) * max(length, candidate["length"]) <= token_budget)
        if fits:
            selected.append(candidate)
            rows += candidate["rows"]
            length = max(length, candidate["length"])
        else:
            remaining.append(candidate)
    buffer.extend(remaining)
    return selected


def collect(rows, adapter, budget, options, event, engine=None, meter=None):
    """Yield one raw result per input; failed batch attempts are split, never dropped."""
    engine = engine or LayaEngine(adapter, budget)
    meter = meter or engine.meter
    options = validate_options(options)
    first_sample = meter.sample()
    free = first_sample["gpu_free"] / 2**30
    options["initial_batch_rows"] = min(options["initial_batch_rows"], 1 if free < 3 else 4 if free < 6 else 8)
    controller = Controller(options, event)
    producers = Producers(rows, engine, options)
    buffer, finished, batch_id = collections.deque(), 0, 0
    monitor = Monitor(meter, controller, options, lambda: producers.queue.qsize() + len(buffer), event)
    event("start", options=options, initial_resources=first_sample)
    producers.start()
    monitor.thread.start()

    def execute(batch, depth=0):
        nonlocal batch_id
        batch_id += 1
        bid = batch_id
        error = None
        monitor.begin_forward()
        try:
            outputs, stats = engine.forward(batch)
            if len(outputs) != len(batch):
                raise ValueError("Heterogeneous batch output count mismatch")
        except Exception as exc:
            error = CapturedError(exc)
        finally:
            monitor.end_forward()
        if error is not None:
            # Outside the except block: traceback frames and activation tensors can be released.
            if error.fatal_cuda:
                raise RuntimeError("Fatal CUDA context error; next model requires a fresh process: " + str(error))
            event("batch_retry", batch_id=bid, case_ids=[p["row"]["id"] for p in batch],
                  error=error.name, message=str(error), oom=error.oom, retry_depth=depth)
            if error.oom:
                controller.pressure("oom")
                engine.recover_oom()
            if len(batch) > 1:
                middle = len(batch) // 2
                yield from execute(batch[:middle], depth + 1)
                yield from execute(batch[middle:], depth + 1)
            else:
                yield dict(prepared=batch[0], output=None, error=error, stage="forward",
                           execution=dict(scheduler="adaptive", batch_id=bid, retry_depth=depth), seconds=0.)
            return
        total_rows = sum(p["rows"] for p in batch)
        length = max(p["length"] for p in batch)
        stats.update(scheduler="adaptive", batch_id=bid, batch_requests=len(batch), batch_rows=total_rows,
                     padded_tokens=total_rows * length, max_tokens=length, retry_depth=depth)
        monitor.completed(stats["gpu_peak_reserved"], len(batch))
        event("batch", case_ids=[p["row"]["id"] for p in batch], **stats)
        for p, output in zip(batch, outputs):
            execution = dict(stats, producer=p["worker"], prepare_seconds=p["prepare_seconds"])
            yield dict(prepared=p, output=output, error=None, execution=execution,
                       seconds=stats["batch_seconds"] * p["rows"] / total_rows)

    try:
        while buffer or finished < len(producers.threads):
            # Bounded lookahead groups visible input lengths, never labels/references.
            while len(buffer) < options["lookahead_cases"] and finished < len(producers.threads):
                try:
                    p = producers.queue.get(timeout=.03 if buffer else .2)
                except queue.Empty:
                    if buffer:
                        break
                    continue
                if p is Producers.DONE:
                    finished += 1
                else:
                    buffer.append(p)
            if not buffer:
                continue
            selected = take_batch(buffer, controller.limit(), options["max_padded_tokens"])
            if "error" in selected[0]:
                yield dict(prepared=selected[0], output=None, error=selected[0]["error"], stage="prepare",
                           execution=dict(scheduler="adaptive", producer=selected[0]["worker"]), seconds=0.)
            else:
                yield from execute(selected)
    finally:
        producers.close()
        monitor.close()
        event("stop", final_batch_rows=controller.limit())
