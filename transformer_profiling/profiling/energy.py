"""Read-only GPU telemetry. No power-cap changes and no privileged operations."""
import csv
import gzip
import math
from pathlib import Path
import subprocess
import threading
import time


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def inventory():
    # CUDA resolves scheduler masks, including ordinals remapped inside a job cgroup.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this job. Check the GPU allocation, driver and PyTorch build.")
    devices = []
    for ordinal in range(torch.cuda.device_count()):
        visible_uuid = "GPU-" + str(torch.cuda.get_device_properties(ordinal).uuid)
        result = subprocess.run(["nvidia-smi", "-i", visible_uuid,
                                 "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        row, = csv.reader(result.stdout.splitlines())
        index, uuid, name, memory, driver, pci = (s.strip() for s in row)
        if uuid.lower() != visible_uuid.lower():
            raise RuntimeError("CUDA and telemetry device identities differ")
        devices.append({"index": int(index), "uuid": uuid, "name": name, "memory_bytes": int(memory) * 1024**2,
                        "driver": driver, "pci_bus_id": pci})
    return sorted(devices, key=lambda gpu: gpu["index"])


class Sensor:
    def __init__(self, uuid):
        self.uuid = uuid
        self.nvml = self.handle = None
        self.init_note = ""
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            self.nvml = nvml
            self.handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
            # Fail over if the library exists but power query is unavailable.
            nvml.nvmlDeviceGetPowerUsage(self.handle)
        except Exception as exc:
            self.init_note = f"NVML unavailable, using nvidia-smi: {type(exc).__name__}: {exc}"
            self.close()

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        self.nvml = self.handle = None

    def read(self):
        start = time.perf_counter()
        if self.nvml:
            n, h = self.nvml, self.handle
            def optional(fn, *args):
                try:
                    return fn(h, *args)
                except n.NVMLError:
                    return None
            power = n.nvmlDeviceGetPowerUsage(h) / 1000
            sample = {"power_w": power, "energy_mj": optional(n.nvmlDeviceGetTotalEnergyConsumption),
                      "temperature_c": optional(n.nvmlDeviceGetTemperature, n.NVML_TEMPERATURE_GPU),
                      "sm_clock_mhz": optional(n.nvmlDeviceGetClockInfo, n.NVML_CLOCK_SM),
                      "memory_clock_mhz": optional(n.nvmlDeviceGetClockInfo, n.NVML_CLOCK_MEM),
                      "power_limit_mw": optional(n.nvmlDeviceGetPowerManagementLimit),
                      "throttle_reasons": optional(n.nvmlDeviceGetCurrentClocksThrottleReasons),
                      "backend": "nvml"}
        else:
            result = subprocess.run(["nvidia-smi", "-i", self.uuid,
                                     "--query-gpu=power.draw,temperature.gpu,clocks.sm,clocks.mem,power.limit",
                                     "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
            fields = next(csv.reader(result.stdout.splitlines()))
            def number(s):
                try:
                    return float(s.strip())
                except ValueError:
                    return None
            p, temp, sm, mem, limit = map(number, fields)
            sample = {"power_w": p, "energy_mj": None, "temperature_c": temp,
                      "sm_clock_mhz": sm, "memory_clock_mhz": mem,
                      "power_limit_mw": None if limit is None else limit * 1000,
                      "throttle_reasons": None, "backend": "nvidia-smi"}
        if sample["power_w"] is None or not math.isfinite(sample["power_w"]) or sample["power_w"] <= 0:
            raise RuntimeError("GPU power unavailable; refusing to report zero or fabricated energy")
        sample.update(monotonic_s=time.perf_counter(), unix_s=time.time(),
                      query_seconds=time.perf_counter() - start, gpu_uuid=self.uuid)
        return sample


def integrate(samples):
    if len(samples) < 2:
        raise ValueError("At least two power samples required")
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        dt = b["monotonic_s"] - a["monotonic_s"]
        if dt < 0:
            raise ValueError("Power timestamps must be monotonic")
        total += 0.5 * (a["power_w"] + b["power_w"]) * dt
    return total


def summarize(samples, interval):
    seconds = samples[-1]["monotonic_s"] - samples[0]["monotonic_s"]
    integrated = integrate(samples)
    start, end = samples[0]["energy_mj"], samples[-1]["energy_mj"]
    counter = None if start is None or end is None else (end - start) / 1000
    notes = []
    use_counter = counter is not None and counter > 0
    if counter is not None and counter <= 0:
        notes.append("counter_reset_or_stalled; used_power_integration")
    gap = max(b["monotonic_s"] - a["monotonic_s"] for a, b in zip(samples, samples[1:]))
    valid = use_counter or gap <= max(5, 4 * interval)
    if not valid:
        notes.append("large_power_sampling_gap")
    if seconds < 10:
        notes.append("short_interval_sensor_smoothing_may_dominate")
    discrepancy = abs(counter - integrated) / counter if use_counter else None
    if discrepancy is not None and discrepancy > 0.2:
        notes.append("counter_vs_power_difference_over_20_percent")
    energy = counter if use_counter else integrated
    return {"seconds": seconds, "energy_j": energy, "energy_wh": energy / 3600,
            "energy_method": "nvml_counter" if use_counter else "power_trapezoid",
            "energy_valid": valid, "mean_w": energy / seconds if seconds > 0 else None,
            "power_integral_j": integrated, "counter_j": counter,
            "counter_power_relative_difference": discrepancy,
            "sample_count": len(samples), "max_sample_gap_s": gap,
            "sensor_note": ";".join(notes)}


class Meter:
    """Continuous raw telemetry plus explicit synchronized epoch boundary samples."""
    def __init__(self, sensor, path, interval=0.5):
        self.sensor, self.interval = sensor, interval
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.phase, self.block, self.epoch = "startup", -1, -1
        self.active = None
        self.error = None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if Path(path).suffix == ".gz" else open
        self.file = opener(path, "wt", newline="", encoding="utf-8")
        first = self.sensor.read()
        self.writer = csv.DictWriter(self.file, fieldnames=list(first) + ["phase", "block_id", "epoch_in_block"])
        self.writer.writeheader()
        self._record(first)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _record(self, sample):
        self.writer.writerow(dict(sample, phase=self.phase, block_id=self.block, epoch_in_block=self.epoch))
        self.file.flush()
        if self.active is not None:
            self.active.append(sample)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                with self.lock:
                    self._record(self.sensor.read())
            except Exception as exc:
                self.error = exc
                return

    def label(self, phase, block=-1, epoch=-1):
        with self.lock:
            self.phase, self.block, self.epoch = phase, block, epoch

    def begin(self, phase, block, epoch):
        with self.lock:
            if self.error:
                raise RuntimeError("Telemetry sampler failed") from self.error
            self.phase, self.block, self.epoch = phase, block, epoch
            self.active = []
            self._record(self.sensor.read())

    def end(self):
        with self.lock:
            if self.error:
                raise RuntimeError("Telemetry sampler failed") from self.error
            self._record(self.sensor.read())
            samples, self.active = self.active, None
            self.phase = "between_phases"
        result = summarize(samples, self.interval)
        if not result["energy_valid"]:
            raise RuntimeError("Invalid energy sampling window; raw telemetry retained")
        return result

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
        self.file.close()
        self.sensor.close()
