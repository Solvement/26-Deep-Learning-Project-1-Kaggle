"""Contain one Python training command before it can spawn any descendants.

Windows uses a private non-inherited KILL_ON_JOB_CLOSE Job Object. The early
bootstrap joins it before importing the requested script. POSIX uses a private
session and an independent owner/deadline watchdog process that terminates its
own group even if the trainer leader is killed. Unresolved cleanup fails closed.
No Torch, data or account API here.

Windows semantics: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
Session creation: https://docs.python.org/3/library/subprocess.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

import psutil

SCHEMA = "owned_python_process_tree_v1"


class TreeCleanupError(RuntimeError):
    """The caller must retain its GPU lease when cleanup is not proved."""


def may_release_lease(receipt):
    """Explicit unverified cleanup dominates a missing early child identity."""
    if "process_tree_cleanup_verified" in receipt:
        return receipt["process_tree_cleanup_verified"] is True
    return receipt.get("child_pid") is None


def matching_process(pid, created):
    try:
        process = psutil.Process(pid)
        return process if (abs(process.create_time()-created) < .01 and
            process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)) else None
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes((json.dumps(value, sort_keys=True, indent=2)+"\n").encode())
    temporary.replace(path)


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _kernel():
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, args, restype in (
        ("CreateJobObjectW", [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
        ("OpenJobObjectW", [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
        ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD], wintypes.BOOL),
        ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
        ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        ("IsProcessInJob", [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
        ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        ("GetCurrentProcess", [], wintypes.HANDLE),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL)):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, restype
    return kernel


def _windows_error(message):
    import ctypes
    return OSError(ctypes.get_last_error(), message)


class WindowsJob:
    def __init__(self, name):
        import ctypes
        from ctypes import wintypes
        if not re.fullmatch(r"Local\\SaberAugMix_[0-9a-f]{32}", name):
            raise ValueError("A unique private process-tree job name is required")

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_user", ctypes.c_longlong), ("job_user", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                ("active_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", Counters), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.kernel, self.handle, self.name = _kernel(), None, name
        handle = self.kernel.CreateJobObjectW(None, name)
        if not handle:
            raise _windows_error("Could not create the private process-tree job")
        self.handle = handle
        if ctypes.get_last_error() == 183:
            self.close()
            raise RuntimeError("Private process-tree job already exists")
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        if not self.kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise _windows_error("Could not arm KILL_ON_JOB_CLOSE")

    def active(self):
        import ctypes
        from ctypes import wintypes

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")]+[
                (name, wintypes.DWORD) for name in ("faults", "total", "active", "terminated")]

        value = Accounting()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
            raise _windows_error("Could not query owned process-tree accounting")
        return int(value.active)

    def terminate_and_confirm(self, seconds=5):
        if self.active() and not self.kernel.TerminateJobObject(self.handle, 137):
            raise _windows_error("Could not terminate the owned process tree")
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if self.active() == 0:
                return True
            time.sleep(.025)
        return self.active() == 0

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _attach_windows(name):
    import ctypes
    from ctypes import wintypes
    kernel = _kernel()
    handle = kernel.OpenJobObjectW(0x0001 | 0x0004, False, name)
    if not handle:
        raise _windows_error("The owning process-tree job is no longer present")
    try:
        current = kernel.GetCurrentProcess()
        if not kernel.AssignProcessToJobObject(handle, current):
            raise _windows_error("Early process-tree membership assignment failed")
        member = wintypes.BOOL()
        if not kernel.IsProcessInJob(current, handle, ctypes.byref(member)) or not member.value:
            raise RuntimeError("Early process-tree membership was not proved")
    finally:
        # Only the host retains a job handle; its abrupt death closes the last
        # handle and kills every member, even before a worker PID is recorded.
        kernel.CloseHandle(handle)


def _stop_own_group(spec):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = _kernel()
        handle = kernel.OpenJobObjectW(0x0004 | 0x0008, False, spec["job_name"])
        if handle:
            member = wintypes.BOOL()
            if kernel.IsProcessInJob(kernel.GetCurrentProcess(), handle, ctypes.byref(member)) and member.value:
                kernel.TerminateJobObject(handle, 124)
            kernel.CloseHandle(handle)
        os._exit(124)
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        os._exit(125)
    os.killpg(os.getpid(), signal.SIGKILL)


def _session_watchdog(spec_path, leader, leader_created):
    """Independent POSIX process: no dependency on the trainer's GIL/finally."""
    spec_path = Path(spec_path).resolve()
    spec = _read(spec_path)
    if (os.name == "nt" or not isinstance(spec, dict) or spec.get("schema") != SCHEMA or
            spec.get("guard_source_sha256") != _sha(__file__) or
            not re.fullmatch(r"[0-9a-f]{32}", str(spec.get("nonce", ""))) or
            spec_path.parent.name != "process_tree_"+spec["nonce"] or
            os.getpgrp() != leader or os.getsid(0) != leader or
            os.getppid() != leader or matching_process(leader, leader_created) is None):
        raise RuntimeError("Independent watchdog did not inherit its private owned session")
    current = psutil.Process()
    _write(spec_path.parent/"watchdog_ready.json", {"schema": SCHEMA, "nonce": spec["nonce"],
        "pid": current.pid, "created_at": current.create_time(), "script_sha256": spec["script_sha256"],
        "leader_pid": leader, "leader_created_at": leader_created})
    while True:
        original = matching_process(leader, leader_created)
        parent = matching_process(spec["host_pid"], spec["host_created_at"])
        if original is None or parent is None or time.time() >= spec["deadline_unix"]:
            # This still-live watchdog itself reserves the original session and
            # group identity. No recycled, unrelated PGID can be targeted.
            if os.getpgrp() == leader and os.getsid(0) == leader:
                os.killpg(leader, signal.SIGKILL)
            os._exit(125)
        time.sleep(.05)


def _group_empty(pgid, created):
    """Read-only check scoped to the private session; never kill by a scan."""
    current_leader = None
    try:
        current_leader = psutil.Process(pgid)
    except psutil.NoSuchProcess:
        pass
    if current_leader is not None and abs(current_leader.create_time()-created) >= .01:
        return False  # PGID/PID was reused; never act on the new process.
    for process in psutil.process_iter(("pid", "status")):
        try:
            if os.getsid(process.pid) == pgid and os.getpgid(process.pid) == pgid:
                if process.create_time() < created-.01 or process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return False
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        except (psutil.AccessDenied, PermissionError):
            return False
    return True


def _valid_child_record(value, spec, pid, created):
    return (type(created) in (int, float) and isinstance(value, dict) and value.get("schema") == SCHEMA and value.get("nonce") == spec["nonce"] and
            value.get("pid") == pid and type(value.get("created_at")) in (int, float) and
            abs(value["created_at"]-created) < .01 and value.get("script_sha256") == spec["script_sha256"])


def run_guarded(command, log_path, timeout, receipt_path, receipt, cwd=None):
    """Execute one script under an early, independently owned tree boundary."""
    cwd = Path(cwd or Path(__file__).resolve().parents[1]).resolve()
    if (not isinstance(command, (list, tuple)) or len(command) < 2 or
            any(not isinstance(part, str) or "\x00" in part for part in command) or
            Path(command[0]).resolve() != Path(sys.executable).resolve()):
        raise ValueError("Only an explicit Python script argument vector is allowed")
    script = (cwd/command[1]).resolve() if not Path(command[1]).is_absolute() else Path(command[1]).resolve()
    if not script.is_file() or not script.is_relative_to(cwd):
        raise ValueError("The controlled script must be a file within its declared working directory")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 10800:
        raise ValueError("A finite positive process-tree timeout no greater than three hours is required")
    receipt_path, log_path = Path(receipt_path).resolve(), Path(log_path).resolve()
    directory = receipt_path.parent/("process_tree_"+uuid.uuid4().hex)
    directory.mkdir(parents=False, exist_ok=False)
    nonce = directory.name.removeprefix("process_tree_")
    host = psutil.Process()
    spec = {"schema": SCHEMA, "nonce": nonce, "host_pid": host.pid, "host_created_at": host.create_time(),
        "deadline_unix": time.time()+timeout, "script": str(script), "script_sha256": _sha(script),
        "arguments": list(command[2:]), "cwd": str(cwd), "guard_source_sha256": _sha(__file__),
        "job_name": "Local\\SaberAugMix_"+nonce if os.name == "nt" else None}
    _write(directory/"spec.json", spec)
    process, job, timed_out, caught, cleanup = None, None, False, None, False
    proof = {"schema": SCHEMA, "nonce": nonce, "method": "windows_job_v1" if os.name == "nt" else "posix_session_v1",
        "guard_directory": str(directory), "guard_source_sha256": spec["guard_source_sha256"], "cleanup_verified": False}
    receipt["process_tree_cleanup_verified"] = False
    _write(receipt_path, receipt)
    try:
        if os.name == "nt":
            job = WindowsJob(spec["job_name"])
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--bootstrap", str(directory/"spec.json")],
                cwd=cwd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                close_fds=True, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            # Publish that a process exists before querying an optional
            # user-mode identity API that can itself fail during early exit.
            receipt["child_pid"] = process.pid
            proof["child_pid"] = process.pid
            _write(receipt_path, receipt)
            created = psutil.Process(process.pid).create_time()
            receipt.update(child_pid=process.pid, child_created_at=created)
            proof.update(child_pid=process.pid, child_created_at=created)
            _write(receipt_path, receipt)
            try:
                process.wait(timeout=max(0, spec["deadline_unix"]-time.time()))
                completed = _read(directory/"completion.json")
                # The independent child deadline can win the race with the
                # host's wait timeout. Its forced exit is still a timeout,
                # never an unauthenticated successful command.
                if (time.time() >= spec["deadline_unix"] and
                        process.returncode in (124, -getattr(signal, "SIGKILL", 9)) and
                        not _valid_child_record(completed, spec, process.pid, created)):
                    timed_out = True
                    caught = subprocess.TimeoutExpired(command, timeout)
            except subprocess.TimeoutExpired as error:
                timed_out, caught = True, error
    except BaseException as error:
        caught = error
    finally:
        try:
            if os.name == "nt":
                cleanup = True if job is None else job.terminate_and_confirm()
                if process is not None:
                    process.wait(timeout=5)
            elif process is None:
                cleanup = True
            else:
                ready = _read(directory/"ready.json")
                alive = matching_process(process.pid, proof["child_created_at"])
                if alive is not None and process.poll() is None:
                    if _valid_child_record(ready, spec, process.pid, proof["child_created_at"]):
                        # The handler runs in the original live session leader,
                        # so the group identity cannot refer to an unrelated PID.
                        alive.send_signal(signal.SIGUSR1)
                    else:
                        # No engine may run before the ready record. No children
                        # can have been spawned by the requested script yet.
                        alive.kill()
                    process.wait(timeout=5)
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    if _group_empty(process.pid, proof["child_created_at"]):
                        cleanup = True
                        break
                    time.sleep(.05)
        except BaseException as error:
            proof["cleanup_error_type"] = type(error).__name__
            cleanup = False
        finally:
            if job is not None:
                job.close()
        completion = _read(directory/"completion.json")
        authenticated = (process is not None and _valid_child_record(completion, spec, process.pid, proof.get("child_created_at")))
        proof.update(cleanup_verified=cleanup, timed_out=timed_out, child_returncode=None if process is None else process.poll(),
            completion_authenticated=authenticated, completion_exit_code=completion.get("exit_code") if authenticated else None)
        _write(directory/"host_result.json", proof)
        receipt.setdefault("process_tree_guard_results", []).append(proof)
        receipt["process_tree_cleanup_verified"] = cleanup
        _write(receipt_path, receipt)
    if not cleanup:
        raise TreeCleanupError("Owned process-tree cleanup is unverified; retain the GPU lease") from caught
    if caught is not None:
        raise caught
    if not authenticated or type(completion.get("exit_code")) is not int:
        raise RuntimeError("The guarded command did not produce its authenticated completion")
    if completion["exit_code"]:
        raise subprocess.CalledProcessError(completion["exit_code"], command)
    return proof


def _bootstrap(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = _read(spec_path)
    if (not isinstance(spec, dict) or spec.get("schema") != SCHEMA or not re.fullmatch(r"[0-9a-f]{32}", str(spec.get("nonce", ""))) or
            spec_path.parent.name != "process_tree_"+spec["nonce"] or spec.get("guard_source_sha256") != _sha(__file__) or
            not isinstance(spec.get("arguments"), list) or any(not isinstance(v, str) or "\x00" in v for v in spec["arguments"])):
        raise ValueError("Invalid process-tree bootstrap specification")
    if (psutil.Process().ppid() != spec["host_pid"] or
            matching_process(spec["host_pid"], spec["host_created_at"]) is None or time.time() >= spec["deadline_unix"]):
        raise RuntimeError("Owning supervisor disappeared or its deadline expired before startup")
    script = Path(spec["script"]).resolve()
    cwd = Path(spec["cwd"]).resolve()
    if not script.is_relative_to(cwd) or _sha(script) != spec["script_sha256"]:
        raise RuntimeError("The controlled script changed before guarded startup")
    if os.name == "nt":
        expected = "Local\\SaberAugMix_"+spec["nonce"]
        if spec["job_name"] != expected:
            raise ValueError("Private Windows Job Object identity differs")
        _attach_windows(expected)
    elif os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise RuntimeError("A private POSIX process session was not established")
    current = psutil.Process()
    identity = {"schema": SCHEMA, "nonce": spec["nonce"], "pid": current.pid, "created_at": current.create_time(),
        "script_sha256": spec["script_sha256"], "method": "windows_job_v1" if os.name == "nt" else "posix_session_v1"}
    if os.name != "nt":
        signal.signal(signal.SIGUSR1, lambda *_: _stop_own_group(spec))

    stopped = threading.Event()
    def monitor_owner():
        while not stopped.wait(.05):
            if matching_process(spec["host_pid"], spec["host_created_at"]) is None or time.time() >= spec["deadline_unix"]:
                _stop_own_group(spec)
    if os.name == "nt":
        thread = threading.Thread(target=monitor_owner, daemon=True, name="owned-tree-deadline")
        thread.start()
    else:
        # Start and authenticate this independent process before the requested
        # module can import Torch or create a single DataLoader worker.
        watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--session-watchdog", str(spec_path),
            "--leader", str(current.pid), "--leader-created", str(current.create_time())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        watcher_created = psutil.Process(watcher.pid).create_time()
        while True:
            acknowledged = _read(spec_path.parent/"watchdog_ready.json")
            if (_valid_child_record(acknowledged, spec, watcher.pid, watcher_created) and
                    acknowledged.get("leader_pid") == current.pid and
                    abs(acknowledged.get("leader_created_at", 0)-current.create_time()) < .01):
                identity.update(watchdog_pid=watcher.pid, watchdog_created_at=watcher_created)
                break
            if watcher.poll() is not None or time.time() >= spec["deadline_unix"]:
                _stop_own_group(spec)
            time.sleep(.01)
    _write(spec_path.parent/"ready.json", identity)
    code = 0
    previous = sys.argv
    try:
        os.chdir(cwd)
        sys.argv = [str(script), *spec["arguments"]]
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        code = error.code if type(error.code) is int else (0 if error.code is None else 1)
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        sys.argv = previous
        _write(spec_path.parent/"completion.json", {**identity, "exit_code": code})
        stopped.set()
    if os.name != "nt":
        # End this still-live, authenticated session from inside it. This also
        # closes unregistered workers if the script failed during pool startup.
        _stop_own_group(spec)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bootstrap", type=Path)
    group.add_argument("--session-watchdog", type=Path)
    parser.add_argument("--leader", type=int)
    parser.add_argument("--leader-created", type=float)
    args = parser.parse_args()
    if args.session_watchdog:
        if args.leader is None or args.leader_created is None:
            parser.error("The private session leader's exact identity is required")
        return _session_watchdog(args.session_watchdog, args.leader, args.leader_created)
    return _bootstrap(args.bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
