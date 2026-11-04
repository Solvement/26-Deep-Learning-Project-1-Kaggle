"""Real CPU process-tree containment; no training, GPU or official inputs.

Every child is created by this test and recorded with its creation time. The
tests never scan or terminate unrelated processes.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

ROOT = Path(__file__).resolve().parents[1]


class ProcessTreeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = importlib.import_module("research.process_tree_guard")

    def test_explicit_unverified_cleanup_never_releases_an_unrecorded_child_lease(self):
        for receipt in ({"process_tree_cleanup_verified": False},
                        {"process_tree_cleanup_verified": None},
                        {"process_tree_cleanup_verified": False, "child_pid": 123},
                        {"child_pid": 123}):
            with self.subTest(receipt=receipt):
                self.assertFalse(self.guard.may_release_lease(receipt))
        self.assertTrue(self.guard.may_release_lease({}))
        self.assertTrue(self.guard.may_release_lease({"process_tree_cleanup_verified": True}))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="saber-tree-tests-")
        self.directory = Path(self.temp.name).resolve()
        self.assertTrue(self.directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.directory.name.startswith("saber-tree-tests-"))
        self.addCleanup(self._cleanup)
        self.hosts = []
        self.receipt = {}
        self._scripts()

    def _popen(self, command):
        process = subprocess.Popen(command, cwd=self.directory, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        record = {"pid": process.pid, "created_at": psutil.Process(process.pid).create_time()}
        self.hosts.append((process, record))
        return process, record

    def _scripts(self):
        shared = '''from pathlib import Path
import json, os, psutil, subprocess, sys, time
folder = Path(sys.argv[1])
def record(name):
    (folder/(name+'.json')).write_text(json.dumps({'pid':os.getpid(), 'created_at':psutil.Process().create_time()}))
hidden = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
'''
        (self.directory/"grandchild.py").write_text(shared+'''record('grandchild')
time.sleep(60)
''', encoding="utf8")
        (self.directory/"child.py").write_text(shared+'''record('child')
if len(sys.argv) < 3 or sys.argv[2] != 'partial':
    subprocess.Popen([sys.executable,str(folder/'grandchild.py'),str(folder)], creationflags=hidden)
time.sleep(60)
''', encoding="utf8")
        (self.directory/"target.py").write_text(shared+'''record('target')
mode=sys.argv[2]
subprocess.Popen([sys.executable,str(folder/'child.py'),str(folder),mode],creationflags=hidden)
needed='child' if mode=='partial' else 'grandchild'
until=time.monotonic()+5
while not (folder/(needed+'.json')).exists():
    if time.monotonic()>until: raise RuntimeError('Own descendant did not start')
    time.sleep(.02)
print('GUARD_TEST_ARGV='+json.dumps(sys.argv[3:]),flush=True)
if mode=='normal': raise SystemExit(0)
if mode=='nonzero':
    print('intentional exit seven',file=sys.stderr,flush=True)
    raise SystemExit(7)
time.sleep(60)
''', encoding="utf8")
        (self.directory/"canary.py").write_text(shared+'''record('canary')
time.sleep(60)
''', encoding="utf8")
        (self.directory/"host.py").write_text('''from pathlib import Path
import sys
sys.path.insert(0, '''+repr(str(ROOT))+''')
from research.process_tree_guard import run_guarded
folder=Path(sys.argv[1]); mode=sys.argv[2]
run_guarded([sys.executable,str(folder/'target.py'),str(folder),mode],
    folder/'task.log',30,folder/'receipt.json',{},cwd=folder)
''', encoding="utf8")

    def _record(self, name):
        path = self.directory/(name+".json")
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            try:
                value = json.loads(path.read_text())
                if type(value.get("pid")) is int and type(value.get("created_at")) in (float, int):
                    return value
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(.02)
        self.fail("Own process did not publish its identity: "+name)

    @staticmethod
    def _matching(record):
        try:
            process = psutil.Process(record["pid"])
            if abs(process.create_time()-record["created_at"]) >= .01:
                return None
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except psutil.NoSuchProcess:
            return None

    def _assert_stopped(self, record):
        deadline = time.monotonic()+6
        while self._matching(record) is not None and time.monotonic() < deadline:
            time.sleep(.025)
        self.assertIsNone(self._matching(record), "Own descendant remained alive: "+str(record["pid"]))

    def _cleanup(self):
        # All paths were generated below one verified tempfile directory. Never
        # infer ownership from a bare PID or enumerate other users' processes.
        for path in (self.directory/name for name in ("target.json", "child.json", "grandchild.json", "canary.json")):
            try:
                record = json.loads(path.read_text())
                process = self._matching(record)
                if process is not None:
                    process.kill()
                    process.wait(timeout=4)
            except (FileNotFoundError, json.JSONDecodeError, psutil.NoSuchProcess):
                pass
        for process, record in self.hosts:
            exact = self._matching(record)
            if exact is not None:
                exact.kill()
            try: process.wait(timeout=4)
            except subprocess.TimeoutExpired: pass
        self.temp.cleanup()

    def _run(self, mode, timeout=8, arguments=()):
        return self.guard.run_guarded(
            [sys.executable,str(self.directory/"target.py"),str(self.directory),mode,*arguments],
            self.directory/"task.log",timeout,self.directory/"receipt.json",self.receipt,cwd=self.directory)

    def test_normal_exit_reaps_child_and_grandchild_without_pool_receipts(self):
        proof = self._run("normal")
        self.assertIs(proof["cleanup_verified"], True)
        self.assertIs(self.receipt["process_tree_cleanup_verified"], True)
        for name in ("target", "child", "grandchild"):
            self._assert_stopped(self._record(name))
        self.assertFalse(any(self.directory.glob("worker_pool_*.json")))

    def test_nonzero_exit_is_preserved_after_the_complete_tree_is_reaped(self):
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self._run("nonzero")
        self.assertEqual(caught.exception.returncode, 7)
        self.assertIs(self.receipt["process_tree_cleanup_verified"], True)
        self.assertIn("intentional exit seven",(self.directory/"task.log").read_text())
        for name in ("target", "child", "grandchild"):
            self._assert_stopped(self._record(name))

    def test_deadline_kills_complete_tree_and_reports_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self._run("wait",timeout=3)
        self.assertIs(self.receipt["process_tree_cleanup_verified"], True)
        for name in ("target", "child", "grandchild"):
            self._assert_stopped(self._record(name))

    def test_abrupt_owner_death_reaps_an_unrecorded_tree_and_leaves_canary(self):
        _,canary = self._popen([sys.executable,str(self.directory/"canary.py"),str(self.directory)])
        self._record("canary")
        host,identity = self._popen([sys.executable,str(self.directory/"host.py"),str(self.directory),"wait"])
        records = [self._record(name) for name in ("target", "child", "grandchild")]
        exact = self._matching(identity)
        self.assertIsNotNone(exact)
        exact.kill(); host.wait(timeout=4)
        for record in records: self._assert_stopped(record)
        self.assertIsNotNone(self._matching(canary), "Unrelated own canary was affected")
        self.assertFalse(any(self.directory.glob("worker_pool_*.json")))

    def test_owner_death_during_partial_startup_needs_no_worker_pool_receipt(self):
        host,identity = self._popen([sys.executable,str(self.directory/"host.py"),str(self.directory),"partial"])
        records = [self._record(name) for name in ("target", "child")]
        self.assertFalse((self.directory/"grandchild.json").exists())
        self.assertFalse(any(self.directory.glob("worker_pool_*.json")))
        exact = self._matching(identity)
        self.assertIsNotNone(exact)
        exact.kill(); host.wait(timeout=4)
        for record in records: self._assert_stopped(record)

    def test_argument_stdout_and_nonce_evidence_survive_the_bootstrap_boundary(self):
        arguments = ("spaces stay together", "literal $() and `characters`", "unicode_模型")
        proof = self._run("normal",arguments=arguments)
        text = (self.directory/"task.log").read_text(encoding="utf8")
        line = next(line for line in text.splitlines() if line.startswith("GUARD_TEST_ARGV="))
        self.assertEqual(json.loads(line.split("=",1)[1]),list(arguments))
        self.assertIs(proof["cleanup_verified"], True)
        self.assertIs(proof["completion_authenticated"], True)
        self.assertRegex(proof["nonce"],r"^[0-9a-f]{32}$")
        saved = json.loads((self.directory/"receipt.json").read_text())
        self.assertEqual(saved["process_tree_guard_results"][-1]["nonce"],proof["nonce"])

    def test_creation_time_mismatch_never_matches_a_live_process(self):
        process = psutil.Process()
        self.assertEqual(self.guard.matching_process(process.pid,process.create_time()).pid,process.pid)
        self.assertIsNone(self.guard.matching_process(process.pid,process.create_time()+60))

    def test_child_completion_requires_nonce_pid_creation_and_script_identity(self):
        spec = {"nonce":"a"*32,"script_sha256":"b"*64}
        record = {"schema":self.guard.SCHEMA,"nonce":spec["nonce"],"pid":1234,
                  "created_at":1234.5,"script_sha256":spec["script_sha256"]}
        self.assertTrue(self.guard._valid_child_record(record,spec,1234,1234.5))
        for key,value in (("nonce","c"*32),("pid",5678),("created_at",1235.5),("script_sha256","d"*64)):
            with self.subTest(field=key):
                self.assertFalse(self.guard._valid_child_record({**record,key:value},spec,1234,1234.5))

    def test_hard_killed_trainer_still_reaps_unregistered_descendants(self):
        host,_ = self._popen([sys.executable,str(self.directory/"host.py"),str(self.directory),"wait"])
        records = [self._record(name) for name in ("target","child","grandchild")]
        self.assertFalse(any(self.directory.glob("worker_pool_*.json")))
        before = json.loads((self.directory/"receipt.json").read_text())
        self.assertEqual(before["child_pid"],records[0]["pid"])
        self.assertLess(abs(before["child_created_at"]-records[0]["created_at"]),.01)
        exact = self._matching(records[0])
        self.assertIsNotNone(exact)
        exact.kill()
        self.assertNotEqual(host.wait(timeout=8),0)
        for record in records: self._assert_stopped(record)
        saved = json.loads((self.directory/"receipt.json").read_text())
        self.assertIs(saved["process_tree_cleanup_verified"],True)
        self.assertIs(saved["process_tree_guard_results"][-1]["completion_authenticated"],False)

    def test_command_outside_explicit_workspace_is_rejected_before_launch(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.guard.run_guarded([sys.executable,str(ROOT/"research/train_augmix.py")],
                self.directory/"task.log",3,self.directory/"receipt.json",{},cwd=self.directory)
        self.assertFalse((self.directory/"target.json").exists())

    def test_guarded_main_module_supports_real_cpu_dataloader_spawn(self):
        script = self.directory/"cpu_spawn.py"
        script.write_text('''import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import json, psutil, sys
import torch
from torch.utils.data import Dataset, DataLoader

class TinyDataset(Dataset):
    def __len__(self): return 8
    def __getitem__(self, index):
        assert not torch.cuda.is_initialized()
        return torch.tensor(index)

if __name__ == '__main__':
    torch.set_num_threads(1)
    loader = DataLoader(TinyDataset(), batch_size=2, num_workers=2,
                        multiprocessing_context='spawn', persistent_workers=False)
    iterator = iter(loader)
    records = [{'pid':w.pid,'created_at':psutil.Process(w.pid).create_time()} for w in iterator._workers]
    values = torch.cat(list(iterator)).tolist()
    assert values == list(range(8)) and not torch.cuda.is_initialized()
    Path(sys.argv[1]).write_text(json.dumps({'workers':records,'values':values}))
''', encoding="utf8")
        output = self.directory/"spawn_result.json"
        proof = self.guard.run_guarded([sys.executable,str(script),str(output)],
            self.directory/"spawn.log",25,self.directory/"spawn_receipt.json",{},cwd=self.directory)
        self.assertTrue(proof["cleanup_verified"])
        result = json.loads(output.read_text())
        self.assertEqual(result["values"], list(range(8)))
        self.assertEqual(len(result["workers"]), 2)
        for record in result["workers"]:
            self._assert_stopped(record)


if __name__ == "__main__": unittest.main()
