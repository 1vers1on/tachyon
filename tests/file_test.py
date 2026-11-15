"""Small, offline malformed-model regressions; no model downloads or Python packages."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

probe = str(Path(sys.argv[1]).resolve())
checks = 0


def check(mode, path, valid=False, message=""):
    global checks
    p = subprocess.run([probe, mode, str(path)], capture_output=True, text=True)
    assert p.returncode == (0 if valid else 1), (mode, p.returncode, p.stderr)
    assert message in p.stderr, (message, p.stderr)
    checks += 1


def tach(path, header, version=2, payload=b"\0" * 512):
    h = json.dumps(header).encode()
    if version == 2:
        path.write_bytes(struct.pack("<8sQQ", b"TACHYON2", 4096 + len(payload), len(h))
                         + b"\0" * (4096 - 24) + payload + h)
    else:
        start = (16 + len(h) + 4095) // 4096 * 4096
        path.write_bytes(struct.pack("<8sQ", b"TACHYON1", len(h)) + h
                         + b"\0" * (start - 16 - len(h)) + payload)


with tempfile.TemporaryDirectory(prefix="tachyon-files-") as tmp:
    root = Path(tmp)
    model = root / "test.tach"
    tensor = dict(name="w", type="q4", rows=1, cols=64, offset=0, data_bytes=32,
                  scale_offset=256, scale_bytes=4)
    header = dict(config={}, tensors=[tensor], blobs=[dict(name="tok", offset=32, size=16)])
    for version in (1, 2):
        tach(model, header, version)
        check("tach", model, valid=True)
    for raw in (b"", b"TACHYON2", struct.pack("<8sQQ", b"TACHYON2", 2**64 - 1, 128),
                struct.pack("<8sQ", b"TACHYON1", 2**64 - 1)):
        model.write_bytes(raw)
        check("tach", model)
    for key, value, error in (("offset", -1, "invalid"), ("offset", 1.5, "invalid"),
                              ("offset", 10000, "out of range"),
                              ("offset", 512, "out of range"),
                              ("scale_offset", 1024, "out of range"),
                              ("scale_bytes", 0, "byte size"),
                              ("data_bytes", 31, "byte size"), ("cols", 63, "byte size"),
                              ("rows", 0, "dimensions"), ("rows", 2**40, "dimensions")):
        h = copy.deepcopy(header)
        h["tensors"][0][key] = value
        tach(model, h)
        check("tach", model, message=error)
    for key in ("offset", "size"):
        h = copy.deepcopy(header)
        h["blobs"][0][key] = 10000
        tach(model, h)
        check("tach", model, message="out of range")
    for key in ("tensors", "blobs"):
        h = copy.deepcopy(header)
        h[key] *= 2
        tach(model, h)
        check("tach", model, message="duplicate")

    sf = root / "model.safetensors"

    def safe(meta, payload=b"\0" * 16):
        h = json.dumps(meta).encode()
        sf.write_bytes(struct.pack("<Q", len(h)) + h + payload)

    meta = dict(w=dict(dtype="F32", shape=[4], data_offsets=[0, 16]))
    safe(meta)
    check("safe", root, valid=True)
    check("read", root, message="read out of range")
    for raw in (b"", b"\0" * 4, struct.pack("<Q", 2**64 - 1)):
        sf.write_bytes(raw)
        check("safe", root, message="truncated")
    for field, value, error in (("data_offsets", [], "offsets"),
                               ("data_offsets", [0, 17], "out of range"),
                               ("data_offsets", [16, 0], "out of range"),
                               ("data_offsets", [-1, 16], "invalid"),
                               ("shape", [8], "byte size"), ("shape", [-1], "invalid"),
                               ("shape", [2**40, 2**40], "overflow"),
                               ("shape", None, "shape")):
        m = copy.deepcopy(meta)
        m["w"][field] = value
        safe(m)
        check("safe", root, message=error)
    for shape in ([], [0]):
        safe(dict(w=dict(dtype="F32", shape=shape, data_offsets=[0, 4 if not shape else 0])),
             b"\0" * (4 if not shape else 0))
        check("safe", root, valid=True)

print(f"file loading: {checks} checks passed")
