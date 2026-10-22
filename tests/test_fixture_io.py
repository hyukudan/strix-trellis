"""Prove unsafe range responses and changed fixture bytes are rejected."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from strix_trellis import load_fixture

spec = importlib.util.spec_from_file_location(
    "fetch_fixture", Path(__file__).resolve().parents[1] / "tools/fetch_fixture.py")
fetch_fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_fixture)


class Response:
    def __init__(self, status=206, content_range="bytes 10-13/1000", data=b"abcd"):
        self.status, self.headers, self.data = status, {"Content-Range": content_range}, data
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        self.read_called = True
        return self.data[:size]


class RangeTests(unittest.TestCase):
    def test_exact_range_request_and_budget(self):
        response = Response()
        reader = fetch_fixture.RangeReader("https://example.invalid/shard", budget=4)
        with patch.object(fetch_fixture.urllib.request, "urlopen", return_value=response) as open_url:
            self.assertEqual(reader.read(10, 14), b"abcd")
            request = open_url.call_args.args[0]
            self.assertEqual(request.get_header("Range"), "bytes=10-13")
            self.assertEqual(reader.used, 4)
            self.assertEqual(reader.records[0]["sha256"], hashlib.sha256(b"abcd").hexdigest())
            with self.assertRaises(ValueError):
                reader.read(14, 15)
            self.assertEqual(open_url.call_count, 1)

    def test_full_response_or_wrong_range_rejected_before_read(self):
        for response in [Response(status=200), Response(content_range="bytes 0-3/1000")]:
            with patch.object(fetch_fixture.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(RuntimeError):
                    fetch_fixture.RangeReader("https://example.invalid/shard").read(10, 14)
                self.assertFalse(response.read_called)

    def test_truncated_or_overlong_response_rejected(self):
        for data in (b"abc", b"abcde"):
            with patch.object(fetch_fixture.urllib.request, "urlopen", return_value=Response(data=data)):
                with self.assertRaises(RuntimeError):
                    fetch_fixture.RangeReader("https://example.invalid/shard").read(10, 14)


class LoaderTests(unittest.TestCase):
    def test_hash_corruption_and_unknown_multiplier_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = {"fixture_directory": str(directory), "tensors": {}}
            for suffix, array, dtype in [
                ("trellis", np.zeros((8, 8, 16), dtype="<i2"), "I16"),
                ("suh", np.ones(128, dtype="<f2"), "F16"),
                ("svh", np.ones(128, dtype="<f2"), "F16"),
                ("mul1", np.array(0, dtype="<i4"), "I32"),
            ]:
                data = array.tobytes()
                (directory / suffix).write_bytes(data)
                metadata["tensors"]["synthetic." + suffix] = dict(
                    file=suffix, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    shape=list(array.shape), dtype=dtype)
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported mul1"):
                load_fixture(manifest)
            (directory / "suh").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_fixture(manifest)


if __name__ == "__main__":
    unittest.main()
