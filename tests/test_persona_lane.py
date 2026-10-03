import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from corpora import persona

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", ["persona.py", *args]), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = persona.main()
    return rc, out.getvalue(), err.getvalue()


def _files(d):
    return sorted(os.path.relpath(os.path.join(r, f), d)
                  for r, _, fs in os.walk(d) for f in fs)


def _subjects(root):
    """The subject each file is read as, the way train.py _lanes groups them:
    the first folder under the read root."""
    root = os.path.abspath(root)
    return {os.path.relpath(os.path.join(r, f), root).split(os.sep)[0]
            for r, _, fs in os.walk(root) for f in fs}


DIALOG = {"messages": [
    {"role": "system", "content": "dropped"},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "A reader of <user> turns."}]}


class PersonaLaneTest(unittest.TestCase):
    def test_jsonl_becomes_a_lane_with_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "sft.jsonl")
            with open(src, "w") as f:
                for _ in range(40):
                    f.write(json.dumps(DIALOG) + "\n")
                f.write(json.dumps({"messages": [
                    {"role": "assistant", "content": "no user first"}]}) + "\n")
            out = os.path.join(tmp, "data")
            rc, _, _ = _run("--src", src, "--out", out, "--group", "50",
                            "--hold", "2")
            self.assertEqual(rc, 0)

            train = os.path.join(out, "train", "persona")
            val = os.path.join(out, "val", "persona")
            self.assertEqual(_files(val), ["part-000000.txt",
                                           "part-000001.txt"])
            self.assertEqual(len(_files(train)), 38)
            self.assertTrue(all(f.startswith("0000" + os.sep)
                                for f in _files(train)))
            with open(os.path.join(train, "0000", "part-000000.txt")) as f:
                text = f.read()
            self.assertTrue(text.startswith(
                "<user>\nWho are you?\n</user>\n<bot>\n"))
            self.assertNotIn("dropped", text)
            self.assertNotIn("no user first", text)
            # a marker inside content is text, not a forged turn boundary
            self.assertIn("A reader of &lt;user&gt; turns.\n</bot>\n", text)
            self.assertEqual(text.count("<user>"), text.count("</bot>"))
            self.assertEqual(_subjects(os.path.join(out, "train")),
                             {"persona"})
            self.assertEqual(_subjects(os.path.join(out, "val")), {"persona"})

    def test_exported_folder_is_copied_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "export")
            held = os.path.join(tmp, "export_val")
            for d, names in ((os.path.join(src, "0000"),
                              ["part-000000.txt", "part-000001.txt"]),
                             (os.path.join(src, "0001"), ["part-002000.txt"]),
                             (held, ["part-000000.txt"])):
                os.makedirs(d)
                for n in names:
                    with open(os.path.join(d, n), "w") as f:
                        f.write(f"<user>\n{n}\n</user>\n<bot>\nok\n</bot>\n")
            out = os.path.join(tmp, "data")
            rc, _, _ = _run("--src", src, "--src-val", held, "--out", out)
            self.assertEqual(rc, 0)
            train = os.path.join(out, "train", "persona")
            got = []
            for name in _files(train):
                with open(os.path.join(train, name)) as f:
                    got.append(f.read().split("\n")[1])
            self.assertEqual(got, ["part-000000.txt", "part-000001.txt",
                                   "part-002000.txt"])
            self.assertEqual(_files(os.path.join(out, "val", "persona")),
                             ["part-000000.txt"])

    def test_existing_lane_needs_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "sft.jsonl")
            with open(src, "w") as f:
                f.write(json.dumps(DIALOG) + "\n")
            out = os.path.join(tmp, "data")
            stale = os.path.join(out, "train", "persona", "0000")
            os.makedirs(stale)
            with open(os.path.join(stale, "part-000009.txt"), "w") as f:
                f.write("old")
            rc, _, err = _run("--src", src, "--out", out)
            self.assertEqual(rc, 1)
            self.assertIn("--force", err)
            rc, _, _ = _run("--src", src, "--out", out, "--force")
            self.assertEqual(rc, 0)
            self.assertEqual(_files(os.path.join(out, "train", "persona")),
                             [os.path.join("0000", "part-000000.txt")])

    def test_source_without_dialogs_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "empty.jsonl")
            open(src, "w").close()
            rc, _, err = _run("--src", src, "--out", os.path.join(tmp, "d"))
            self.assertEqual(rc, 1)
            self.assertIn("no chat text", err)

    def test_registered_as_corpora_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "sft.jsonl")
            with open(src, "w") as f:
                f.write(json.dumps(DIALOG) + "\n")
            r = subprocess.run([sys.executable, "-m", "corpora", "persona",
                                "--src", src, "--out",
                                os.path.join(tmp, "data")],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.isdir(
                os.path.join(tmp, "data", "train", "persona")))


if __name__ == "__main__":
    unittest.main()
