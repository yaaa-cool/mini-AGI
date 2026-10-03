"""
Keep a dry read off the weights directory it reads.

A read without --save still trains in memory, and the paged pool parks every
expert that leaves the card as dirty. When the RAM cache is full a dirty
expert is written back to its file - into the real weights directory, while
the trunk, routers, optimiser and manifest are never saved. The result is a
hybrid checkpoint (issue #6).

Opening the pool read-only is not the answer: an evicted expert would then
lose what it learned this session and come back as it was on disk. So the
writes go to a temporary overlay instead. An expert written there is read
back from there; every other expert is still read from the source, which is
never written. The overlay is deleted when the process ends.
"""

import os
import tempfile


def overlay(pool, path):
    """Send `pool`'s expert writeback to a temporary directory beside `path`."""
    tiers = pool.tiers
    parent = os.path.dirname(os.path.abspath(path))
    try:
        tmp = tempfile.TemporaryDirectory(prefix=".minagi-dry-", dir=parent)
    except OSError:                      # parent not writable - use the default
        tmp = tempfile.TemporaryDirectory(prefix="minagi-dry-")
    source_file, to_disk = tiers._file, tiers._to_disk
    writing = [False]

    def _file(i):
        f = os.path.join(tmp.name, "e%05d.npz" % i)
        return f if writing[0] or os.path.exists(f) else source_file(i)

    def _to_disk(i, ent):
        writing[0] = True
        try:
            to_disk(i, ent)
        finally:
            writing[0] = False

    tiers._file, tiers._to_disk = _file, _to_disk
    # held by the pool, so the directory lives exactly as long as it can be read
    pool._dry_overlay = tmp
    return tmp.name
