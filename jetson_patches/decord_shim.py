"""Minimal `decord` shim backed by imageio-ffmpeg.

The real `decord` package has no aarch64 wheel and its source does not build
against the system ffmpeg 6.1 on this Jetson Thor. NeoVerse only uses a tiny
slice of decord's API (VideoReader -> len / get_batch().asnumpy()), so this
shim reproduces exactly that surface using imageio-ffmpeg, which ships its own
ffmpeg binary and is already a NeoVerse dependency.

If you ever need the real decord (e.g. GPU video decode), remove this directory
and install the genuine package.
"""
import numpy as np
import imageio.v2 as imageio


class _NDArrayBatch:
    """Mimics decord's NDArray batch: holds frames, exposes .asnumpy()."""

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, i):
        return self._arr[i]


class VideoReader:
    """Drop-in replacement for decord.VideoReader for NeoVerse's needs.

    Materializes all frames up front (NeoVerse clips are short, ~81 frames),
    giving reliable random access via get_batch / indexing.
    """

    def __init__(self, uri, ctx=None, num_threads=0, width=-1, height=-1,
                 *args, **kwargs):
        reader = imageio.get_reader(uri, "ffmpeg")
        self._frames = np.stack([np.asarray(f) for f in reader], axis=0)
        reader.close()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return _NDArrayBatch(self._frames[idx])

    def get_batch(self, indices):
        idx = [int(i) for i in indices]
        return _NDArrayBatch(self._frames[idx])

    def get_avg_fps(self):
        return 0.0


# --- decord module-level helpers used elsewhere (no-ops here) ---
def cpu(device_id=0):
    return None


def gpu(device_id=0):
    return None


class _Bridge:
    def set_bridge(self, name):
        # Inference path uses .asnumpy(); torch bridge not needed.
        pass


bridge = _Bridge()

__version__ = "0.6.0+imageio-shim"
