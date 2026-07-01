import os.path as osp
import numpy as np
import cv2
import numpy as np
import json
import os
import sys
import pandas as pd
from decord import VideoReader
import gc
from contextlib import contextmanager

from tqdm import tqdm
from ..base_dataset import BaseDataset


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


class SpatialVID(BaseDataset):
    def __init__(self, ROOT, labels_dir=None, *args, **kwargs):
        """
        Args:
            ROOT: SpatialVID root directory.
            labels_dir: (semantic finetune) directory containing per-clip SAM3 label npz
                files, named "<scene_id>.npz" with a "labels" array of shape
                [N_total_frames, H, W] int8 that matches the raw video's frame count.
                If None, no labels are loaded and normal RGB training proceeds unchanged.
        """
        self.ROOT = ROOT
        self.labels_dir = labels_dir
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        metadata = pd.read_csv(osp.join(self.ROOT, "data/train/SpatialVID_HQ_metadata.csv"))
        min_anno_length = (self.num_views - 1) * self.min_interval + 1
        annotation_interval = (0.2 * metadata["fps"]).astype(int)
        min_clip_length = annotation_interval * (min_anno_length - 1) + 1
        self.scenes = metadata[metadata["num frames"] >= min_clip_length]

    def __len__(self):
        return len(self.scenes)

    def _get_views(self, idx, rng, num_context_views):
        scene_info = self.scenes.iloc[idx]
        video_path = osp.join(self.ROOT, "SpatialVid/HQ", scene_info["video path"])
        annotation_dir = osp.join(self.ROOT, "SpatialVid/HQ", scene_info["annotation path"])

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            video_length = len(video_reader)
            sample_index, reverse = self.sample_from_video(
                video_length, self.num_views, self.min_interval, self.max_interval, rng
            )
            sample_context_index = sample_index[np.linspace(0, self.num_views - 1, num_context_views, dtype=int)]
            images = video_reader.get_batch(sample_index).asnumpy()

        with open(osp.join(annotation_dir, "caption.json"), 'r') as f:
            captions = json.load(f)
            text_prompt = captions["SceneDescription"]

        # SEMANTIC FINETUNE (guarded): if a labels_dir is configured and a per-clip npz
        # exists, load per-frame semantic label maps aligned to the SAME sampled frame
        # indices we used above. per_frame_labels[v] is None on rows where the clip has
        # no precomputed labels; the downstream 4DPreprocesser only renders the semantic
        # feature when source_views has a "labels" tensor.
        per_frame_labels = None
        if self.labels_dir is not None:
            labels_path = osp.join(self.labels_dir, f"{scene_info['id']}.npz")
            if osp.exists(labels_path):
                with np.load(labels_path) as d:
                    all_labels = d["labels"]                # [N_total, H, W] int8
                per_frame_labels = all_labels[sample_index]  # [num_views, H, W]

        context_views = []
        target_views = []
        for v, rgb_image in enumerate(images):
            timestamp = sample_index[v] - sample_index[0] if not reverse else sample_index[0] - sample_index[v]
            rgb_image, *_ = self._crop_resize_if_necessary(
                rgb_image, (self.width, self.height), rng=rng, info=(idx, v),
            )
            view = dict(
                img=rgb_image,
                dataset="SpatialVID",
                video_name=scene_info["id"],
                image_name=f"frame_{sample_index[v]:06d}",
                is_static=False,
                is_target=sample_index[v] not in sample_context_index,
                timestamp=timestamp,
                prompt=text_prompt,
            )
            if per_frame_labels is not None:
                # NOTE: labels are NOT crop/resized here — they should already match the
                # video resolution (that's how sam3_precompute_labels.py saves them, using
                # load_video with the same width/height as this dataset).
                view["labels"] = per_frame_labels[v]
            if view["is_target"]:
                target_views.append(view)
            else:
                context_views.append(view)
        views = context_views + target_views
        return views
