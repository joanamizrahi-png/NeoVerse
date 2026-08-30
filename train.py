import torch, os
import argparse
from omegaconf import OmegaConf

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline, ModelConfig
from training.utils import DiffusionTrainingModule, launch_training_task
from training.data.datasets.spatialvid import SpatialVID
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_path="models",
        reconstructor_path="models/NeoVerse/reconstructor.ckpt",
        pipeline_kwargs={},
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_exclude_modules=None, lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        semantic_channels: int = 0,   # SEMANTIC FINETUNE: 0 = disabled (default), 16 = enabled
        semantic_expansion_version: int = 1,  # 1 = v3/4/5 in-place grow, 2 = v6 parallel _sem
        distill_lora_path: str = None,        # v6: merge lightx2v distill LoRA at TRAIN time
        distill_lora_alpha: float = 1.0,      # so train + eval share the same 4-step regime
        semantic_loss_weight: float = 4.0,    # weight on the semantic MSE half (RGB half is 1.0)
        semantic_x0_prediction: bool = False,  # v8: sem half predicts the clean latent (x0), not velocity
        semantic_ce_weight: float = 0.0,       # v8 Change 2: decoded-space CE weight (0 = off)
        semantic_ce_sigma_max: float = 0.7,    # apply CE only at timesteps with sigma below this
        semantic_ce_latent_frames: int = 2,    # latent frames to VAE-decode for CE (memory bound)
        semantic_ce_gt_only: bool = False,      # v16 fix 3: class losses only on clips with dense
                                                # human GT (pseudo-labelled clips would teach
                                                # "reproduce the SAM3 hint", cancelling the
                                                # "correct the hint" behaviour GT clips teach).
        semantic_ce_ignore_void: bool = False,  # v16: skip void(0) pixels in the CE. Clips without
                                                # dense GT train against SAM3 output, whose
                                                # "unlabeled"->void regions otherwise teach the
                                                # model to emit void (v15: 57.6% void on SCAND
                                                # at 99.9% real geometry).
        semantic_ce_class_weights: "list | None" = None,  # v19: per-class CE weights
                                                # (14 floats; upweight rare classes like person
                                                # so they carry gradient — the v18 diagnosis:
                                                # person pixels drowned in the mixture).
        pseudo_gt_reliable_classes: "list | None" = None,  # v20: on pseudo-GT clips,
                                                # CE only where GT is one of these class ids
                                                # (supervisor's home domain, e.g. [6,7,12,13]);
                                                # its noisy vegetation/terrain labels never
                                                # touch the loss.
        pseudo_gt_scene_prefixes: "list | None" = None,  # v20: clip-name prefixes counted
                                                # as pseudo-GT (e.g. ["scand_","gnd_","go2w_"]);
                                                # everything else keeps full CE (human GT).
        semantic_seg_weight: float = 0.0,      # v8 Change 3: SAM2 segment-homogeneity weight (0 = off)
        semantic_seg_min_px: int = 0,          # 0 = no size filter (default); >0 enables the confetti guard
        num_semantic_classes: int = 30,        # class-count for the CE head (class-set agnostic)
        semantic_palette_version: int = 1,     # v14 color set (v22+: 2)
        rgb_preservation_weight: float = 0.0,  # v10 candidate: MSE to the frozen vanilla RGB prediction (0 = off)
        rgb_preservation_ramp_steps: int = 0,  # v10d: linear ramp of pres weight over first N steps (0 = constant)
        semantic_head_hidden: int = 64,        # v10e: reader head width
        semantic_head_depth: int = 2,          # v10e: reader head 3x3-conv count (>2 adds dilation ladder)
        snr_gamma: float = 0.0,                # v10 candidate: min-SNR timestep weighting cap (0 = off)
        semantic_analog_bits: bool = False,    # Track B: 4-bit class codes at latent res replace colorize+VAE (semantic_channels must be 4)
        dino_hint_channels: int = 0,           # v25: frozen-DINOv2 hint into the control branch (0 = off; 384 = ViT-S)
    ):
        super().__init__()
        # Load models. If distill_lora_path is set, the distill LoRA is merged
        # into the base DiT BEFORE any expansion/freezing. This is critical to
        # avoid a train/eval regime mismatch — inference uses the 4-step distill
        # LoRA, so training must too, or the semantic head learns behavior that
        # gets scrambled by the distilled sampler at inference.
        self.pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=model_path,
            reconstructor_path=reconstructor_path,
            pipeline_kwargs=pipeline_kwargs,
            lora_path=distill_lora_path,
            lora_alpha=distill_lora_alpha,
            device="cpu",
            torch_dtype=torch.bfloat16,
        )

        # SEMANTIC FINETUNE: expand DiT + control branch to co-denoise semantics.
        # MUST happen AFTER loading pretrained weights (and any distill LoRA merge)
        # but BEFORE any freezing / LoRA. Zero-init on new channels -> step 0
        # behavior matches pretrained RGB model.
        #
        # semantic_expansion_version:
        #   1 = grow patch_embedding / head / control_patch_embedding IN PLACE
        #       (v3/4/5 approach). New channels are extra rows/columns on the
        #       existing weight tensors. If the RGB slice is trained via LoRA,
        #       RGB can drift.
        #   2 = add parallel `_sem` submodules alongside the pretrained ones.
        #       Base modules are UNTOUCHED (RGB path is bit-identical to
        #       pretrained until unfrozen). Semantic gets its own full-rank
        #       weights. This is v6.
        if semantic_channels > 0:
            if semantic_expansion_version == 1:
                from diffsynth.utils.semantics import (
                    expand_dit_for_semantics,
                    expand_control_branch_for_semantics,
                )
                self.pipe.semantic_channels = semantic_channels
                expand_dit_for_semantics(self.pipe.dit, extra=semantic_channels)
                if self.pipe.control_branch is not None:
                    expand_control_branch_for_semantics(self.pipe.control_branch, extra=semantic_channels)
            elif semantic_expansion_version == 2:
                from diffsynth.utils.semantics import (
                    expand_dit_for_semantics_v2,
                    expand_control_branch_for_semantics_v2,
                )
                self.pipe.semantic_channels = semantic_channels
                expand_dit_for_semantics_v2(self.pipe.dit, extra=semantic_channels)
                if self.pipe.control_branch is not None:
                    expand_control_branch_for_semantics_v2(self.pipe.control_branch, extra=semantic_channels)
            else:
                raise ValueError(f"unknown semantic_expansion_version={semantic_expansion_version}")
            # Loss weight for the semantic half of the 32-ch MSE. Default 4.0.
            self.pipe.semantic_loss_weight = float(semantic_loss_weight)
            self.pipe.semantic_x0_prediction = bool(semantic_x0_prediction)
            # v14: colorize/decode follow the configured class count everywhere.
            from diffsynth.utils.semantics import set_active_palette
            set_active_palette(int(num_semantic_classes),
                               version=int(semantic_palette_version))
            # v8 Change 2: decoded-space CE. The head must exist BEFORE
            # freeze_except so `semantic_class_head` in trainable_models can
            # unfreeze it; requires x0-prediction (it reads the clean-latent guess).
            self.pipe.semantic_ce_weight = float(semantic_ce_weight)
            self.pipe.semantic_ce_sigma_max = float(semantic_ce_sigma_max)
            self.pipe.semantic_ce_latent_frames = int(semantic_ce_latent_frames)
            self.pipe.semantic_ce_ignore_void = bool(semantic_ce_ignore_void)
            self.pipe.semantic_ce_gt_only = bool(semantic_ce_gt_only)
            self.pipe.semantic_ce_class_weights = (
                [float(x) for x in semantic_ce_class_weights]
                if semantic_ce_class_weights is not None else None)
            self.pipe.pseudo_gt_reliable_classes = (
                [int(x) for x in pseudo_gt_reliable_classes]
                if pseudo_gt_reliable_classes is not None else None)
            self.pipe.pseudo_gt_scene_prefixes = (
                [str(x) for x in pseudo_gt_scene_prefixes]
                if pseudo_gt_scene_prefixes is not None else None)
            self.pipe.semantic_seg_weight = float(semantic_seg_weight)
            self.pipe.semantic_seg_min_px = int(semantic_seg_min_px)
            self.pipe.rgb_preservation_weight = float(rgb_preservation_weight)
            self.pipe.rgb_preservation_ramp_steps = int(rgb_preservation_ramp_steps)
            self.pipe.snr_gamma = float(snr_gamma)
            self.pipe.semantic_analog_bits = bool(semantic_analog_bits)
            if semantic_analog_bits:
                assert semantic_channels == 4, \
                    "analog bits carries 4 channels; set semantic_channels: 4"
                assert semantic_x0_prediction, \
                    "analog bits requires x0-prediction (bits are targets, not velocities)"
            if semantic_seg_weight > 0.0:
                assert semantic_ce_weight > 0.0, \
                    "semantic_seg_weight rides the CE head's decoded logits; set semantic_ce_weight too"
            if semantic_ce_weight > 0.0 and not semantic_analog_bits:
                # (analog-bits CE lives at latent resolution and needs no head)
                assert semantic_x0_prediction, \
                    "semantic_ce_weight requires semantic_x0_prediction: true"
                from diffsynth.utils.semantics import SemanticClassHead
                self.pipe.semantic_class_head = SemanticClassHead(
                    num_classes=int(num_semantic_classes),
                    hidden=int(semantic_head_hidden),
                    depth=int(semantic_head_depth))
            # v25: DINO hint. Attach AFTER the v2 expansion (needs the Split
            # wrapper) and BEFORE freeze_except (so the yaml's trainable_models
            # entry `control_branch.control_patch_embedding.dino_proj` lights
            # the zero-init projection up). Inert at 0.
            if int(dino_hint_channels) > 0:
                assert semantic_expansion_version == 2, \
                    "dino_hint_channels requires semantic_expansion_version: 2"
                from diffsynth.utils.dino_hint import attach_dino_hint
                attach_dino_hint(self.pipe.control_branch,
                                 dino_dim=int(dino_hint_channels))
                self.pipe.dino_hint_channels = int(dino_hint_channels)

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add LoRA to the base models
        if lora_base_model is not None:
            if lora_exclude_modules is not None:
                lora_exclude_modules = lora_exclude_modules.split(",")
                if len(lora_exclude_modules) == 1:
                    lora_exclude_modules = lora_exclude_modules[0]

            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                exclude_modules=lora_exclude_modules,
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Freeze untrainable models.
        # freeze_except only sets matched params to requires_grad=True; it does
        # NOT explicitly freeze non-matched params in trainable models. For v6
        # (surgical: only .sem submodules trainable in control_branch), we
        # explicitly freeze all control_branch params first so freeze_except's
        # regex only lights up the .sem ones.
        if semantic_expansion_version == 2 and self.pipe.control_branch is not None:
            for p in self.pipe.control_branch.parameters():
                p.requires_grad = False
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","), lora_base_model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data[0]["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": None,
            "height": data[0]["img"].shape[-2],
            "width": data[0]["img"].shape[-1],
            "num_frames": len(data),
            "source_views": data,
            "control_scale": 1,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.debug:
        config.num_workers = 0
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    args = config

    print(f"Preparing dataset {args.train_dataset}")
    dataset = eval(args.train_dataset)
    model = WanTrainingModule(
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        pipeline_kwargs=args.pipeline_kwargs,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_exclude_modules=args.lora_exclude_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        semantic_channels=int(getattr(args, "semantic_channels", 0)),
        semantic_expansion_version=int(getattr(args, "semantic_expansion_version", 1)),
        distill_lora_path=getattr(args, "distill_lora_path", None),
        distill_lora_alpha=float(getattr(args, "distill_lora_alpha", 1.0)),
        semantic_loss_weight=float(getattr(args, "semantic_loss_weight", 4.0)),
        semantic_x0_prediction=bool(getattr(args, "semantic_x0_prediction", False)),
        semantic_ce_weight=float(getattr(args, "semantic_ce_weight", 0.0)),
        semantic_ce_sigma_max=float(getattr(args, "semantic_ce_sigma_max", 0.7)),
        semantic_ce_latent_frames=int(getattr(args, "semantic_ce_latent_frames", 2)),
        semantic_ce_ignore_void=bool(getattr(args, "semantic_ce_ignore_void", False)),
        semantic_ce_gt_only=bool(getattr(args, "semantic_ce_gt_only", False)),
        semantic_ce_class_weights=getattr(args, "semantic_ce_class_weights", None),
        semantic_seg_weight=float(getattr(args, "semantic_seg_weight", 0.0)),
        semantic_seg_min_px=int(getattr(args, "semantic_seg_min_px", 0)),
        rgb_preservation_weight=float(getattr(args, "rgb_preservation_weight", 0.0)),
        rgb_preservation_ramp_steps=int(getattr(args, "rgb_preservation_ramp_steps", 0)),
        semantic_head_hidden=int(getattr(args, "semantic_head_hidden", 64)),
        semantic_head_depth=int(getattr(args, "semantic_head_depth", 2)),
        snr_gamma=float(getattr(args, "snr_gamma", 0.0)),
        semantic_analog_bits=bool(getattr(args, "semantic_analog_bits", False)),
        num_semantic_classes=int(getattr(args, "num_semantic_classes", 30)),
        semantic_palette_version=int(getattr(args, "semantic_palette_version", 1)),
        dino_hint_channels=int(getattr(args, "dino_hint_channels", 0)),
    )
    # SEMANTIC FINETUNE debug: set `debug_save_root: /path/dir` in the config to make
    # 4DPreprocesser dump gt.mp4 + gt_semantic_hint.mp4 + gt_semantic_target.mp4 for
    # every training iteration under <root>/<dataset>/<video_name>/. Set null in the
    # yaml to disable. Cheap side-effect on top of an existing pipe knob.
    debug_save_root = getattr(args, "debug_save_root", None)
    if debug_save_root:
        model.pipe.save_root = debug_save_root
        os.makedirs(debug_save_root, exist_ok=True)
        print(f"[debug] pipe.save_root = {debug_save_root}")

    # WARM-START: overlay a previous finetune checkpoint on top of the initialized
    # model. Guarded by the yaml `pretrained_path` field (null / missing => no-op).
    # v3's checkpoints saved trainable params only, with `remove_prefix_in_ckpt: pipe.`
    # stripping the "pipe." prefix -- re-add it so keys line up with WanTrainingModule.
    # strict=False because the checkpoint intentionally omits frozen params.
    pretrained_path = getattr(args, "pretrained_path", None)
    if pretrained_path is not None:
        from safetensors.torch import load_file
        sd = {f"pipe.{k}": v for k, v in load_file(pretrained_path).items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[pretrained_path] loaded {len(sd)} tensors from {pretrained_path}")
        print(f"[pretrained_path] missing={len(missing)}, unexpected={len(unexpected)}")
        if unexpected:
            print(f"[pretrained_path] first unexpected: {unexpected[0]}")
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, optimizer, scheduler, args
    )
