"""Read-only PI0.5 attention diagnostics for single-environment evaluation.

Capture post-RoPE queries and cached keys at one Gemma layer. The original
attention backend still computes the policy output; only selected query rows
are recomputed in float32 for visualization. No gradients or extra policy
forward passes are used.
"""

from __future__ import annotations

import copy
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class AttentionVideoConfig:
    source: str = "action"
    layer: int = -1
    camera: int = 0
    cameras: str = ""  # Optional comma-separated indices; overrides camera.
    snapshot_every: int = 0  # Every N predictions, plus the last; 0 disables PNGs.
    denoise: str = "mean"
    alpha: float = 0.82
    vmax: float = 0.0  # zero: relative to the maximum in this prediction's map
    font_path: str | None = None
    record_routing: bool = False  # Save all-key group masses per head and denoising pass.

    def __post_init__(self):
        if self.source not in {"action", "vlm_prompt"}:
            raise ValueError("ATTENTION_SOURCE must be action or vlm_prompt")
        if self.denoise not in {"mean", "first", "last"}:
            raise ValueError("ATTENTION_DENOISE must be mean, first, or last")
        if self.camera < 0:
            raise ValueError("ATTENTION_CAMERA must be a non-negative camera index")
        indices = self.camera_indices
        if not 1 <= len(indices) <= 2 or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("ATTENTION_CAMERAS must contain one or two distinct non-negative indices")
        if self.snapshot_every < 0 or (self.snapshot_every and len(indices) != 2):
            raise ValueError("ATTENTION_SNAPSHOT_EVERY requires two cameras and a non-negative integer")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("ATTENTION_ALPHA must be in [0, 1]")
        if not math.isfinite(self.vmax) or self.vmax < 0:
            raise ValueError("ATTENTION_VMAX must be finite and non-negative")

    @property
    def camera_indices(self):
        return (
            tuple(int(value.strip()) for value in self.cameras.split(",")) if self.cameras else (self.camera,)
        )

    @classmethod
    def from_env(cls):
        return cls(
            source=os.environ.get("ATTENTION_SOURCE", "action"),
            layer=int(os.environ.get("ATTENTION_LAYER", "-1")),
            camera=int(os.environ.get("ATTENTION_CAMERA", "0")),
            cameras=os.environ.get("ATTENTION_CAMERAS", ""),
            snapshot_every=int(os.environ.get("ATTENTION_SNAPSHOT_EVERY", "0")),
            denoise=os.environ.get("ATTENTION_DENOISE", "mean"),
            alpha=float(os.environ.get("ATTENTION_ALPHA", "0.82")),
            vmax=float(os.environ.get("ATTENTION_VMAX", "0")),
            font_path=os.environ.get("ATTENTION_FONT_PATH") or None,
            record_routing=os.environ.get("ATTENTION_RECORD_ROUTING", "0") == "1",
        )


@lru_cache(maxsize=8)
def load_attention_font(font_path=None, size=14):
    """Resolve actual Times New Roman; never silently substitute another family."""
    candidates = [str(Path(font_path).expanduser())] if font_path else []
    if not font_path:
        try:
            matched = subprocess.run(
                ["fc-match", "-f", "%{file}", "Times New Roman"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            ).stdout.strip()
            if matched:
                candidates.append(matched)
        except (OSError, subprocess.TimeoutExpired):
            pass
        candidates.extend(
            [
                "Times New Roman.ttf",
                "times.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
                "/usr/local/share/fonts/times.ttf",
                str(Path.home() / ".local/share/fonts/times.ttf"),
                "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                "/Library/Fonts/Microsoft/Times New Roman.ttf",
                str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/times.ttf"),
            ]
        )
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, size)
        except OSError:
            continue
        family = "".join(character.lower() for character in font.getname()[0] if character.isalnum())
        if family.startswith("timesnewroman"):
            return font
    raise ValueError(
        "Times New Roman was not found. Install it or set ATTENTION_FONT_PATH=/absolute/path/times.ttf "
        "(replot CLI: --font-path=/absolute/path/times.ttf). Other fonts are not substituted."
    )


@torch.no_grad()
def selected_attention(query, key, mask, scaling, start, stop, *, average_heads=True):
    """Average selected query rows, with softmax over ALL keys; optionally retain heads."""
    if not 0 <= start < stop <= query.shape[-2]:
        raise ValueError(f"Invalid query range [{start}, {stop}) for {query.shape[-2]} tokens")
    query = query.detach()[:, :, start:stop].float()
    key = key.detach().float()
    if query.shape[1] % key.shape[1]:
        raise ValueError("Query head count must be divisible by KV head count")
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    logits = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if mask is None or mask.ndim != 4:
        raise ValueError("PI0.5 visualization requires its explicit 4D visibility mask")
    selected_mask = mask[..., start:stop, : key.shape[-2]]
    if selected_mask.dtype == torch.bool:
        allowed = selected_mask
        logits = logits.masked_fill(~allowed, -torch.inf)
    else:
        # PI0.5 uses a finite float32 negative sentinel for disallowed positions.
        allowed = selected_mask > -1e4
        logits = logits + selected_mask.float()
    if not allowed.any(dim=-1).all():
        raise ValueError("Selected attention queries include a fully masked/padded row")
    weights = torch.softmax(logits, dim=-1).masked_fill(~allowed, 0)
    if not torch.isfinite(weights).all():
        raise ValueError("Non-finite attention probabilities")
    return weights.mean(dim=(1, 2)) if average_heads else weights.mean(dim=2)


def _diagnostic_attention(module, query, key, value, attention_mask, **kwargs):
    recorder = module._pi05_attention_recorder
    recorder.observe(query, key, attention_mask, kwargs.get("scaling", module.scaling))
    return module._pi05_attention_original_backend(module, query, key, value, attention_mask, **kwargs)


def overlay_attention(rgb, patch_map, alpha=0.82, vmax=0.0):
    """Blue/purple low values, bright red/yellow high values; preserve spatial order."""
    patch_map = np.asarray(patch_map, dtype=np.float32)
    if not np.isfinite(patch_map).all() or (patch_map < 0).any():
        raise ValueError("Attention map must contain finite non-negative probabilities")
    maximum = float(vmax or patch_map.max())
    intensity = np.clip(patch_map / maximum, 0, 1) if maximum > 0 else np.zeros_like(patch_map)
    heat = np.asarray(
        Image.fromarray(intensity).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)
    )
    stops = [0.0, 0.25, 0.50, 0.72, 0.90, 1.0]
    palette = np.array(
        [
            [36, 18, 100],
            [45, 55, 160],
            [145, 28, 150],
            [248, 40, 40],
            [255, 155, 15],
            [255, 255, 65],
        ]
    )
    color = np.stack([np.interp(heat, stops, channel) for channel in palette.T], axis=-1)
    # Uniform opacity makes the blue/purple background visible as well as peaks.
    return np.clip(rgb * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


class PI05AttentionRecorder:
    """Temporary instrumentation. Restore all model methods/configs on exit."""

    def __init__(self, policy, config: AttentionVideoConfig):
        self.eval_policy = policy
        self.policy = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        self.config = config
        self.font = None
        self._restore = []
        self.reset_episode()

    def _set(self, obj, name, value):
        self._restore.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    def __enter__(self):
        try:
            self._attach()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for obj, name, existed, old in reversed(self._restore):
            if existed:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self._restore.clear()

    def _attach(self):
        from transformers import AttentionInterface
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma.modeling_gemma import eager_attention_forward

        self.font = load_attention_font(self.config.font_path)
        policy = self.policy
        if getattr(policy, "name", None) != "pi05":
            raise ValueError("Attention videos currently support PI0.5 only")
        if policy.config.compile_model or policy._rtc_enabled():
            raise ValueError("Attention videos require compile_model=false and RTC disabled")
        if self.config.source == "vlm_prompt" and not policy.config.num_vlm_prompt_tokens:
            raise ValueError("This checkpoint has no VLM prompts; use ATTENTION_SOURCE=action")
        model = policy.model
        composite = model.paligemma_with_expert
        decoder = (
            composite.gemma_expert.model
            if self.config.source == "action"
            else composite.paligemma.model.language_model
        )
        self.layer = self.config.layer
        if self.layer < 0:
            self.layer += len(decoder.layers)
        if not 0 <= self.layer < len(decoder.layers):
            raise ValueError(f"ATTENTION_LAYER out of range for {len(decoder.layers)} layers")
        attention = decoder.layers[self.layer].self_attn
        implementation = attention.config._attn_implementation
        if implementation not in {"sdpa", "eager"}:
            raise ValueError(f"Unsupported attention backend for diagnostics: {implementation}")
        backend = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
        AttentionInterface.register("pi05_eval_visualization", _diagnostic_attention)
        config = copy.copy(attention.config)
        config._attn_implementation = "pi05_eval_visualization"
        self._set(attention, "config", config)
        self._set(attention, "_pi05_attention_original_backend", backend)
        self._set(attention, "_pi05_attention_recorder", self)
        self.patch_size = composite.paligemma.config.vision_config.patch_size

        original_prefix = model.embed_prefix
        original_select = policy.select_action

        def embed_prefix(images, img_masks, tokens, masks):
            result = original_prefix(images, img_masks, tokens, masks)
            self.begin_prediction(images, img_masks, masks, result[0].shape[1], tokens)
            return result

        def select_action(batch):
            self.step += 1
            self.new_prediction = False
            image_keys = [key for key in policy.config.image_features if key in batch]
            image_keys += [key for key in policy.config.image_features if key not in batch]
            self.camera_features = [
                image_keys[i] if i < len(image_keys) else None for i in self.config.camera_indices
            ]
            self.camera_feature = self.camera_features[0]
            result = original_select(batch)
            if self.new_prediction:
                self.finish_prediction()
            return result

        self._set(model, "embed_prefix", embed_prefix)
        self._set(policy, "select_action", select_action)
        self._set(self.eval_policy, "_eval_attention_visualizer", self)

    def reset_episode(self):
        self.step = -1
        self.prediction_step = -1
        self.new_prediction = False
        self.latest_map = None
        self.image = None
        self.maps = []
        self.steps = []
        self.counts = []
        self.rows = []
        self.latest_camera_maps = None
        self.source_images = None
        self.camera_maps = []
        self.camera_images = []
        self.routing_records = []
        self.routing_rows = []

    def begin_prediction(self, images, img_masks, language_mask, prefix_length, tokens=None):
        if images[0].shape[0] != 1:
            raise ValueError("Attention videos require eval.batch_size=1")
        indices = self.config.camera_indices
        for index in indices:
            if index >= len(images) or not bool(img_masks[index][0]):
                raise ValueError(f"Camera {index} is missing/padded; choose a present policy camera")
        sizes = []
        for image in images:
            height, width = image.shape[-2:]
            if height % self.patch_size or width % self.patch_size:
                raise ValueError("Image dimensions must be multiples of vision patch size")
            sizes.append((height // self.patch_size, width // self.patch_size))
        lengths = [h * w for h, w in sizes]
        expected = sum(lengths) + language_mask.shape[1] + self.policy.config.num_vlm_prompt_tokens
        if prefix_length != expected:
            raise ValueError("Unexpected image-token layout; refusing to misalign attention and pixels")
        self.prefix_length = prefix_length
        self.grid = sizes[indices[0]]
        if any(sizes[i] != self.grid for i in indices):
            raise ValueError("Selected cameras must have the same policy-input patch grid")
        self.image_ranges = [(sum(lengths[:i]), sum(lengths[: i + 1])) for i in indices]
        if self.config.record_routing:
            from lerobot.policies.pi05.attention_routing import build_key_groups

            cfg = self.policy.config
            self.routing_names, self.routing_group_ids = build_key_groups(
                lengths,
                language_mask[0].detach().cpu().numpy(),
                cfg.num_vlm_prompt_tokens,
                cfg.num_prompt_tokens if self.config.source == "action" else 0,
                cfg.chunk_size if self.config.source == "action" else 0,
                min(cfg.n_action_steps, cfg.chunk_size),
            )
            self.routing_token_ids = np.full(len(self.routing_group_ids), -1, dtype=np.int64)
            if tokens is not None:
                self.routing_token_ids[sum(lengths) : sum(lengths) + tokens.shape[1]] = (
                    tokens[0].detach().cpu().numpy()
                )
            self.routing_rows = []
        # These are the EXACT oriented/resized/padded pixels passed to SigLIP.
        self.source_images = np.stack(
            [
                ((images[i][0].detach().float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5)
                .clip(0, 255)
                .round()
                .astype(np.uint8)
                for i in indices
            ]
        )
        self.image = self.source_images[0]
        self.prediction_step = self.step
        self.rows = []
        self.new_prediction = True

    def observe(self, query, key, mask, scaling):
        if not self.new_prediction:
            raise RuntimeError("Attention captured outside an action prediction")
        if self.config.source == "action":
            start = self.policy.config.num_prompt_tokens
            stop = start + min(self.policy.config.n_action_steps, self.policy.config.chunk_size)
            if key.shape[-2] != self.prefix_length + query.shape[-2]:
                raise ValueError("Expected image/language prefix followed by cached action suffix")
        else:
            start = self.prefix_length - self.policy.config.num_vlm_prompt_tokens
            stop = self.prefix_length
        if self.config.record_routing:
            from lerobot.policies.pi05.attention_routing import summarize_routing

            head_probabilities = selected_attention(
                query, key, mask, scaling, start, stop, average_heads=False
            )
            self.routing_rows.append(
                summarize_routing(
                    head_probabilities, mask, start, stop, self.routing_group_ids, self.routing_names
                )
            )
            probabilities = head_probabilities.mean(dim=1)
        else:
            probabilities = selected_attention(query, key, mask, scaling, start, stop)
        # Softmax includes language, prompts, other cameras, and action tokens.
        # Do not renormalize on just the chosen camera.
        self.rows.append(
            np.stack([probabilities[0, start:stop].cpu().numpy() for start, stop in self.image_ranges])
        )

    def finish_prediction(self):
        if not self.rows:
            raise RuntimeError("No attention was captured; refusing to generate an empty heatmap video")
        if self.config.denoise == "first":
            values = self.rows[0]
        elif self.config.denoise == "last":
            values = self.rows[-1]
        else:
            values = np.mean(self.rows, axis=0)
        self.latest_camera_maps = values.reshape(len(self.config.camera_indices), *self.grid).copy()
        self.latest_map = self.latest_camera_maps[0]
        self.maps.append(self.latest_map)
        if len(self.config.camera_indices) == 2:
            self.camera_maps.append(self.latest_camera_maps)
            self.camera_images.append(self.source_images.copy())
        self.steps.append(self.prediction_step)
        self.counts.append(len(self.rows))
        if self.config.record_routing:
            self.routing_records.append(
                {
                    "head_mass": np.stack([row[0] for row in self.routing_rows]),
                    "visible_key_counts": np.stack([row[1] for row in self.routing_rows]),
                    "token_mass": np.stack([row[2] for row in self.routing_rows]),
                    "key_group_ids": self.routing_group_ids.copy(),
                    "token_ids": self.routing_token_ids.copy(),
                }
            )
        self.new_prediction = False

    def routing_arrays(self):
        """Compact sufficient statistics, including every pass irrespective of display selection."""
        if not self.config.record_routing:
            return {}
        if not self.routing_records:
            raise RuntimeError("Routing requested but no routing observations were captured")
        return {
            "routing_group_names": np.asarray(self.routing_names),
            **{
                f"routing_{key}": np.stack([record[key] for record in self.routing_records])
                for key in self.routing_records[0]
            },
        }

    def metadata(self):
        return {
            **asdict(self.config),
            "camera": self.config.camera_indices[0],
            "font_family": self.font.getname()[0],
            "resolved_font_path": str(self.font.path),
            "colormap": "blue_purple_red_yellow",
            "camera_feature": self.camera_feature,
            "camera_indices": list(self.config.camera_indices),
            "camera_features": self.camera_features,
            "layout": "two_camera_grid" if len(self.config.camera_indices) == 2 else "live_input_overlay",
            "resolved_layer_zero_based": self.layer,
            "query_tokens": "executed action-token prefix"
            if self.config.source == "action"
            else "VLM prompts",
            "action_query_count": min(self.policy.config.n_action_steps, self.policy.config.chunk_size),
            "heads": "all, arithmetic mean",
            "denoise_note": "VLM prefix is evaluated once per prediction"
            if self.config.source == "vlm_prompt"
            else "flow denoising passes",
            "normalization": "softmax over all visible keys before camera selection",
            "display_scale": "fixed"
            if self.config.vmax
            else "shared maximum over selected cameras per prediction",
            "pixel_alignment": "exact oriented/resized/padded policy input; no additional flip",
            "time_alignment": "maps held on their exact source images during queued actions",
            "interpretation": "qualitative attention routing, not segmentation, causal attribution, or proof of effectiveness",
            "routing_schema": {
                "version": 1,
                "head_mass_axes": ["prediction", "denoising_pass", "head", "key_group"],
                "token_mass_axes": ["prediction", "denoising_pass", "key_position"],
                "visible_key_counts_axes": ["prediction", "denoising_pass", "key_group"],
                "query_reduction": "arithmetic mean over selected query rows",
                "token_mass_head_reduction": "arithmetic mean over all query heads",
                "pass_order": "actual forward-call order; 0 is the first pass",
                "language_state": "task, quantized state, template and special tokens; padding separate",
                "token_ids": "actual tokenizer IDs at language positions; -1 for other modalities",
            }
            if self.config.record_routing
            else None,
        }

    def render_frames(self, frames):
        if len(frames) != 1:
            raise ValueError("Attention rendering requires a single environment")
        if len(self.config.camera_indices) == 2:
            return render_camera_attention_frame(
                self.source_images,
                self.latest_camera_maps,
                self.config,
                self.font,
                self.layer,
                self.step,
                self.prediction_step,
            )[None]
        return render_attention_frame(
            frames[0],
            self.image,
            self.latest_map,
            self.config,
            self.font,
            self.layer,
            self.step,
            self.prediction_step,
        )[None]


def render_attention_frame(live_frame, source_image, patch_map, config, font, layer, step, prediction_step):
    """Shared layout for online videos and offline restyling of saved recordings."""
    side, header = 256, 24
    canvas = Image.new("RGB", (side * 3, 320), "white")
    draw = ImageDraw.Draw(canvas)
    live = Image.fromarray(live_frame).resize((side, side), Image.Resampling.BILINEAR)
    canvas.paste(live, (0, header))
    for column, title in enumerate(("Live environment", "Policy input (source frame)", "Attention overlay")):
        draw.text((column * side + 8, 5), title, fill="#282828", font=font)
    if patch_map is None:
        draw.text((side + 12, 130), "Waiting for first prediction", fill="#666666", font=font)
    else:
        overlay = overlay_attention(source_image, patch_map, config.alpha, config.vmax)
        for column, rgb in ((1, source_image), (2, overlay)):
            canvas.paste(
                Image.fromarray(rgb).resize((side, side), Image.Resampling.BILINEAR),
                (column * side, header),
            )
        mass = float(patch_map.sum())
        maximum = config.vmax or float(patch_map.max())
        draw.text(
            (8, 284),
            f"{config.source} -> image | layer {layer} | camera {config.camera_indices[0]} | image mass {mass:.3f}",
            fill="#333333",
            font=font,
        )
        scale = "fixed" if config.vmax else "relative"
        draw.text(
            (8, 301),
            f"Input step {prediction_step}; queued-action age {step - prediction_step} | {scale} color range 0..{maximum:.4g}",
            fill="#555555",
            font=font,
        )
    return np.asarray(canvas)


def render_camera_attention_frame(source_images, patch_maps, config, font, layer, step, prediction_step):
    """Two synchronized camera rows with one shared probability scale; no live-frame misalignment."""
    side, row_height, footer_height = 320, 352, 80
    canvas = Image.new("RGB", (2 * side, 2 * row_height + footer_height), "white")
    draw = ImageDraw.Draw(canvas)
    for row, camera in enumerate(config.camera_indices):
        draw.text((8, row * row_height + 5), f"Camera {camera} | policy input", fill="#282828", font=font)
        draw.text(
            (side + 8, row * row_height + 5),
            "Prompt attention" if config.source == "vlm_prompt" else "Action attention",
            fill="#282828",
            font=font,
        )
    if patch_maps is None:
        draw.text((12, 150), "Waiting for first prediction", fill="#666666", font=font)
        return np.asarray(canvas)
    maximum = config.vmax or float(np.max(patch_maps))
    for row, (rgb, patch_map) in enumerate(zip(source_images, patch_maps, strict=True)):
        # If both maps are zero, the overlay uses the uniform low-value color.
        overlay = overlay_attention(rgb, patch_map, config.alpha, maximum)
        for col, pixels in enumerate((rgb, overlay)):
            canvas.paste(
                Image.fromarray(pixels).resize((side, side), Image.Resampling.BILINEAR),
                (col * side, row * row_height + 24),
            )
    footer = 2 * row_height
    draw.text(
        (8, footer + 2),
        f"{config.source} -> image | layer {layer} | input step {prediction_step} | age {step - prediction_step}",
        fill="#333333",
        font=font,
    )
    masses = " | ".join(
        f"camera {i} mass {float(m.sum()):.4f}"
        for i, m in zip(config.camera_indices, patch_maps, strict=True)
    )
    draw.text((8, footer + 20), masses, fill="#333333", font=font)
    scale = "fixed" if config.vmax else "shared per prediction"
    draw.text((8, footer + 38), f"Patch probability: 0 to {maximum:.4g} ({scale})", fill="#555555", font=font)
    bar = overlay_attention(
        np.zeros((10, 624, 3), dtype=np.uint8), np.linspace(0, 1, 624)[None], alpha=1, vmax=1
    )
    canvas.paste(Image.fromarray(bar), (8, footer + 60))
    return np.asarray(canvas)
