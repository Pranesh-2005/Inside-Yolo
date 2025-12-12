# ==========================================================
#  YOLOv8n Visualizer — Inside Object Detection (Advanced)
#  - Uses Ultralytics YOLOv8n (small, CPU-friendly)
#  - Step 0: Input image
#  - Step 1: Early feature activation (edges/textures)
#  - Step 2: Middle feature activation (parts/shapes)
#  - Step 3: Late feature activation (objects)
#  - Step 4: Final detections (boxes + labels)
#  - Activation-CAM overlay (late layer heatmap on image)
#  - Channel explorer for late layer (view individual channels)
# ==========================================================

import gradio as gr
import torch
import numpy as np
from PIL import Image

from ultralytics import YOLO

# ------------------- GLOBALS -------------------

DEVICE = "cpu"
MODEL = None
FEATURE_MAPS = {}  # {layer_name: tensor(B,C,H,W)}


# ------------------- MODEL LOADING -------------------

def load_model():
    """
    Load YOLOv8n once and register forward hooks
    on backbone/head layers to capture feature maps.
    """
    global MODEL, FEATURE_MAPS
    if MODEL is not None:
        return MODEL

    model = YOLO("yolov8n.pt")

    # ensure on CPU
    if hasattr(model, "to"):
        model.to(DEVICE)
    else:
        model.model.to(DEVICE)
    model.model.eval()

    FEATURE_MAPS = {}

    # model.model.model is the list of modules (backbone + head)
    for idx, layer in enumerate(model.model.model):
        def make_hook(name):
            def hook(module, inputs, output):
                with torch.no_grad():
                    out = output
                    if isinstance(out, (list, tuple)):
                        out = next(
                            (o for o in out if isinstance(o, torch.Tensor)),
                            None
                        )
                    if isinstance(out, torch.Tensor):
                        FEATURE_MAPS[name] = out.detach().cpu()
            return hook

        layer.register_forward_hook(make_hook(str(idx)))

    MODEL = model
    return MODEL


# ------------------- FEATURE MAP UTILITIES -------------------

def tensor_to_heatmap(fm, out_size):
    """
    Convert a feature map tensor (C,H,W) to a grayscale heatmap PIL image.
    """
    if fm.ndim != 3:
        return None

    fm_np = fm.numpy().astype(np.float32)
    heat = fm_np.mean(axis=0)  # (H,W)

    if not np.any(heat):
        heat = np.zeros_like(heat)
    else:
        heat -= heat.min()
        maxv = heat.max()
        if maxv > 0:
            heat /= maxv

    img = (heat * 255).astype("uint8")
    pil = Image.fromarray(img, mode="L")
    pil = pil.resize(out_size, Image.NEAREST)
    return pil


def heat_array_from_fm(fm):
    """
    Same as tensor_to_heatmap but returns 0..1 numpy array (H,W).
    """
    fm_np = fm.numpy().astype(np.float32)
    heat = fm_np.mean(axis=0)
    if not np.any(heat):
        heat = np.zeros_like(heat)
    else:
        heat -= heat.min()
        maxv = heat.max()
        if maxv > 0:
            heat /= maxv
    return heat


def pick_feature_maps():
    """
    Choose three feature maps: early, middle, late.
    FEATURE_MAPS keys are stringified indices "0", "1", ...
    Returns list[(name, fm_tensor(C,H,W))]
    """
    if not FEATURE_MAPS:
        return []

    keys = sorted(FEATURE_MAPS.keys(), key=lambda x: int(x))
    fms = []
    for k in keys:
        t = FEATURE_MAPS[k]
        if isinstance(t, torch.Tensor) and t.ndim == 4:
            fms.append((k, t[0]))  # (name, (C,H,W))

    if not fms:
        return []

    idxs = [0, len(fms) // 2, len(fms) - 1]
    idxs = sorted(set(idxs))

    chosen = []
    for i in idxs:
        chosen.append(fms[i])
    return chosen


def make_cam_overlay(base_pil, heat_01):
    """
    Build a simple activation-CAM overlay (heatmap over image).
    heat_01: numpy (H_fm, W_fm) in [0,1], resized to image size.
    """
    base = np.array(base_pil).astype(np.float32) / 255.0  # H,W,3

    h, w = base.shape[:2]
    heat_resized = Image.fromarray((heat_01 * 255).astype("uint8"), mode="L").resize(
        (w, h), Image.BILINEAR
    )
    heat_resized = np.array(heat_resized).astype(np.float32) / 255.0  # H,W

    # simple blue→red colormap
    r = heat_resized
    g = np.zeros_like(heat_resized)
    b = 1.0 - heat_resized
    cam = np.stack([r, g, b], axis=-1)  # H,W,3

    alpha = 0.45
    blended = (1 - alpha) * base + alpha * cam
    blended = np.clip(blended * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(blended)


def single_channel_heatmap(channel_2d, out_size):
    """
    Convert 2D channel to grayscale PIL heatmap.
    """
    arr = channel_2d.astype(np.float32)
    if not np.any(arr):
        arr = np.zeros_like(arr)
    else:
        arr -= arr.min()
        maxv = arr.max()
        if maxv > 0:
            arr /= maxv

    img = (arr * 255).astype("uint8")
    pil = Image.fromarray(img, mode="L")
    pil = pil.resize(out_size, Image.NEAREST)
    return pil


# ------------------- MAIN ANALYSIS FUNCTION -------------------

def analyze_yolo(img, conf_thres, iou_thres, simple_mode):
    """
    Run YOLOv8n on input image and produce:
      - detection image with boxes
      - early/mid/late feature map heatmaps
      - activation-CAM overlay
      - channel explorer state
      - explanation markdown
    """
    if img is None:
        return (
            None,  # det img
            None,  # early
            None,  # mid
            None,  # late
            None,  # cam overlay
            "⚠️ Please upload an image first.",
            "",    # channel info
            gr.update(maximum=0, value=0),
            None,  # channel heatmap
            {}     # state
        )

    model = load_model()
    FEATURE_MAPS.clear()

    pil = img
    conf = float(conf_thres)
    iou = float(iou_thres)

    with torch.no_grad():
        results = model(pil, conf=conf, iou=iou, verbose=False)

    res = results[0]
    det_np = res.plot()  # numpy HWC
    det_img = Image.fromarray(det_np)

    chosen = pick_feature_maps()
    W, H = pil.size
    heatmaps = [None, None, None]
    late_fm_np = None
    late_name = None

    for idx, item in enumerate(chosen):
        name, fm = item  # fm: (C,H,W)
        hm = tensor_to_heatmap(fm, (W, H))
        heatmaps[idx] = hm
        if idx == len(chosen) - 1:
            late_fm_np = fm.numpy().astype(np.float32)  # (C,H,W)
            late_name = name

    # Activation-CAM overlay (using late feature map mean)
    cam_overlay = None
    channel_slider_update = gr.update(maximum=0, value=0)
    channel_info = ""
    channel_heatmap_img = None
    state = {}

    if late_fm_np is not None:
        C, H_fm, W_fm = late_fm_np.shape
        late_fm_tensor = torch.from_numpy(late_fm_np)
        heat_01 = heat_array_from_fm(late_fm_tensor)
        cam_overlay = make_cam_overlay(pil, heat_01)

        # Channel explorer: compute mean abs activation per channel
        means = np.mean(np.abs(late_fm_np), axis=(1, 2))  # (C,)
        order = np.argsort(means)[::-1]
        top_k = order[: min(8, C)].tolist()

        channel_info = (
            f"Late layer **{late_name}** feature map: {C} channels of size {H_fm}×{W_fm}.\n"
            f"Top active channels (by mean |activation|): {top_k}"
        )

        # default channel = strongest
        default_ch = int(top_k[0]) if top_k else 0
        channel_slider_update = gr.update(maximum=C - 1, value=default_ch)

        # build heatmap for default channel
        default_ch_map = late_fm_np[default_ch]
        channel_heatmap_img = single_channel_heatmap(default_ch_map, (W, H))

        # state for slider changes
        state = {
            "late_fm": late_fm_np,
            "W": W,
            "H": H,
        }

    # Explanation
    if simple_mode:
        explanation = (
            "🧒 **Simple explanation of what you see:**\n\n"
            "- **Step 0 – Input image:** your original picture.\n"
            "- **Step 1 – Early layer heatmap:** the model sees edges and tiny details.\n"
            "- **Step 2 – Middle layer heatmap:** it starts seeing parts of objects and shapes.\n"
            "- **Step 3 – Late layer heatmap:** it focuses on full objects and important regions.\n"
            "- **Activation overlay:** colored map (blue→red) over the image showing *where* the model\n"
            "  is looking the most in the final stage.\n"
            "- **Channel explorer:** each channel is like a tiny specialist (e.g., vertical lines,\n"
            "  corners, or specific textures). You can slide through channels to see different patterns.\n"
        )
    else:
        explanation = (
            "🔬 **Technical explanation:**\n\n"
            "- We run **YOLOv8n** (Ultralytics) on CPU.\n"
            "- Forward hooks capture internal feature maps from several backbone/head blocks.\n"
            "- For each chosen layer, we take `(C,H,W)` and average over channels to get a 2D activation\n"
            "  map `(H,W)`, normalize it, and upsample it to image resolution.\n"
            "- Early ≈ low-level features; Middle ≈ mid-level parts; Late ≈ high-level object-centric\n"
            "  features.\n"
            "- The activation overlay is a CAM-style visualization built from the **mean late-layer\n"
            "  activation**, colored and blended with the original image (not full gradient-based Grad-CAM,\n"
            "  but an activation-based approximation).\n"
            "- In the channel explorer, channels are ranked by mean |activation|, and you can inspect each\n"
            "  channel separately as a grayscale map, revealing different spatial patterns.\n"
        )

    # Add feature map shapes if we have them
    if chosen:
        explanation += "\n**Captured feature map shapes (C,H,W):**\n"
        for name, fm in chosen:
            explanation += f"- Layer {name}: {tuple(fm.shape)}\n"

    return (
        det_img,
        heatmaps[0],
        heatmaps[1],
        heatmaps[2],
        cam_overlay,
        explanation,
        channel_info,
        channel_slider_update,
        channel_heatmap_img,
        state,
    )


# ------------------- CHANNEL SLIDER UPDATE -------------------

def update_channel(state, ch_idx):
    """
    When slider moves, update the channel heatmap (late layer).
    """
    if not state or "late_fm" not in state:
        return gr.update(value=None)

    late_fm = state["late_fm"]  # (C,H,W)
    W = state["W"]
    H = state["H"]

    C = late_fm.shape[0]
    idx = int(ch_idx)
    if idx < 0 or idx >= C:
        idx = 0

    ch_map = late_fm[idx]
    img = single_channel_heatmap(ch_map, (W, H))
    return gr.update(value=img)


# ------------------- GRADIO UI -------------------

with gr.Blocks(title="YOLOv8n Visualizer — Inside Object Detection (Advanced)") as demo:

    gr.Markdown("# 🧠 YOLOv8n Visualizer — Inside Object Detection (Advanced)")
    gr.Markdown(
        "Explore what happens **inside** an object detection model.\n\n"
        "**Steps shown:**\n"
        "- **Step 0** — Input image\n"
        "- **Step 1** — Early layer activation (edges & textures)\n"
        "- **Step 2** — Middle layer activation (parts & shapes)\n"
        "- **Step 3** — Late layer activation (objects)\n"
        "- **Step 4** — Final detections (boxes & labels)\n"
        "- **Activation overlay** — CAM-style heatmap over the image\n"
        "- **Channel explorer** — inspect individual channels in the late layer\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            in_img = gr.Image(
                label="Step 0 — Input image",
                type="pil"
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=0.25,
                label="Confidence threshold"
            )
            iou_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=0.45,
                label="IoU threshold (NMS)"
            )
            simple_ck = gr.Checkbox(
                label="Explain in simple terms (kids/elders)",
                value=True
            )
            run_btn = gr.Button("Run YOLO & Visualize", variant="primary")

        with gr.Column(scale=1):
            out_det = gr.Image(
                label="Step 4 — Final detections (YOLOv8n)",
                interactive=False
            )
            cam_img = gr.Image(
                label="Activation overlay (late layer focus)",
                interactive=False
            )
            explanation_md = gr.Markdown(label="Explanation")

    gr.Markdown("### 🔍 Steps 1–3: internal feature maps (what the network focuses on)")

    with gr.Row():
        fm1 = gr.Image(
            label="Step 1 — Early layer activation (edges & textures)",
            interactive=False
        )
        fm2 = gr.Image(
            label="Step 2 — Middle layer activation (parts & shapes)",
            interactive=False
        )
        fm3 = gr.Image(
            label="Step 3 — Late layer activation (objects)",
            interactive=False
        )

    gr.Markdown("### 🔬 Channel explorer (late layer)")

    channel_info_md = gr.Markdown()
    channel_slider = gr.Slider(
        minimum=0,
        maximum=0,
        step=1,
        value=0,
        label="Channel index (late layer)"
    )
    channel_heatmap = gr.Image(
        label="Selected channel heatmap (grayscale)",
        interactive=False
    )

    state = gr.State()

    run_btn.click(
        analyze_yolo,
        inputs=[in_img, conf_slider, iou_slider, simple_ck],
        outputs=[
            out_det,
            fm1,
            fm2,
            fm3,
            cam_img,
            explanation_md,
            channel_info_md,
            channel_slider,
            channel_heatmap,
            state,
        ],
    )

    channel_slider.change(
        update_channel,
        inputs=[state, channel_slider],
        outputs=[channel_heatmap],
    )

demo.launch()