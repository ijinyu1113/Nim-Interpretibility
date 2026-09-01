"""Side-by-side stitch of the eval-curve PNG and the logit-lens evolution heatmap.

Left:   new_result/plots/maxrem5_eval_curve_clean.png
Right:  new_result/plots/logit_lens_evolution_subset2.png
Output: new_result/plots/eval_and_logit_lens_2panel.png

Both images are resized; the right panel sets the canvas height. The left panel
is scaled by LEFT_SCALE relative to the right and vertically aligned so that
its *visual content* (non-white bounding box) center matches the right's.
"""
import os
import numpy as np
from PIL import Image

LEFT = "new_result/plots/maxrem5_eval_curve_clean.png"
RIGHT = "new_result/plots/logit_lens_evolution_subset2.png"
OUT = "new_result/plots/eval_and_logit_lens_2panel.png"
PAD_PX = 120         # horizontal white space between panels
MAX_OUT_W = 2400     # cap final width
LEFT_SCALE = 0.65    # shrink left panel relative to right (1.0 = same height)
LEFT_VOFFSET_PX = 80 # nudge left panel down (positive) after content-center align
BG = (255, 255, 255)


def to_rgb(img):
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, BG)
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def scale_to_height(im, h):
    if im.height == h:
        return im
    w = round(im.width * (h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def content_y_center(im, white_threshold=250):
    """Return the y-coordinate of the vertical center of non-white pixels."""
    arr = np.asarray(im.convert("RGB"))
    # A pixel is "content" if any channel is darker than the threshold
    nonwhite = (arr < white_threshold).any(axis=2)
    rows_with_content = np.where(nonwhite.any(axis=1))[0]
    if rows_with_content.size == 0:
        return im.height // 2
    return int((rows_with_content.min() + rows_with_content.max()) // 2)


def main():
    left = to_rgb(Image.open(LEFT))
    right = to_rgb(Image.open(RIGHT))

    # Right panel sets the canvas height. Left panel is scaled down by LEFT_SCALE
    # and aligned so that the visual content (non-white bbox) centers match.
    target_h = max(left.height, right.height)
    right = scale_to_height(right, target_h)
    left = scale_to_height(left, max(1, round(target_h * LEFT_SCALE)))

    right_center_y = content_y_center(right)
    left_center_y_local = content_y_center(left)
    left_y = right_center_y - left_center_y_local + LEFT_VOFFSET_PX
    # clamp so the left image stays inside the canvas
    left_y = max(0, min(left_y, target_h - left.height))

    canvas_w = left.width + PAD_PX + right.width
    canvas = Image.new("RGB", (canvas_w, target_h), BG)
    canvas.paste(left, (0, left_y))
    canvas.paste(right, (left.width + PAD_PX, 0))

    if canvas.width > MAX_OUT_W:
        new_h = round(canvas.height * (MAX_OUT_W / canvas.width))
        canvas = canvas.resize((MAX_OUT_W, new_h), Image.LANCZOS)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT, dpi=(200, 200))
    print(f"Saved: {OUT}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
