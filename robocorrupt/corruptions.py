"""Parameterized visual corruptions for the benchmark.

Each corruption is a function ``fn(img, severity, rng)`` mapping a uint8
HWC image to a uint8 HWC image. ``severity`` in [0, 1]; the benchmark uses
levels [0.33, 0.66, 1.0]. All randomness comes from the passed rng, so
corruptions are reproducible given a seed.
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _pil(img):
    return Image.fromarray(img)


def brightness(img, s, rng):
    f = 1.0 + 0.9 * s * (1 if rng.random() < 0.5 else -1)
    out = ImageEnhance.Brightness(_pil(img)).enhance(max(f, 0.05))
    return np.asarray(out)


def contrast(img, s, rng):
    f = 1.0 + 1.2 * s * (1 if rng.random() < 0.5 else -1)
    out = ImageEnhance.Contrast(_pil(img)).enhance(max(f, 0.05))
    return np.asarray(out)


def gaussian_noise(img, s, rng):
    noise = rng.normal(0, 28 * s, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def blur(img, s, rng):
    return np.asarray(_pil(img).filter(
        ImageFilter.GaussianBlur(radius=0.4 + 2.6 * s)))


def occlusion(img, s, rng):
    out = img.copy()
    h, w, _ = out.shape
    n = 1 + int(3 * s)
    for _ in range(n):
        ph = int(h * (0.10 + 0.25 * s * rng.random()))
        pw = int(w * (0.10 + 0.25 * s * rng.random()))
        y = rng.integers(0, max(h - ph, 1))
        x = rng.integers(0, max(w - pw, 1))
        out[y:y + ph, x:x + pw] = rng.integers(0, 60, size=3, dtype=np.uint8)
    return out


def hue_shift(img, s, rng):
    hsv = _pil(img).convert("HSV")
    arr = np.asarray(hsv).astype(np.int32)
    shift = int(90 * s * (1 if rng.random() < 0.5 else -1))
    arr[:, :, 0] = (arr[:, :, 0] + shift) % 256
    return np.asarray(Image.fromarray(
        arr.astype(np.uint8), mode="HSV").convert("RGB"))


def camera_jitter(img, s, rng):
    h, w, _ = img.shape
    dy = int(rng.integers(-7, 8) * s)
    dx = int(rng.integers(-7, 8) * s)
    out = np.roll(img, shift=(dy, dx), axis=(0, 1))
    return out


def pixelate(img, s, rng):
    h, w, _ = img.shape
    k = 1 + int(5 * s)
    small = _pil(img).resize((max(w // k, 4), max(h // k, 4)),
                             Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.NEAREST))


def fog(img, s, rng):
    veil = np.full_like(img, 205, dtype=np.float32)
    a = 0.55 * s
    return np.clip((1 - a) * img.astype(np.float32) + a * veil,
                   0, 255).astype(np.uint8)


def speckle(img, s, rng):
    out = img.copy()
    n = int(img.size / 3 * 0.06 * s)
    ys = rng.integers(0, img.shape[0], size=n)
    xs = rng.integers(0, img.shape[1], size=n)
    vals = rng.choice(np.array([0, 255], dtype=np.uint8), size=n)
    out[ys, xs] = vals[:, None]
    return out


CORRUPTIONS = {
    "brightness": brightness,
    "contrast": contrast,
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "occlusion": occlusion,
    "hue_shift": hue_shift,
    "camera_jitter": camera_jitter,
    "pixelate": pixelate,
    "fog": fog,
    "speckle": speckle,
}

SEVERITY_LEVELS = [0.33, 0.66, 1.0]


def apply(img, name, severity, rng):
    return CORRUPTIONS[name](img, severity, rng)
