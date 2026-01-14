import numpy as np
import torch

class KeepOneRandomPatchOnly:
    """
    Keeps exactly one random (patch_size x patch_size) spatial patch from a video
    and fills everything else with a constant value.

    Expects input shaped either:
      - (C, T, H, W)  [default]
      - (T, C, H, W)

    Works with numpy arrays or torch tensors.
    """
    def __init__(
        self,
        patch_size: int = 16,
        fill: str | int | float = "black",   # "black", "white", or numeric constant
        layout: str = "CTHW",                # "CTHW" or "TCHW"
        same_patch_for_all_frames: bool = True,
        require_divisible: bool = True,
    ):
        self.patch_size = int(patch_size)
        self.fill = fill
        self.layout = layout.upper()
        self.same_patch_for_all_frames = bool(same_patch_for_all_frames)
        self.require_divisible = bool(require_divisible)

        if self.layout not in ("CTHW", "TCHW"):
            raise ValueError(f"layout must be 'CTHW' or 'TCHW', got {layout}")

    def _resolve_fill_value(self, dtype):
        # dtype: numpy dtype or torch dtype
        if isinstance(self.fill, str):
            mode = self.fill.lower()
            if mode not in ("black", "white"):
                raise ValueError("fill must be 'black', 'white', or a numeric constant")

            if mode == "black":
                return 0

            # white
            if isinstance(dtype, torch.dtype):
                if dtype.is_floating_point:
                    return 1.0
                return torch.iinfo(dtype).max
            else:
                if np.issubdtype(dtype, np.floating):
                    return 1.0
                return np.iinfo(dtype).max

        # numeric constant
        return self.fill

    def __call__(self, x):
        is_torch = torch.is_tensor(x)
        is_numpy = isinstance(x, np.ndarray)
        if not (is_torch or is_numpy):
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

        ps = self.patch_size

        if self.layout == "CTHW":
            # (C, T, H, W)
            C, T, H, W = x.shape
            h_dim, w_dim = 2, 3
        else:
            # (T, C, H, W)
            T, C, H, W = x.shape
            h_dim, w_dim = 2, 3

        if H < ps or W < ps:
            raise ValueError(f"H and W must be >= patch_size ({ps}), got H={H}, W={W}")

        if self.require_divisible and (H % ps != 0 or W % ps != 0):
            raise ValueError(f"H and W must be divisible by {ps}. Got H={H}, W={W}")

        nph = H // ps
        npw = W // ps

        fill_val = self._resolve_fill_value(x.dtype if is_torch else x.dtype)

        if is_torch:
            out = torch.full_like(x, fill_val)
            if self.same_patch_for_all_frames:
                pi = int(torch.randint(nph, (1,), device=x.device).item())
                pj = int(torch.randint(npw, (1,), device=x.device).item())
                y0, x0 = pi * ps, pj * ps
                if self.layout == "CTHW":
                    out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
                else:
                    out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
            else:
                # different patch per frame
                for t in range(T):
                    pi = int(torch.randint(nph, (1,), device=x.device).item())
                    pj = int(torch.randint(npw, (1,), device=x.device).item())
                    y0, x0 = pi * ps, pj * ps
                    if self.layout == "CTHW":
                        out[:, t, y0:y0+ps, x0:x0+ps] = x[:, t, y0:y0+ps, x0:x0+ps]
                    else:
                        out[t, :, y0:y0+ps, x0:x0+ps] = x[t, :, y0:y0+ps, x0:x0+ps]
            return out

        # numpy
        out = np.full_like(x, fill_val)
        if self.same_patch_for_all_frames:
            # use torch RNG so DataLoader worker seeding stays consistent with torch
            pi = int(torch.randint(nph, (1,)).item())
            pj = int(torch.randint(npw, (1,)).item())
            y0, x0 = pi * ps, pj * ps
            if self.layout == "CTHW":
                out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
            else:
                out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
        else:
            for t in range(T):
                pi = int(torch.randint(nph, (1,)).item())
                pj = int(torch.randint(npw, (1,)).item())
                y0, x0 = pi * ps, pj * ps
                if self.layout == "CTHW":
                    out[:, t, y0:y0+ps, x0:x0+ps] = x[:, t, y0:y0+ps, x0:x0+ps]
                else:
                    out[t, :, y0:y0+ps, x0:x0+ps] = x[t, :, y0:y0+ps, x0:x0+ps]
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(patch_size={self.patch_size}, fill={self.fill}, "
                f"layout={self.layout}, same_patch_for_all_frames={self.same_patch_for_all_frames}, "
                f"require_divisible={self.require_divisible})")


class KeepTwoRandomPatchesSplitMiddle:
    """
    Keeps exactly TWO random (patch_size x patch_size) spatial patches from a video:
      - one patch fully inside the LEFT half  [0 : mid)
      - one patch fully inside the RIGHT half [mid : W)
    where mid = W // 2 (exact middle split).
    Fills everything else with a constant value.

    Expects input shaped either:
      - (C, T, H, W)  [default]
      - (T, C, H, W)

    Works with numpy arrays or torch tensors.
    """
    def __init__(
        self,
        patch_size: int = 16,
        fill: str | int | float = "black",   # "black", "white", or numeric constant
        layout: str = "CTHW",                # "CTHW" or "TCHW"
        same_patch_for_all_frames: bool = True,
        require_divisible: bool = True,
    ):
        self.patch_size = int(patch_size)
        self.fill = fill
        self.layout = layout.upper()
        self.same_patch_for_all_frames = bool(same_patch_for_all_frames)
        self.require_divisible = bool(require_divisible)

        if self.layout not in ("CTHW", "TCHW"):
            raise ValueError(f"layout must be 'CTHW' or 'TCHW', got {layout}")

    def _resolve_fill_value(self, dtype):
        if isinstance(self.fill, str):
            mode = self.fill.lower()
            if mode not in ("black", "white"):
                raise ValueError("fill must be 'black', 'white', or a numeric constant")

            if mode == "black":
                return 0

            # white
            if isinstance(dtype, torch.dtype):
                if dtype.is_floating_point:
                    return 1.0
                return torch.iinfo(dtype).max
            else:
                if np.issubdtype(dtype, np.floating):
                    return 1.0
                return np.iinfo(dtype).max

        return self.fill

    def __call__(self, x):
        is_torch = torch.is_tensor(x)
        is_numpy = isinstance(x, np.ndarray)
        if not (is_torch or is_numpy):
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

        ps = self.patch_size

        if self.layout == "CTHW":
            C, T, H, W = x.shape
        else:
            T, C, H, W = x.shape

        if H < ps or W < ps:
            raise ValueError(f"H and W must be >= patch_size ({ps}), got H={H}, W={W}")

        if self.require_divisible and (H % ps != 0 or W % ps != 0):
            raise ValueError(f"H and W must be divisible by {ps}. Got H={H}, W={W}")

        nph = H // ps

        mid = W // 2  # exact middle split

        # Choose patch x-positions so patches are fully inside each half:
        # left:  x0 + ps <= mid
        # right: x0 >= mid  AND x0 + ps <= W
        left_max_pj = (mid - ps) // ps
        if left_max_pj < 0:
            raise ValueError(
                f"Left half too narrow for a patch: mid={mid}, patch_size={ps}. "
                f"Need mid >= patch_size."
            )

        right_min_pj = (mid + ps - 1) // ps  # ceil(mid / ps)
        right_max_pj = (W - ps) // ps
        if right_min_pj > right_max_pj:
            raise ValueError(
                f"Right half too narrow for a patch: mid={mid}, W={W}, patch_size={ps}. "
                f"Need (W - mid) >= patch_size and at least one aligned position."
            )

        fill_val = self._resolve_fill_value(x.dtype if is_torch else x.dtype)

        out = torch.full_like(x, fill_val) if is_torch else np.full_like(x, fill_val)

        def _randint(low_incl, high_incl, device=None):
            # inclusive bounds
            if is_torch:
                return int(torch.randint(low_incl, high_incl + 1, (1,), device=device).item())
            # numpy path: use torch RNG for worker seeding consistency
            return int(torch.randint(low_incl, high_incl + 1, (1,)).item())

        def _copy_patch_all_frames(y0, x0):
            if self.layout == "CTHW":
                out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
            else:  # TCHW
                out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]

        def _copy_patch_frame(t, y0, x0):
            if self.layout == "CTHW":
                out[:, t, y0:y0+ps, x0:x0+ps] = x[:, t, y0:y0+ps, x0:x0+ps]
            else:  # TCHW
                out[t, :, y0:y0+ps, x0:x0+ps] = x[t, :, y0:y0+ps, x0:x0+ps]

        dev = x.device if is_torch else None

        if self.same_patch_for_all_frames:
            # LEFT patch
            pi_l = _randint(0, nph - 1, dev)
            pj_l = _randint(0, left_max_pj, dev)
            y0_l, x0_l = pi_l * ps, pj_l * ps

            # RIGHT patch
            pi_r = _randint(0, nph - 1, dev)
            pj_r = _randint(right_min_pj, right_max_pj, dev)
            y0_r, x0_r = pi_r * ps, pj_r * ps

            _copy_patch_all_frames(y0_l, x0_l)
            _copy_patch_all_frames(y0_r, x0_r)
        else:
            # different patches per frame
            for t in range(T):
                # LEFT patch
                pi_l = _randint(0, nph - 1, dev)
                pj_l = _randint(0, left_max_pj, dev)
                y0_l, x0_l = pi_l * ps, pj_l * ps

                # RIGHT patch
                pi_r = _randint(0, nph - 1, dev)
                pj_r = _randint(right_min_pj, right_max_pj, dev)
                y0_r, x0_r = pi_r * ps, pj_r * ps

                _copy_patch_frame(t, y0_l, x0_l)
                _copy_patch_frame(t, y0_r, x0_r)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(patch_size={self.patch_size}, fill={self.fill}, "
            f"layout={self.layout}, same_patch_for_all_frames={self.same_patch_for_all_frames}, "
            f"require_divisible={self.require_divisible})"
        )


class KeepOneRandomPatchOnlyUCF:
    """
    Keeps exactly one random (patch_size x patch_size) spatial patch from a video
    and fills everything else with a constant value.

    Expects input shaped either:
      - (C, T, H, W) or (B, C, T, H, W)   [layout="CTHW"]
      - (T, C, H, W) or (B, T, C, H, W)   [layout="TCHW"]

    Works with numpy arrays or torch tensors.
    """
    def __init__(
        self,
        patch_size: int = 16,
        fill: str | int | float = "black",   # "black", "white", or numeric constant
        layout: str = "CTHW",                # "CTHW" or "TCHW"
        same_patch_for_all_frames: bool = True,
        require_divisible: bool = True,
    ):
        self.patch_size = int(patch_size)
        self.fill = fill
        self.layout = layout.upper()
        self.same_patch_for_all_frames = bool(same_patch_for_all_frames)
        self.require_divisible = bool(require_divisible)

        if self.layout not in ("CTHW", "TCHW"):
            raise ValueError(f"layout must be 'CTHW' or 'TCHW', got {layout}")

    def _resolve_fill_value(self, dtype):
        # dtype: numpy dtype or torch dtype
        if isinstance(self.fill, str):
            mode = self.fill.lower()
            if mode not in ("black", "white"):
                raise ValueError("fill must be 'black', 'white', or a numeric constant")

            if mode == "black":
                return 0

            # white
            if isinstance(dtype, torch.dtype):
                if dtype.is_floating_point:
                    return 1.0
                return torch.iinfo(dtype).max
            else:
                if np.issubdtype(dtype, np.floating):
                    return 1.0
                return np.iinfo(dtype).max

        # numeric constant
        return self.fill

    def __call__(self, x):
        is_torch = torch.is_tensor(x)
        is_numpy = isinstance(x, np.ndarray)
        if not (is_torch or is_numpy):
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

        ps = self.patch_size

        if x.ndim not in (4, 5):
            raise ValueError(f"Expected 4D or 5D input, got shape {tuple(x.shape)}")

        has_batch = (x.ndim == 5)

        if self.layout == "CTHW":
            # (C, T, H, W) or (B, C, T, H, W)
            if has_batch:
                B, C, T, H, W = x.shape
            else:
                C, T, H, W = x.shape
                B = None
            h_dim, w_dim = (3, 4) if has_batch else (2, 3)
        else:
            # (T, C, H, W) or (B, T, C, H, W)
            if has_batch:
                B, T, C, H, W = x.shape
            else:
                T, C, H, W = x.shape
                B = None
            h_dim, w_dim = (3, 4) if has_batch else (2, 3)

        if H < ps or W < ps:
            raise ValueError(f"H and W must be >= patch_size ({ps}), got H={H}, W={W}")

        if self.require_divisible and (H % ps != 0 or W % ps != 0):
            raise ValueError(f"H and W must be divisible by {ps}. Got H={H}, W={W}")

        nph = H // ps
        npw = W // ps

        fill_val = self._resolve_fill_value(x.dtype if is_torch else x.dtype)

        if is_torch:
            out = torch.full_like(x, fill_val)

            # ====== batched torch path ======
            if has_batch:
                P = ps * ps
                dy = torch.arange(ps, device=x.device, dtype=torch.long)
                dx = torch.arange(ps, device=x.device, dtype=torch.long)
                patch_offsets = (dy[:, None] * W + dx[None, :]).reshape(-1)  # (P,)

                if self.same_patch_for_all_frames:
                    pi = torch.randint(nph, (B,), device=x.device)
                    pj = torch.randint(npw, (B,), device=x.device)
                    base = (pi * ps) * W + (pj * ps)  # (B,)
                    idx = base[:, None] + patch_offsets[None, :]  # (B, P)

                    if self.layout == "CTHW":
                        x_flat = x.reshape(B, C, T, H * W)
                        out_flat = out.reshape(B, C, T, H * W)
                        idx_full = idx[:, None, None, :].expand(B, C, T, P)
                        patch = torch.gather(x_flat, dim=3, index=idx_full)
                        out_flat.scatter_(dim=3, index=idx_full, src=patch)
                    else:
                        x_flat = x.reshape(B, T, C, H * W)
                        out_flat = out.reshape(B, T, C, H * W)
                        idx_full = idx[:, None, None, :].expand(B, T, C, P)
                        patch = torch.gather(x_flat, dim=3, index=idx_full)
                        out_flat.scatter_(dim=3, index=idx_full, src=patch)

                else:
                    # different patch per (batch, frame)
                    pi = torch.randint(nph, (B, T), device=x.device)
                    pj = torch.randint(npw, (B, T), device=x.device)
                    base = (pi * ps) * W + (pj * ps)  # (B, T)
                    idx = base[..., None] + patch_offsets[None, None, :]  # (B, T, P)

                    if self.layout == "CTHW":
                        x_flat = x.reshape(B, C, T, H * W)
                        out_flat = out.reshape(B, C, T, H * W)
                        idx_full = idx[:, None, :, :].expand(B, C, T, P)
                        patch = torch.gather(x_flat, dim=3, index=idx_full)
                        out_flat.scatter_(dim=3, index=idx_full, src=patch)
                    else:
                        x_flat = x.reshape(B, T, C, H * W)
                        out_flat = out.reshape(B, T, C, H * W)
                        idx_full = idx[:, :, None, :].expand(B, T, C, P)
                        patch = torch.gather(x_flat, dim=3, index=idx_full)
                        out_flat.scatter_(dim=3, index=idx_full, src=patch)

                return out

            # ====== original (unbatched) torch path ======
            if self.same_patch_for_all_frames:
                pi = int(torch.randint(nph, (1,), device=x.device).item())
                pj = int(torch.randint(npw, (1,), device=x.device).item())
                y0, x0 = pi * ps, pj * ps
                if self.layout == "CTHW":
                    out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
                else:
                    out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
            else:
                # different patch per frame
                for t in range(T):
                    pi = int(torch.randint(nph, (1,), device=x.device).item())
                    pj = int(torch.randint(npw, (1,), device=x.device).item())
                    y0, x0 = pi * ps, pj * ps
                    if self.layout == "CTHW":
                        out[:, t, y0:y0+ps, x0:x0+ps] = x[:, t, y0:y0+ps, x0:x0+ps]
                    else:
                        out[t, :, y0:y0+ps, x0:x0+ps] = x[t, :, y0:y0+ps, x0:x0+ps]
            return out

        # ====== numpy path ======
        out = np.full_like(x, fill_val)

        # batched numpy path
        if has_batch:
            P = ps * ps
            patch_offsets = (np.arange(ps, dtype=np.int64)[:, None] * np.int64(W) +
                             np.arange(ps, dtype=np.int64)[None, :]).reshape(-1)  # (P,)

            if self.same_patch_for_all_frames:
                # use torch RNG so DataLoader worker seeding stays consistent with torch
                pi = torch.randint(nph, (B,)).cpu().numpy().astype(np.int64, copy=False)
                pj = torch.randint(npw, (B,)).cpu().numpy().astype(np.int64, copy=False)
                base = (pi * ps) * np.int64(W) + (pj * ps)  # (B,)
                idx = base[:, None] + patch_offsets[None, :]  # (B, P)

                if self.layout == "CTHW":
                    x_flat = x.reshape(B, C, T, H * W)
                    out_flat = out.reshape(B, C, T, H * W)
                    idx_full = np.broadcast_to(idx[:, None, None, :], (B, C, T, P))
                    patch = np.take_along_axis(x_flat, idx_full, axis=3)
                    np.put_along_axis(out_flat, idx_full, patch, axis=3)
                else:
                    x_flat = x.reshape(B, T, C, H * W)
                    out_flat = out.reshape(B, T, C, H * W)
                    idx_full = np.broadcast_to(idx[:, None, None, :], (B, T, C, P))
                    patch = np.take_along_axis(x_flat, idx_full, axis=3)
                    np.put_along_axis(out_flat, idx_full, patch, axis=3)

            else:
                pi = torch.randint(nph, (B, T)).cpu().numpy().astype(np.int64, copy=False)
                pj = torch.randint(npw, (B, T)).cpu().numpy().astype(np.int64, copy=False)
                base = (pi * ps) * np.int64(W) + (pj * ps)  # (B, T)
                idx = base[..., None] + patch_offsets[None, None, :]  # (B, T, P)

                if self.layout == "CTHW":
                    x_flat = x.reshape(B, C, T, H * W)
                    out_flat = out.reshape(B, C, T, H * W)
                    idx_full = np.broadcast_to(idx[:, None, :, :], (B, C, T, P))
                    patch = np.take_along_axis(x_flat, idx_full, axis=3)
                    np.put_along_axis(out_flat, idx_full, patch, axis=3)
                else:
                    x_flat = x.reshape(B, T, C, H * W)
                    out_flat = out.reshape(B, T, C, H * W)
                    idx_full = np.broadcast_to(idx[:, :, None, :], (B, T, C, P))
                    patch = np.take_along_axis(x_flat, idx_full, axis=3)
                    np.put_along_axis(out_flat, idx_full, patch, axis=3)

            return out

        # original (unbatched) numpy path
        if self.same_patch_for_all_frames:
            # use torch RNG so DataLoader worker seeding stays consistent with torch
            pi = int(torch.randint(nph, (1,)).item())
            pj = int(torch.randint(npw, (1,)).item())
            y0, x0 = pi * ps, pj * ps
            if self.layout == "CTHW":
                out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
            else:
                out[:, :, y0:y0+ps, x0:x0+ps] = x[:, :, y0:y0+ps, x0:x0+ps]
        else:
            for t in range(T):
                pi = int(torch.randint(nph, (1,)).item())
                pj = int(torch.randint(npw, (1,)).item())
                y0, x0 = pi * ps, pj * ps
                if self.layout == "CTHW":
                    out[:, t, y0:y0+ps, x0:x0+ps] = x[:, t, y0:y0+ps, x0:x0+ps]
                else:
                    out[t, :, y0:y0+ps, x0:x0+ps] = x[t, :, y0:y0+ps, x0:x0+ps]
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(patch_size={self.patch_size}, fill={self.fill}, "
                f"layout={self.layout}, same_patch_for_all_frames={self.same_patch_for_all_frames}, "
                f"require_divisible={self.require_divisible})")
