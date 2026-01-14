import numpy as np
from torch.utils import data
from itertools import product
from collections.abc import Sequence

def create_drifting_grating_frames(
    width=256,
    height=256,
    n_frames=60,
    spatial_freq=4.0,       # cycles across the width/height
    orientation_deg=0.0,    # 0 = horizontal stripes, 90 = vertical, etc.
    temporal_freq=2.0,      # how many cycles per second (if you have a known frame rate)
    frame_rate=30.0,        # frames per second
    contrast=1.0,           # peak-to-peak contrast [0..1]
    phase_offset=0.0        # initial phase (radians)
):
    """
    Generate a stack of drifting grating frames as a NumPy array of shape (n_frames, height, width).

    :param width: width of each frame in pixels
    :param height: height of each frame in pixels
    :param n_frames: total number of frames
    :param spatial_freq: number of grating cycles that span the full width (and height if square)
    :param orientation_deg: orientation of the grating in degrees
    :param temporal_freq: cycles of drift per second (requires frame_rate to convert to phase increment)
    :param frame_rate: frames per second used to compute phase change per frame
    :param contrast: contrast (0 to 1). 1.0 => sine wave from -1 to 1, which can be rescaled to 0..1
    :param phase_offset: initial phase in radians
    :return: A NumPy array of shape (n_frames, height, width) with values in the range [-1..1]
    """

    # Convert orientation to radians
    theta = np.deg2rad(orientation_deg)

    # Create a 2D coordinate system (centered or 0-based, either is fine)
    # Here we center around (0, 0) for symmetrical sine patterns:
    x_vals = np.linspace(-1, 1, width)
    y_vals = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Rotate coordinates by theta to set the orientation of the stripes
    # x' = x*cos(theta) + y*sin(theta)
    # y' = -x*sin(theta) + y*cos(theta)  (not necessarily needed here, we mainly drift in x')
    x_prime = xx * np.cos(theta) + yy * np.sin(theta)

    # The total number of cycles we want across the image is `spatial_freq`
    # => one cycle means the sine pattern goes from 0->2π
    # We can define 2π * spatial_freq as the "base frequency" in x'.
    base_frequency = 2.0 * np.pi * spatial_freq * 0.5  # factor 0.5 if we consider "1->-1" as 2 units wide
    # Explanation: Our -1..1 coordinate system is 2 units in extent.
    # If you want `spatial_freq` cycles within that 2-unit range, multiply by 0.5.

    # Phase increment per frame, based on temporal frequency
    # temporal_freq (cycles per second) => cycles per frame = temporal_freq / frame_rate
    # => phase increment = 2π * (cycles per frame)
    phase_inc = 2.0 * np.pi * (temporal_freq / frame_rate)

    # Prepare the output array
    frames = np.zeros((n_frames, height, width), dtype=np.float32)

    # Generate each frame
    for i in range(n_frames):
        # Current phase = initial + i * phase_inc
        current_phase = phase_offset + i * phase_inc

        # Sine wave: sin( frequency * x_prime + current_phase )
        # Range is [-1..1]
        grating = np.sin(base_frequency * x_prime + current_phase)

        # Apply contrast scaling: if contrast=1, range is [-1..1],
        # smaller contrasts reduce amplitude.
        grating *= contrast

        # Store it in the 3D array
        frames[i, :, :] = grating

    return frames

def convert_bw_to_rgb(bw_movie: np.ndarray) -> np.ndarray:
    """
    Convert a black-and-white movie stored as a NumPy array of shape
    (n_frames, height, width) into a movie of shape (n_frames, 3, height, width),
    preserving the black-and-white appearance.

    Parameters
    ----------
    bw_movie : np.ndarray
        Input NumPy array with shape (n_frames, height, width), representing a
        black-and-white movie.

    Returns
    -------
    np.ndarray
        Output NumPy array with shape (n_frames, 3, height, width), where each
        original frame has been replicated across the 3 RGB channels.
    """
    return np.repeat(bw_movie[:, np.newaxis, :, :], repeats=3, axis=1)

def color_to_rgb_vector(color):
    """
    Convert a color specification into a length-3 RGB vector.

    Parameters
    ----------
    color : str | sequence[float] | None
        - If None or one of {"gray", "grey", "bw", "none"} -> [1.0, 1.0, 1.0]
          (i.e., no chromatic bias; grayscale).
        - If str, must be one of the named colors below.
        - If sequence, must be length-3 and will be converted to float32.

    Returns
    -------
    np.ndarray
        Shape (3,), dtype float32.
    """
    if color is None:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Handle common "no-color" labels as grayscale/white
    if isinstance(color, str):
        c = color.strip().lower()
        if c in {"gray", "grey", "bw", "none"}:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

        named_colors = {
            "red":      (1.0, 0.0, 0.0),
            "green":    (0.0, 1.0, 0.0),
            "blue":     (0.0, 0.0, 1.0),
            "yellow":   (1.0, 1.0, 0.0),
            "magenta":  (1.0, 0.0, 1.0),
            "cyan":     (0.0, 1.0, 1.0),
            "white":    (1.0, 1.0, 1.0),
            "black":    (0.0, 0.0, 0.0),
            "orange":   (1.0, 0.5, 0.0),
        }

        if c not in named_colors:
            raise ValueError(
                f"Unknown color name '{color}'. "
                f"Supported names: {sorted(named_colors.keys())} "
                f"or provide an explicit RGB triplet."
            )

        return np.array(named_colors[c], dtype=np.float32)

    # Sequence case: explicit RGB values
    arr = np.asarray(color, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(
            f"Color as a sequence must have shape (3,), got shape {arr.shape}."
        )
    return arr

def apply_color_to_grating(bw_movie: np.ndarray, color) -> np.ndarray:
    """
    Take a grayscale drifting grating movie (n_frames, H, W) and apply
    a color vector, returning an RGB movie (n_frames, 3, H, W).

    The grating values remain in [-1, 1]; the color vector determines
    the direction in RGB space (e.g. pure red [1,0,0]).
    """
    rgb_vec = color_to_rgb_vector(color)  # (3,)
    # (n_frames, H, W) -> (n_frames, 1, H, W) * (1, 3, 1, 1) = (n_frames, 3, H, W)
    return bw_movie[:, np.newaxis, :, :] * rgb_vec[np.newaxis, :, np.newaxis, np.newaxis]


class DatasetDriftingGrating(data.Dataset):
    def __init__(
        self,
        parameter_list: str | list[dict] = "DEFAULT",
        y_par=["spatial_freq", "orientation_deg", "temporal_freq", "contrast", "color"],
        color_dim=3,
        color_as_rgb: bool = True,
        transform=None
    ):
        """
        Dataset of drifting gratings.

        Parameters
        ----------
        parameter_list : {"DEFAULT", "DEFAULT_TWO_REGION"} or list[dict]
            - "DEFAULT": original single-grating stimuli across the full frame.
            - "DEFAULT_TWO_REGION": two independent gratings in the left and
              right halves of the frame, sharing global parameters like
              spatial_freq/temporal_freq but differing in orientation.
              Each dict will also contain an "orientation_deg_left",
              "orientation_deg_right", and "orientation_delta" key (along with
              the shared keys like "spatial_freq", "temporal_freq", etc.).
            - list[dict]: user-specified parameter dicts (single-grating mode).
        y_par : list[str] | None
            Names of parameters to be returned as the target vector y.
            If None, all keys of the parameter dict are used.
        color_dim : int
            1 -> return grayscale movie, shape (n_frames, H, W)
            3 -> return RGB movie, shape (n_frames, 3, H, W).
                 If a 'color' is specified in the parameter dict and is not
                 gray/bw/none, the stripes will be colored accordingly.
        color_as_rgb : bool
            Whether to encode "color" in y as RGB (3 floats) or as a single
            categorical index (int).
        """
        self.color_dim = color_dim
        self.multi_region = False  # flag for two-region stimuli
        self.color_as_rgb = color_as_rgb  # <-- store toggle
        self._color_to_index: dict = {}   # <-- mapping color spec -> int index
        self.transform = transform

        if isinstance(parameter_list, str):
            if parameter_list == "DEFAULT":
                width = [128]
                height = [128]
                n_frames = [16]
                spatial_freq = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
                orientation_deg = [
                    0.0,
                    30.0,
                    60.0,
                    90.0,
                    120.0,
                    150.0,
                    180.0,
                    210.0,
                    240.0,
                    270.0,
                    300.0,
                    330.0,
                ]
                temporal_freq = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
                frame_rate = [6.25]
                contrast = [0.2, 0.4, 0.6, 0.8, 1.0]
                phase_offset = [0.0]

                # color is a global parameter for the whole frame
                color = ["gray", "red", "green", "blue", "orange"]

                tuple_pars = product(
                    width,
                    height,
                    n_frames,
                    spatial_freq,
                    orientation_deg,
                    temporal_freq,
                    frame_rate,
                    contrast,
                    phase_offset,
                    color,
                )
                self.pars = [
                    {
                        "width": x[0],
                        "height": x[1],
                        "n_frames": x[2],
                        "spatial_freq": x[3],
                        "orientation_deg": x[4],
                        "temporal_freq": x[5],
                        "frame_rate": x[6],
                        "contrast": x[7],
                        "phase_offset": x[8],
                        "color": x[9],
                    }
                    for x in tuple_pars
                ]
                self.multi_region = False

            elif parameter_list == "DEFAULT_TWO_REGION":
                """
                Two-region (left/right) stimuli.

                - Left and right halves share:
                    width, height, n_frames, spatial_freq, temporal_freq,
                    frame_rate, contrast, phase_offset, color.
                - They differ in:
                    orientation_deg_left, orientation_deg_right.
                - We also store orientation_delta = (right - left) modulo 360
                  so it can be used as an additional low-level label.
                """
                width = [128]
                height = [128]
                n_frames = [16]
                spatial_freq_left = [4.0, 10.0, 16.0]
                spatial_freq_right = [4.0, 10.0, 16.0]
                orientation_deg_left = [
                    0.0,
                    30.0,
                    60.0,
                    90.0,
                    120.0,
                    150.0,
                    180.0,
                    210.0,
                    240.0,
                    270.0,
                    300.0,
                    330.0,
                ]
                orientation_delta = [0.0, 45.0, 90.0, 135.0]
                temporal_freq = [2.0, 5.0, 8.0]
                frame_rate = [6.25]
                contrast = [0.33, 0.67, 1.0]
                phase_offset = [0.0]
                color = ["gray", "red", "green", "blue"]

                tuple_pars = product(
                    width,
                    height,
                    n_frames,
                    spatial_freq_left,
                    spatial_freq_right,
                    orientation_deg_left,
                    orientation_delta,
                    temporal_freq,
                    frame_rate,
                    contrast,
                    phase_offset,
                    color,
                )

                pars = []
                for (w, h, nf, sf_left, sf_right, theta_left, dtheta,
                     tf, fr, c, phase, col) in tuple_pars:
                    theta_right = (theta_left + dtheta) % 360.0
                    sf_delta = sf_right - sf_left

                    if sf_left > sf_right:
                        sf_higher_side = 0
                    elif sf_right > sf_left:
                        sf_higher_side = 1
                    else:
                        sf_higher_side = 2

                    spatial_freq_match = float(np.isclose(sf_left, sf_right))
                    orientation_match = float(np.isclose(dtheta % 360.0, 0.0))
                    pars.append(
                        {
                            "width": w,
                            "height": h,
                            "n_frames": nf,
                            "spatial_freq_left": sf_left,
                            "spatial_freq_right": sf_right,
                            "sf_delta": sf_delta,
                            "sf_higher_side": sf_higher_side,
                            "spatial_freq_match": spatial_freq_match,
                            "orientation_deg_left": theta_left,
                            "orientation_deg_right": theta_right,
                            "orientation_delta": dtheta,
                            "orientation_match": orientation_match,
                            "temporal_freq": tf,
                            "frame_rate": fr,
                            "contrast": c,
                            "phase_offset": phase,
                            "color": col,
                        }
                    )
                self.pars = pars
                self.multi_region = True

            else:
                raise ValueError(
                    'If parameter_list is a string, it must be "DEFAULT" or "DEFAULT_TWO_REGION".'
                )
        elif isinstance(parameter_list, list):
            # For custom lists we assume single-region stimuli by default.
            self.pars = []
            for p in parameter_list:
                if not isinstance(p, dict):
                    raise ValueError(
                        "Each element of parameter_list must be a dict."
                    )
                if "color" not in p:
                    # Default to None -> treated as grayscale
                    q = dict(p)
                    q["color"] = None
                    self.pars.append(q)
                else:
                    self.pars.append(p)
            self.multi_region = False
        else:
            raise ValueError(
                "parameter_list should be either of type list or of type string."
            )

        self.y_par = self.pars[0].keys() if y_par is None else y_par

    def _normalize_color_key(self, color):
        """
        Produce a hashable, normalized representation of the color so that
        all equivalent specs map to the same key (for categorical indexing).
        """
        if color is None:
            return "none"
        if isinstance(color, str):
            return color.strip().lower()
        # Sequence / array case (e.g. [1.0, 0.0, 0.0])
        arr = np.asarray(color, dtype=np.float32)
        return tuple(arr.tolist())

    def _get_color_index(self, color):
        """
        Map a color specification to a stable integer index.
        """
        key = self._normalize_color_key(color)
        if key not in self._color_to_index:
            self._color_to_index[key] = len(self._color_to_index)
        return self._color_to_index[key]

    def __len__(self):
        return len(self.pars)

    def __getitem__(self, idx):
        pars = self.pars[idx]

        if not self.multi_region:
            # ----- Single-region (original) behavior -----
            color = pars.get("color", None)
            grating_pars = {k: v for k, v in pars.items() if k != "color"}

            # Generate the base grayscale movie: (n_frames, H, W)
            bw_movie = create_drifting_grating_frames(**grating_pars)

        else:
            # ----- Two-region (left/right) behavior -----
            color = pars.get("color", None)

            # Shared parameters for both halves
            shared_pars = {
                "width": pars["width"],
                "height": pars["height"],
                "n_frames": pars["n_frames"],
                "temporal_freq": pars["temporal_freq"],
                "frame_rate": pars["frame_rate"],
                "contrast": pars["contrast"],
                "phase_offset": pars["phase_offset"],
            }

            # Left and right orientations
            left_pars = dict(shared_pars)
            left_pars["spatial_freq"] = pars["spatial_freq_left"]
            left_pars["orientation_deg"] = pars["orientation_deg_left"]

            right_pars = dict(shared_pars)
            right_pars["spatial_freq"] = pars["spatial_freq_right"]
            right_pars["orientation_deg"] = pars["orientation_deg_right"]

            # Generate two grayscale movies
            left_movie = create_drifting_grating_frames(**left_pars)
            right_movie = create_drifting_grating_frames(**right_pars)

            if left_movie.shape != right_movie.shape:
                raise RuntimeError(
                    f"Left and right movies must have the same shape, "
                    f"got {left_movie.shape} vs {right_movie.shape}."
                )

            # Combine them into a single movie, splitting the frame vertically:
            # left half from left_movie, right half from right_movie.
            n_frames, H, W = left_movie.shape
            mid_x = W // 2

            bw_movie = np.empty_like(left_movie)
            # Left half [0 : mid_x)
            bw_movie[:, :, :mid_x] = left_movie[:, :, :mid_x]
            # Right half [mid_x : W)
            bw_movie[:, :, mid_x:] = right_movie[:, :, mid_x:]

        # ----- Color / channel handling -----
        if self.color_dim == 1:
            X = bw_movie  # shape (n_frames, H, W)
        elif self.color_dim == 3:
            # Decide whether to colorize or just replicate to RGB.
            # "gray" / "bw" / "none" / None -> grayscale RGB
            if isinstance(color, str) and color.strip().lower() in {
                "gray",
                "grey",
                "bw",
                "none",
            }:
                X = convert_bw_to_rgb(bw_movie)
            elif color is None:
                X = convert_bw_to_rgb(bw_movie)
            else:
                X = apply_color_to_grating(bw_movie, color)
        else:
            raise ValueError("Unknown color_dim; expected 1 or 3.")

        # ----- y construction with RGB color expansion -----
        targets = []
        for name in self.y_par:
            if name == "color":
                if self.color_as_rgb:
                    # Expand to 3 entries [R, G, B]
                    rgb = color_to_rgb_vector(color)
                    targets.extend(rgb.tolist())
                else:
                    color_idx = self._get_color_index(color)
                    targets.append(color_idx)
            else:
                targets.append(pars[name])

        # X: (n_frames, C, H, W) -> (C, n_frames, H, W)
        X = np.transpose(X, (1, 0, 2, 3))
        if self.transform is not None:
            X = self.transform(X)

        y = np.array(targets, dtype=np.float32)

        return X, y
