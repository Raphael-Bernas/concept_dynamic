import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Any
import numpy as np

def tojson_decoder(d):
        if "__ndarray__" in d:
            arr = np.array(d["__ndarray__"], dtype=d["dtype"])
            arr = arr.reshape(d["shape"])
            return arr.T
        return d

def concat_concept_matrices(
    file_paths: Iterable[str],
    step_keys: Iterable[str],
    layer_keys: Iterable[str],
    axis: int = 0,
    allow_missing: bool = False,
    return_numpy: bool = True,
) -> Union[np.ndarray, List]:

    file_paths = list(file_paths)
    if not file_paths:
        raise ValueError("file_paths must be a non-empty iterable of JSON file paths.")

    loaded = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON root in {fp} must be an object/dict.")
        loaded.append((fp, data))

    # Check duplicate step keys across files
    step_to_files = {}
    for fp, data in loaded:
        for step in data.keys():
            step_to_files.setdefault(step, []).append(fp)
    duplicates = {k: v for k, v in step_to_files.items() if len(v) > 1}
    if duplicates:
        lines = []
        for step, files in duplicates.items():
            lines.append(f"{step}: {files}")
        raise ValueError("Duplicate step keys across provided files:\n" + "\n".join(lines))

    # Merge dictionaries into one mapping
    merged = {}
    for _, data in loaded:
        merged.update(data)

    matrices = []
    shapes = []  
    for s in step_keys:
        if s not in merged:
            if allow_missing:
                continue
            raise ValueError(f"Requested step key '{s}' not present in merged JSONs.")
        step_obj = merged[s]
        if not isinstance(step_obj, dict):
            if allow_missing:
                continue
            raise ValueError(f"Value for step '{s}' is not a dict.")
        concepts = step_obj.get("extracted_concepts")
        if concepts is None:
            if allow_missing:
                continue
            raise ValueError(f"Step '{s}' has no 'concepts' entry.")
        if not isinstance(concepts, dict):
            if allow_missing:
                continue
            raise ValueError(f"In step '{s}', 'concepts' is not a dict.")

        for l in layer_keys:
            if l not in concepts:
                if allow_missing:
                    continue
                raise ValueError(f"Requested layer key '{l}' not found in step '{s}'.")
            mat = concepts[l]
            mat = tojson_decoder(mat)
            if isinstance(mat, list):
                arr = np.array(mat)
            elif isinstance(mat, np.ndarray):
                arr = mat
            else:
                try:
                    arr = np.array(mat)
                except Exception as exc:
                    raise ValueError(
                        f"Value at [{s}]['concepts'][{l}] is not an array-like object: {exc}"
                    ) from exc

            if arr.ndim == 0:
                arr = arr.reshape((1,))
            matrices.append(arr)
            shapes.append(arr.shape)

    if len(matrices) == 0:
        raise ValueError("No matrices were collected for the given step_keys/layer_keys (check allow_missing).")

    axis_pos = axis if axis >= 0 else axis + matrices[0].ndim
    if axis_pos < 0 or axis_pos >= matrices[0].ndim:
        raise ValueError(f"Concatenation axis {axis} out of range for gathered arrays with ndim {matrices[0].ndim}.")

    ndims = [m.ndim for m in matrices]
    if not all(nd == ndims[0] for nd in ndims):
        raise ValueError(f"All arrays must have the same number of dimensions. Found ndims: {ndims}")

    ref_shape = list(matrices[0].shape)
    for idx, m in enumerate(matrices[1:], start=1):
        for d in range(m.ndim):
            if d == axis_pos:
                continue
            if m.shape[d] != ref_shape[d]:
                raise ValueError(
                    f"Shape mismatch at matrix index {idx}: shape {m.shape} differs from reference {tuple(ref_shape)} "
                    f"on axis {d} (non-concatenation axis)."
                )

    result = np.concatenate(matrices, axis=axis_pos)

    return result if return_numpy else result.tolist()
