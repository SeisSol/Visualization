import sys

import h5py
import numpy as np
from tqdm import tqdm

# Fields that do not change in time: written once, offsets all zero.
STATIC_1D_ARRAYS = {
    "locationflag",
    "fault-tag",
    "partition",
    "clustering",
    "global-id",
    "group",
    "boundary",
}


def _vtk_cell_type(node_per_cell):
    # VTK cell type mapping for simplex cells
    mapping = {
        3: 5,  # VTK_TRIANGLE
        4: 10,  # VTK_TETRA
    }
    if node_per_cell not in mapping:
        raise ValueError(
            f"Unsupported cell topology for VTKHDF: {node_per_cell} nodes/cell. "
            "Supported: 3 (triangle), 4 (tetra)."
        )
    return mapping[node_per_cell]


def _is_static(name):
    return name.lower() in STATIC_1D_ARRAYS


def _infer_field_location(name, sx, arr, n_cells, n_points):
    """
    Return the VTKHDF target group: 'PointData' or 'CellData'.
    SeisSol output is cell based, so CellData is the default.
    """
    if hasattr(sx, "GetFieldLocation"):
        loc = sx.GetFieldLocation(name)
        if loc in ("PointData", "CellData"):
            return loc
    if _is_static(name):
        return "CellData"
    if arr.shape[0] == n_cells:
        return "CellData"
    if arr.shape[0] == n_points:
        return "PointData"
    return "CellData"


def _target_dtype(arr, reduce_precision):
    kind = arr.dtype.kind
    if kind == "f":
        return np.dtype(np.float32) if reduce_precision else arr.dtype
    if kind in ("i", "u"):
        if reduce_precision and arr.dtype.itemsize == 8:
            return np.dtype(np.int32 if kind == "i" else np.uint32)
        return arr.dtype
    return np.dtype(np.float32)


def _compression_options(compression_level):
    if compression_level and compression_level > 0:
        return {"compression": "gzip", "compression_opts": int(compression_level)}
    return {}


def _write_vtkhdf_mesh(vtk, xyz, connect, cell_type, version=(1, 0)):
    """Write the mesh into the VTKHDF group (static, written once)."""
    npoints = int(xyz.shape[0])
    ncells = int(connect.shape[0])
    nper = int(connect.shape[1])

    flat = np.asarray(connect, dtype=np.int64).ravel()
    offsets = np.arange(ncells + 1, dtype=np.int64) * nper
    types = np.full(ncells, cell_type, dtype=np.uint8)

    vtk.attrs["Type"] = np.bytes_(b"UnstructuredGrid")
    vtk.attrs["Version"] = np.array(version, dtype=np.int64)

    vtk.create_dataset("NumberOfPoints", data=np.array([npoints], dtype=np.int64))
    vtk.create_dataset("NumberOfCells", data=np.array([ncells], dtype=np.int64))
    vtk.create_dataset(
        "NumberOfConnectivityIds", data=np.array([flat.size], dtype=np.int64)
    )

    vtk.create_dataset("Points", data=np.asarray(xyz))
    vtk.create_dataset("Connectivity", data=flat)
    vtk.create_dataset("Offsets", data=offsets)
    vtk.create_dataset("Types", data=types)


def _write_field_data(vtk, time_value, index_value):
    field = vtk.require_group("FieldData")
    field.create_dataset("Time", data=np.array([float(time_value)], dtype=np.float64))
    field.create_dataset("Index", data=np.array([int(index_value)], dtype=np.uint64))


def _create_steps_group(vtk, time_values):
    """
    /VTKHDF/Steps for a transient file with a static mesh: every step uses the
    same mesh part, so all mesh offsets are zero.
    """
    nsteps = len(time_values)
    steps = vtk.create_group("Steps")
    steps.attrs["NSteps"] = np.int64(nsteps)
    steps.create_dataset("Values", data=np.asarray(time_values, dtype=np.float64))
    for key in ("PartOffsets", "PointOffsets", "CellOffsets", "ConnectivityIdOffsets"):
        steps.create_dataset(key, data=np.zeros(nsteps, dtype=np.int64))
    steps.create_dataset("NumberOfParts", data=np.ones(nsteps, dtype=np.int64))
    steps.create_group("PointDataOffsets")
    steps.create_group("CellDataOffsets")
    return steps


def _read_static(sx, name, n_elements):
    """Read a time independent 1D array (tags, partition, ...)."""
    if hasattr(sx, "Read1dData"):
        arr = np.asarray(sx.Read1dData(name, n_elements, isInt=True))
    else:
        arr = np.asarray(sx.ReadData(name, 0))
    return arr.reshape(-1)


def _read_field(sx, name, idt, n_elements, n_points, indices):
    arr = np.asarray(sx.ReadData(name, idt))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    loc = _infer_field_location(name, sx, arr, n_elements, n_points)
    if loc == "CellData" and indices is not None and arr.shape[0] == n_elements:
        arr = arr[indices]
    return arr, loc


def _fit_step(arr, n, ref, name, idt):
    """Replace corrupted (wrong length) time steps by NaN."""
    if arr.shape[0] == n:
        return arr
    if ref.dtype.kind != "f":
        raise ValueError(
            f"time step {idt} of {name} has {arr.shape[0]} values, expected {n}"
        )
    print(f"time step {idt} of {name} is corrupted, replacing with nans")
    return np.full((n,) + ref.shape[1:], np.nan, dtype=ref.dtype)


def write_vtkhdf_from_seissol_output(
    prefix,
    sx,
    var_names,
    time_indices,
    reduce_precision=False,
    compression_level=4,
    filtered_cells=slice(None),
):
    """
    Write VTKHDF output from a SeisSol reader object.

    One time index  -> prefix.vtkhdf (steady file, VTKHDF 1.0)
    Several indices -> prefix.vtkhdf as a single transient file (VTKHDF 2.0,
                       static mesh, /VTKHDF/Steps), one file with all time steps.
    """
    if isinstance(time_indices, (int, np.integer)):
        time_indices = [int(time_indices)]
    else:
        time_indices = [int(v) for v in time_indices]
    if len(time_indices) == 0:
        time_indices = [0]

    nsteps = len(time_indices)
    transient = nsteps > 1
    comp = _compression_options(compression_level)

    # Mesh
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()

    use_filter = not (
        isinstance(filtered_cells, slice) and filtered_cells == slice(None)
    )
    if use_filter:
        indices = np.asarray(filtered_cells)
        connect = connect[indices, :]
    else:
        indices = None

    cell_type = _vtk_cell_type(connect.shape[1])
    n_elements = sx.ReadNElements()  # number of cells before filtering
    n_points = int(xyz.shape[0])

    all_times = sx.ReadTimes() if hasattr(sx, "ReadTimes") else [0.0]
    time_values = [all_times[i] if i < len(all_times) else 0.0 for i in time_indices]

    outname = f"{prefix}.vtkhdf"
    with h5py.File(outname, "w") as h5f:
        vtk = h5f.create_group("VTKHDF")
        _write_vtkhdf_mesh(
            vtk, xyz, connect, cell_type, version=(2, 0) if transient else (1, 0)
        )

        if transient:
            steps = _create_steps_group(vtk, time_values)
        else:
            _write_field_data(vtk, time_values[0], 0)

        for name in var_names:
            if _is_static(name):
                arr = _read_static(sx, name, n_elements)
                if use_filter and arr.shape[0] == n_elements:
                    arr = arr[indices]
                loc = "CellData"
                dtype = _target_dtype(arr, reduce_precision)
                group = vtk.require_group(loc)
                group.create_dataset(name, data=arr.astype(dtype), **comp)
                if transient:
                    steps[f"{loc}Offsets"].create_dataset(
                        name, data=np.zeros(nsteps, dtype=np.int64)
                    )
                continue

            first, loc = _read_field(
                sx, name, time_indices[0], n_elements, n_points, indices
            )
            dtype = _target_dtype(first, reduce_precision)
            group = vtk.require_group(loc)

            if not transient:
                group.create_dataset(name, data=first.astype(dtype), **comp)
                continue

            n = first.shape[0]
            ds = group.create_dataset(
                name, shape=(nsteps * n,) + first.shape[1:], dtype=dtype, **comp
            )
            ds[0:n] = first.astype(dtype)
            for k in tqdm(
                range(1, nsteps), file=sys.stdout, desc=name, dynamic_ncols=False
            ):
                arr, _ = _read_field(
                    sx, name, time_indices[k], n_elements, n_points, indices
                )
                arr = _fit_step(arr, n, first, name, time_indices[k])
                ds[k * n : (k + 1) * n] = arr.astype(dtype)

            offsets = np.arange(nsteps, dtype=np.int64) * n
            steps[f"{loc}Offsets"].create_dataset(name, data=offsets)

    print(f"done writing {outname} ({nsteps} time step{'s' if nsteps > 1 else ''})")
