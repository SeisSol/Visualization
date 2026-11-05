import h5py
import numpy as np
from .base import SeisSolReaderBase


class VTKHDFReader(SeisSolReaderBase):
    """
    Reader for VTKHDF unstructured grids written by SeisSol.

    Supports
      - steady files / one time step per file (VTKHDF version 1.x)
      - transient files with a static mesh (VTKHDF version 2.x, /VTKHDF/Steps)
    """

    def __init__(self, filename):
        self.xdmfFilename = filename
        self.h5 = h5py.File(filename, "r")
        self.root = self.h5["/VTKHDF"]

        # transient layout: field arrays concatenated along axis 0, see /VTKHDF/Steps
        self._transient = "Steps" in self.root
        self._time_values = self._read_times()

        # mesh size
        self.nCells = int(self.root["NumberOfCells"][0])
        self.nPoints = int(self.root["NumberOfPoints"][0])
        self.nElements = self.nCells

    def _read_times(self):
        if self._transient:
            try:
                t = self.root["Steps"]["Values"][...]
                return np.asarray(t).ravel().tolist()
            except Exception:
                pass
        try:
            t = self.root["FieldData"]["Time"][...]
            return np.asarray(t).ravel().tolist()
        except Exception:
            return [0.0]

    def ReadTimes(self):
        return self._time_values

    def ReadAvailableDataFields(self):
        names = []
        for group_name in ("CellData", "PointData"):
            group = self.root.get(group_name)
            if group is not None:
                names.extend(list(group.keys()))
        return sorted(set(names))

    def ReadGeometry(self):
        return np.asarray(self.root["Points"][...])

    def ReadConnect(self):
        # Standard VTKHDF connectivity: flattened list of point ids
        conn = np.asarray(self.root["Connectivity"][...], dtype=np.int64)
        offsets = np.asarray(self.root["Offsets"][...], dtype=np.int64)

        # If all cells have the same size, reshape to (nCells, nPerCell).
        if len(offsets) > 1:
            cell_sizes = np.diff(offsets)
            if np.all(cell_sizes == cell_sizes[0]):
                return conn.reshape(self.nCells, cell_sizes[0])

        # Fallback: variable-sized cells (rare but possible for mixed types)
        # We create a dense matrix with the maximum cell size and pad with -1.
        max_size = int(np.max(np.diff(offsets))) if len(offsets) > 1 else 0
        out = np.full((self.nCells, max_size), -1, dtype=np.int64)
        for ic in range(self.nCells):
            a, b = int(offsets[ic]), int(offsets[ic + 1])
            out[ic, : b - a] = conn[a:b]
        return out

    def ReadNNodes(self):
        return self.nPoints

    def ReadNElements(self):
        return self.nCells

    def ReadNodesPerElement(self):
        conn = self.ReadConnect()
        if conn.ndim == 1:
            return 1
        return conn.shape[1]

    def ReadData(self, dataName, idt=-1):
        """
        Steady files (one time step per file): idt is ignored.
        Transient files: returns time step idt (idt=-1 is the last step).
        """
        for group_name in ("CellData", "PointData"):
            group = self.root.get(group_name)
            if group is None or dataName not in group:
                continue
            ds = group[dataName]
            if not self._transient:
                return np.asarray(ds[...])

            n = self.nCells if group_name == "CellData" else self.nPoints
            nsteps = len(self._time_values)
            step = idt if idt >= 0 else nsteps + idt
            if not 0 <= step < nsteps:
                raise IndexError(f"time step {idt} out of range (nsteps={nsteps})")
            offsets_group = self.root["Steps"].get(f"{group_name}Offsets")
            if offsets_group is not None and dataName in offsets_group:
                start = int(offsets_group[dataName][step])
            else:
                start = 0
            return np.asarray(ds[start : start + n])
        raise KeyError(f"{dataName} not found in VTKHDF file")

    def GetDataLocationPrecisionMemDimension(self, dataName):
        arr = self.ReadData(dataName, 0)
        dtype = arr.dtype
        if np.issubdtype(dtype, np.floating):
            prec = 8 if dtype == np.float64 else 4
            return dataName, prec, arr.shape
        elif np.issubdtype(dtype, np.integer):
            prec = 8 if dtype in (np.int64, np.uint64) else 4
            return dataName, prec, arr.shape
        else:
            return dataName, 4, arr.shape


class VTKHDFSeries(VTKHDFReader):
    """
    Several VTKHDF files sharing the same mesh (typically one time step each),
    presented as a single time series sorted by time.
    Time index idt refers to the merged, time-sorted series.
    """

    def __init__(self, filenames):
        if len(filenames) == 0:
            raise ValueError("VTKHDFSeries needs at least one file")
        self.files = [VTKHDFReader(f) for f in filenames]
        first = self.files[0]

        for fname, r in zip(filenames, self.files):
            if r.nCells != first.nCells or r.nPoints != first.nPoints:
                raise ValueError(
                    f"{fname} has a different mesh "
                    f"({r.nCells} cells, {r.nPoints} points) than {filenames[0]} "
                    f"({first.nCells} cells, {first.nPoints} points)"
                )

        # (file index, time index inside that file, time value), sorted by time
        entries = []
        for ifile, r in enumerate(self.files):
            for ilocal, t in enumerate(r.ReadTimes()):
                entries.append((ifile, ilocal, float(t)))
        entries.sort(key=lambda e: e[2])
        self._index = [(e[0], e[1]) for e in entries]

        self.xdmfFilename = filenames[0]
        self.h5 = first.h5
        self.root = first.root
        self._transient = False
        self.nCells = first.nCells
        self.nPoints = first.nPoints
        self.nElements = first.nCells
        self._time_values = [e[2] for e in entries]

    def ReadData(self, dataName, idt=-1):
        ifile, ilocal = self._index[idt]
        return self.files[ifile].ReadData(dataName, ilocal)

    def close(self):
        for r in self.files:
            r.h5.close()
