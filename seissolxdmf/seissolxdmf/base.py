import numpy as np
from warnings import warn


class SeisSolReaderBase:
    REGION_VARNAMES = ("fault-tag", "locationFlag", "group")

    def ComputeTimeIndices(self, at_time):
        output_times = np.asarray(self.ReadTimes())
        n = len(output_times)
        all_ids = list(range(n))
        indices = []
        for t in at_time:
            if not t.startswith("i"):
                close = np.flatnonzero(np.isclose(output_times, float(t), atol=1e-4))
                if close.size:
                    indices.append(int(close[0]))
                else:
                    print(f"t={t} not found in {self.xdmfFilename}")
            else:
                s = t[1:]
                if ":" in s or s == "-1":
                    parts = [int(p) if p else None for p in s.split(":")]
                    parts += [None] * (3 - len(parts))
                    indices.extend(all_ids[slice(*parts)])
                elif int(s) < n:
                    indices.append(int(s))
        return sorted(set(indices))

    def _cell_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        if (connect >= 0).all():  # fixed-size cells: vectorised
            return xyz[connect].mean(axis=1)
        valid = connect >= 0  # padded (mixed) cells
        pts = xyz[np.where(valid, connect, 0)]
        return (pts * valid[..., None]).sum(axis=1) / valid.sum(axis=1)[:, None]

    def _region_ids(self, regionFilter):
        available = self.ReadAvailableDataFields()
        name = next((c for c in self.REGION_VARNAMES if c in available), None)
        if name is None:
            raise ValueError(
                f"None of {self.REGION_VARNAMES} are in available variables {available}"
            )
        try:
            regions = {int(r.strip()) for r in regionFilter[0].split(",")}
        except ValueError:
            raise ValueError("All elements in regionFilter must be integers.")
        tags = np.asarray(self.ReadData(name, 0), dtype=np.int64)

        ids = np.flatnonzero(np.isin(tags, list(regions)))
        print(f"cell count after region filtering: {len(ids)}/{self.nElements}")
        return ids

    def GetFilteredCells(self, regionFilter, xRange, yRange, zRange):
        spatial = bool(xRange or yRange or zRange)
        if not (spatial or regionFilter):
            return slice(None)

        ids = np.arange(self.nElements)
        if regionFilter:
            ids = self._region_ids(regionFilter)

        if spatial:
            warn("spatial filtering significantly slows down this script")
            centers = self._cell_centers()
            for axis, rng in enumerate((xRange, yRange, zRange)):
                if rng:
                    m, d = 0.5 * (rng[0] + rng[1]), 0.5 * (rng[1] - rng[0])
                    ids = ids[np.abs(centers[ids, axis] - m) < d]
            print(f"cell count after spatial filtering: {len(ids)}/{self.nElements}")

        if len(ids) == 0:
            raise ValueError("all elements are outside filter range")
        return ids
