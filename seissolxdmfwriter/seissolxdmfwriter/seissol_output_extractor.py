#!/usr/bin/env python3
import argparse
import os
import os.path
from warnings import warn

import numpy as np
import seissolxdmf
import seissolxdmfwriter as sxw

import posixpath
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
import h5py
import shutil


def generate_new_prefix(prefix, append2prefix):
    prefix = os.path.basename(prefix)
    lsplit = prefix.split("-")
    if len(lsplit) > 1:
        if lsplit[-1] in ["surface", "low", "fault"]:
            prefix0 = "-".join(lsplit[0:-1])
            prefix_new = prefix0 + append2prefix + "-" + lsplit[-1]
        else:
            prefix_new = prefix + append2prefix
    else:
        prefix_new = prefix + append2prefix
    return prefix_new


parser = argparse.ArgumentParser(
    description="Extracts and processes data from SeisSol output files"
)
parser.add_argument("xdmfFilename", help="SeisSol XDMF output filename")
parser.add_argument(
    "--add2prefix",
    help="string to append to the prefix in the new file",
    type=str,
    default="_extracted",
)
parser.add_argument(
    "--variables",
    nargs="+",
    metavar="variables",
    help="Names of variables to extract (e.g. SRs or all)",
    default=["all"],
)
parser.add_argument(
    "--precision",
    type=str,
    choices=["float", "double"],
    default="float",
    help="precision of the data in the output file",
)
parser.add_argument(
    "--backend",
    type=str,
    choices=["hdf5", "raw"],
    default="hdf5",
    help="backend used: raw (.bin file), hdf5 (.h5)",
)
parser.add_argument(
    "--compression",
    type=int,
    default=4,
    help="compression level (for hdf5 format only)",
)
parser.add_argument(
    "--time",
    nargs=1,
    default=["i:"],
    help=(
        "simulation time or steps to extract, separated by ','. prepend a i for a step,"
        " or a python slice notation. E.g. 45.0,i2,i4:10:2,i-1 will extract a snapshot"
        " at simulation time 45.0, the 2nd time step, and time steps 4,6, 8 and the"
        " last time step"
    ),
)

parser.add_argument(
    "--xRange",
    nargs=2,
    metavar=("xmin", "xmax"),
    help="filter cells with x center coordinates in range xmin xmax",
    type=float,
)
parser.add_argument(
    "--yRange",
    nargs=2,
    metavar=("ymin", "ymax"),
    help="filter cells with y center coordinates in range ymin ymax",
    type=float,
)
parser.add_argument(
    "--zRange",
    nargs=2,
    metavar=("zmin", "zmax"),
    help="filter cells with z center coordinates in range zmin zmax",
    type=float,
)
parser.add_argument(
    "--regionFilter",
    nargs=1,
    metavar=("comma_separated_tags"),
    help="filter cells by faultTag (fault output), or locationFlag (surface output)",
)

parser.add_argument(
    "--jobs",
    type=int,
    default=1,
    help="number of worker processes (variables are split among them)",
)


class SeissolxdmfExtended(seissolxdmf.seissolxdmf):
    def ComputeTimeIndices(self, at_time):
        """retrive list of time index in file"""
        outputTimes = np.array(self.ReadTimes())
        nOutputTime = len(outputTimes)
        idsReadTimes = list(range(0, nOutputTime))
        lidt = []
        for oTime in at_time:
            if not oTime.startswith("i"):
                idsClose = np.where(np.isclose(outputTimes, float(oTime), atol=0.0001))
                if not len(idsClose[0]):
                    print(f"t={oTime} not found in {self.xdmfFilename}")
                else:
                    lidt.append(idsClose[0][0])
            else:
                sslice = oTime[1:]
                if ":" in sslice or sslice == "-1":
                    parts = sslice.split(":")
                    startstopstep = [None for i in range(3)]
                    for i, part in enumerate(parts):
                        startstopstep[i] = int(part) if part else None
                    new_indices = idsReadTimes[
                        startstopstep[0] : startstopstep[1] : startstopstep[2]
                    ]
                    new_indices = [v for v in new_indices if v < nOutputTime]
                    if len(new_indices) > 0:
                        lidt.extend(new_indices)
                else:
                    new_index = int(sslice)
                    if new_index < nOutputTime:
                        lidt.append(new_index)
        return sorted(list(set(lidt)))

    def ReadData(self, dataName, idt=-1):
        if dataName == "SR" and "SR" not in self.ReadAvailableDataFields():
            SRs = super().ReadData("SRs", idt)
            SRd = super().ReadData("SRd", idt)
            return np.sqrt(SRs**2 + SRd**2)
        else:
            return super().ReadData(dataName, idt)

    def GetDataLocationPrecisionMemDimension(self, dataName):
        if dataName == "SR" and "SR" not in self.ReadAvailableDataFields():
            return super().GetDataLocationPrecisionMemDimension("SRs")
        else:
            return super().GetDataLocationPrecisionMemDimension(dataName)

    def GetFilteredCells(self, regionFilter, xRange, yRange, zRange):
        spatial_filtering = (xRange or yRange) or zRange
        filter_cells = spatial_filtering or regionFilter
        if not filter_cells:
            return slice(None)

        ids = range(0, self.nElements)

        if regionFilter:
            available = self.ReadAvailableDataFields()
            possible_varnames = ["fault-tag", "locationFlag", "group"]
            region_varname = None
            for candidate in possible_varnames:
                if candidate in available:
                    region_varname = candidate
                    break
            if region_varname is None:
                raise ValueError(
                    f"None of {possible_varnames} are in available variables {available}"
                )

            tags = self.Read1dData(region_varname, self.nElements, isInt=True)
            regions = regionFilter[0].split(",")
            try:
                # Convert to a set of integers
                regions = set(int(region.strip()) for region in regions)
                print("Regions to filter:", regions)
            except:
                raise ValueError(
                    "Error: All elements in regionFilter must be integers."
                )
            ids = [v for v in ids if tags[v] in regions]
            print(f"cell count after region filtering: {len(ids)}/{self.nElements}")

        if spatial_filtering:
            xyz = self.ReadGeometry()
            connect = self.ReadConnect()
            warn("spatial filtering significantly slows down this script")
            xyzc = (
                xyz[connect[:, 0], :] + xyz[connect[:, 1], :] + xyz[connect[:, 2], :]
            ) / 3.0

            def filter_cells(coords, filter_range):
                m = 0.5 * (filter_range[0] + filter_range[1])
                d = 0.5 * (filter_range[1] - filter_range[0])
                return np.where(np.abs(coords[:] - m) < d)[0]

            if xRange:
                id0 = filter_cells(xyzc[:, 0], xRange)
                ids = np.intersect1d(ids, id0)
            if yRange:
                id0 = filter_cells(xyzc[:, 1], yRange)
                ids = np.intersect1d(ids, id0)
            if zRange:
                id0 = filter_cells(xyzc[:, 2], zRange)
                ids = np.intersect1d(ids, id0)

            print(f"cell count after spatial filtering: {len(ids)}/{self.nElements}")
        if not len(ids):
            raise ValueError("all elements are outside filter range")
        return ids


def _extract_chunk(
    part_prefix,
    xdmf_filename,
    variables,
    indices,
    ids,
    backend,
    compression,
    reduce_precision,
):
    # each worker opens its own file handle; compression happens here, in parallel
    sx = SeissolxdmfExtended(xdmf_filename)
    sxw.write_from_seissol_output(
        part_prefix,
        sx,
        variables,
        indices,
        reduce_precision=reduce_precision,
        backend=backend,
        compression_level=compression,
        filtered_cells=ids,
    )
    return part_prefix


def _merge_h5(part_prefixes, final_prefix):
    # copies already-compressed datasets, so nothing is recompressed
    with h5py.File(final_prefix + ".h5", "w") as out:
        for p in part_prefixes:
            with h5py.File(p + ".h5", "r") as src:

                def copy_new(name, obj):
                    # geometry/connectivity exist in every part: keep the first copy
                    if isinstance(obj, h5py.Dataset) and name not in out:
                        parent = posixpath.dirname(name)
                        if parent:
                            out.require_group(parent)
                        src.copy(obj, out, name=name)

                src.visititems(copy_new)
            os.remove(p + ".h5")


def _merge_raw(part_prefixes, final_prefix):
    # the raw backend writes one directory per prefix, containing the .bin files
    os.makedirs(final_prefix, exist_ok=True)
    for p in part_prefixes:
        for fn in os.listdir(p):
            dst = os.path.join(final_prefix, fn)
            if os.path.exists(dst):
                continue  # geometry.bin / connect.bin are the same in every part
            shutil.move(os.path.join(p, fn), dst)
        shutil.rmtree(p)


def _merge_xdmf(part_prefixes, final_prefix, ref_suffix):
    trees = [ET.parse(p + ".xdmf") for p in part_prefixes]
    base = trees[0]
    base_grids = base.getroot().findall(".//Grid[@GridType='Uniform']")
    if not base_grids:
        raise RuntimeError("no Uniform grids found in xdmf, cannot merge")
    for other in trees[1:]:
        other_grids = other.getroot().findall(".//Grid[@GridType='Uniform']")
        if len(other_grids) != len(base_grids):
            raise RuntimeError("time step count differs between part files")
        for g_base, g_other in zip(base_grids, other_grids):
            for attr in g_other.findall("Attribute"):
                g_base.append(attr)

    # point all DataItems from the part files/directories to the merged ones
    final_name = os.path.basename(final_prefix)
    for item in base.getroot().iter("DataItem"):
        if item.text:
            for p in part_prefixes:
                item.text = item.text.replace(
                    os.path.basename(p) + ref_suffix, final_name + ref_suffix
                )

    base.write(final_prefix + ".xdmf")
    for p in part_prefixes:
        os.remove(p + ".xdmf")


def main():
    args = parser.parse_args()
    sx = SeissolxdmfExtended(args.xdmfFilename)
    ids = sx.GetFilteredCells(args.regionFilter, args.xRange, args.yRange, args.zRange)
    indices = sx.ComputeTimeIndices(args.time[0].split(","))
    prefix = os.path.splitext(args.xdmfFilename)[0]
    prefix_new = generate_new_prefix(prefix, args.add2prefix)
    reduce_precision = args.precision == "float"

    if args.variables[0] == "all":
        variables = sorted(sx.ReadAvailableDataFields())
        print(f"args.variables was set to all and now contains {variables}")
    else:
        variables = args.variables

    if args.backend == "hdf5" and args.compression > 0:
        print(
            "Writing hdf5 output with compression enabled"
            f" (compression_level={args.compression}). \n"
            "Use --compression=0 if you want to speed-up data extraction."
        )

    n = max(1, min(args.jobs, len(variables)))

    if n == 1:
        # Keep the original serial execution path: no subprocesses,
        # temporary files, or merge step.
        sxw.write_from_seissol_output(
            prefix_new,
            sx,
            variables,
            indices,
            reduce_precision=reduce_precision,
            backend=args.backend,
            compression_level=args.compression,
            filtered_cells=ids,
        )
    else:
        chunks = [variables[i::n] for i in range(n)]
        part_prefixes = [f"{prefix_new}.part{i}" for i in range(n)]

        with ProcessPoolExecutor(max_workers=n) as ex:
            list(
                ex.map(
                    _extract_chunk,
                    part_prefixes,
                    [args.xdmfFilename] * n,
                    chunks,
                    [indices] * n,
                    [ids] * n,
                    [args.backend] * n,
                    [args.compression] * n,
                    [reduce_precision] * n,
                )
            )

        if args.backend == "hdf5":
            _merge_h5(part_prefixes, prefix_new)
            ref_suffix = ".h5"
        else:
            _merge_raw(part_prefixes, prefix_new)
            ref_suffix = "/"
        _merge_xdmf(part_prefixes, prefix_new, ref_suffix)


if __name__ == "__main__":
    main()
