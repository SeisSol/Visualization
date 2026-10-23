#!/usr/bin/env python3
import argparse
import glob
import os
import os.path
import re
from warnings import warn

import numpy as np
import seissolxdmf
from seissolxdmf import VTKHDFSeries

import seissolxdmfwriter as sxw


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


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def expand_and_sort_files(patterns):
    """Expand wildcards (if the shell did not) and sort naturally."""
    files = []
    for p in patterns:
        if any(c in p for c in "*?["):
            files.extend(glob.glob(p))
        else:
            files.append(p)
    # remove duplicates, keep natural order
    return sorted(set(files), key=natural_key)


def series_prefix(files):
    """
    Common prefix of a file series without the trailing step counter, e.g.
    a-fc-surface-0001.vtkhdf, a-fc-surface-0002.vtkhdf -> a-fc-surface
    """
    first = os.path.splitext(files[0])[0]
    if len(files) == 1:
        return first
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    common = os.path.commonprefix(names)
    common = re.sub(r"[-_.\d]+$", "", common)
    if not common:
        return first
    return os.path.join(os.path.dirname(files[0]), common)


parser = argparse.ArgumentParser(
    description="Extracts and processes data from SeisSol output files"
)
parser.add_argument(
    "filenames",
    nargs="+",
    metavar="filename",
    help=(
        "SeisSol output file(s): an XDMF file, or one or more .vtkhdf files."
        " Several .vtkhdf files (one time step each, same mesh) are merged"
        " into a single transient .vtkhdf file."
    ),
)
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
    choices=["hdf5", "raw", "vtkhdf"],
    default="hdf5",
    help="backend used: raw (.bin), hdf5 (.h5), vtkhdf (.vtkhdf)",
)
parser.add_argument(
    "--compression",
    type=int,
    default=4,
    help="compression level (for hdf5 and vtkhdf formats only)",
)
parser.add_argument(
    "--time",
    nargs=1,
    default=["i:"],
    help=(
        "simulation time or steps to extract, separated by ','. prepend a i for a step,"
        " or a python slice notation. E.g. 45.0,i2,i4:10:2,i-1 will extract a snapshot"
        " at simulation time 45.0, the 2nd time step, and time steps 4,6, 8 and the"
        " last time step. When several input files are merged, indices refer to the"
        " merged, time-sorted series."
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

args = parser.parse_args()


class SeissolxdmfExtended(seissolxdmf.seissolxdmf):
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


def main():
    files = expand_and_sort_files(args.filenames)
    if not files:
        parser.error("no input file found")

    all_vtkhdf = all(f.endswith(".vtkhdf") for f in files)
    if len(files) > 1 and not all_vtkhdf:
        parser.error("several input files are only supported for .vtkhdf files")

    if all_vtkhdf:
        if len(files) > 1:
            print(f"merging {len(files)} vtkhdf files")
        sx = VTKHDFSeries(files)
    else:
        sx = SeissolxdmfExtended(files[0])

    ids = sx.GetFilteredCells(args.regionFilter, args.xRange, args.yRange, args.zRange)

    indices = sx.ComputeTimeIndices(args.time[0].split(","))

    prefix = series_prefix(files)
    prefix_new = generate_new_prefix(prefix, args.add2prefix)

    # Write data items
    if args.variables[0] == "all":
        args.variables = sorted(sx.ReadAvailableDataFields())
        print(f"args.variables was set to all and now contains {args.variables}")

    if args.backend in ("hdf5", "vtkhdf") and args.compression > 0:
        print(
            f"Writing {args.backend} output with compression enabled"
            f" (compression_level={args.compression}). \n"
            "Use --compression=0 if you want to speed-up data extraction."
        )

    if args.backend == "vtkhdf" or all_vtkhdf:
        sxw.write_vtkhdf_from_seissol_output(
            prefix_new,
            sx,
            args.variables,
            indices,
            reduce_precision=(args.precision == "float"),
            compression_level=args.compression,
            filtered_cells=ids,
        )
    else:
        sxw.write_from_seissol_output(
            prefix_new,
            sx,
            args.variables,
            indices,
            reduce_precision=(args.precision == "float"),
            backend=args.backend,
            compression_level=args.compression,
            filtered_cells=ids,
        )


if __name__ == "__main__":
    main()
