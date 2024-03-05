#!/usr/bin/env python3
import os
import os.path
import seissolxdmf
import seissolxdmfwriter as sxw
import numpy as np
import argparse
from warnings import warn


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
args = parser.parse_args()


class SeissolxdmfExtended(seissolxdmf.seissolxdmf):
    def ComputeTimeIndices(self, at_time):
        """retrive list of time index in file"""
        outputTimes = np.array(self.ReadTimes())
        idsReadTimes = list(range(0, len(outputTimes)))
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
                    lidt.extend(
                        idsReadTimes[
                            startstopstep[0] : startstopstep[1] : startstopstep[2]
                        ]
                    )
                else:
                    lidt.append(int(sslice))
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


def main():
    sx = SeissolxdmfExtended(args.xdmfFilename)
    spatial_filtering = (args.xRange or args.yRange) or args.zRange

    if spatial_filtering:
        xyz = sx.ReadGeometry()
        connect = sx.ReadConnect()
        warn("spatial filtering significantly slows down this script")
        ids = range(0, sx.nElements)
        xyzc = (
            xyz[connect[:, 0], :] + xyz[connect[:, 1], :] + xyz[connect[:, 2], :]
        ) / 3.0

        def filter_cells(coords, filter_range):
            m = 0.5 * (filter_range[0] + filter_range[1])
            d = 0.5 * (filter_range[1] - filter_range[0])
            return np.where(np.abs(coords[:] - m) < d)[0]

        if args.xRange:
            id0 = filter_cells(xyzc[:, 0], args.xRange)
            ids = np.intersect1d(ids, id0) if len(ids) else id0
        if args.yRange:
            id0 = filter_cells(xyzc[:, 1], args.yRange)
            ids = np.intersect1d(ids, id0) if len(ids) else id0
        if args.zRange:
            id0 = filter_cells(xyzc[:, 2], args.zRange)
            ids = np.intersect1d(ids, id0) if len(ids) else id0

        if len(ids):
            nElements = ids.shape[0]
            if nElements != sx.nElements:
                print(f"extracting {nElements} cells out of {sx.nElements}")
            else:
                spatial_filtering = False
        else:
            raise ValueError("all elements are outside filter range")

    if not spatial_filtering:
        ids = slice(None)

    indices = sx.ComputeTimeIndices(args.time[0].split(","))

    prefix = os.path.splitext(args.xdmfFilename)[0]
    prefix_new = generate_new_prefix(prefix, args.add2prefix)

    # Write data items
    if args.variables[0] == "all":
        args.variables = sorted(sx.ReadAvailableDataFields())
        print(f"args.variables was set to all and now contains {args.variables}")

    if args.backend == "hdf5" and args.compression > 0:
        print(
            "Writing hdf5 output with compression enabled"
            f" (compression_level={args.compression}). \n"
            "Use --compression=0 if you want to speed-up data extraction."
        )

    sxw.write_from_seissol_output(
        prefix_new,
        sx,
        args.variables,
        indices,
        reduce_precision=True,
        backend=args.backend,
        compression_level=args.compression,
        filtered_cells=ids,
    )


if __name__ == "__main__":
    main()
