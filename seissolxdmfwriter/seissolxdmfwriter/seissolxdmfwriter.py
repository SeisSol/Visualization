import numpy as np
import os


def write_timeseries_xdmf(
    prefix,
    nNodes,
    nCells,
    lDataName,
    lData,
    dt,
    node_per_element,
    lidt,
    reduce_precision,
    to_hdf5,
):
    precisionDict = {"int64": 8, "int32": 4, "float64": 8, "float32": 4}
    numberTypeDict = {
        "int64": "UInt",
        "int32": "UInt",
        "float64": "Float",
        "float32": "Float",
    }
    if to_hdf5:
        colon_or_nothing = ".h5:"
        ext = ""
        data_format = "HDF"
    else:
        colon_or_nothing = ""
        ext = ".bin"
        data_format = "Binary"

    topology = "Tetrahedron" if node_per_element == 4 else "Triangle"
    xdmf = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>"""
    for i, idt in enumerate(lidt):
        xdmf += f"""
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
   <Grid Name="step_{idt}" GridType="Uniform">
    <Topology TopologyType="{topology}" NumberOfElements="{nCells}">
     <DataItem NumberType="Int" Precision="8" Format="{data_format}" Dimensions="{nCells} {node_per_element}">{prefix}{colon_or_nothing}/connect{ext}</DataItem>
    </Topology>
    <Geometry name="geo" GeometryType="XYZ" NumberOfElements="{nNodes}">
     <DataItem NumberType="Float" Precision="8" Format="{data_format}" Dimensions="{nNodes} 3">{prefix}{colon_or_nothing}/geometry{ext}</DataItem>
    </Geometry>
    <Time Value="{idt*dt}"/>"""
        for k, dataName in enumerate(lDataName):
            prec = 4 if reduce_precision else precisionDict[lData[k].dtype.name]
            number_type = numberTypeDict[lData[k].dtype.name]
            xdmf += f"""
    <Attribute Name="{dataName}" Center="Cell">
     <DataItem ItemType="HyperSlab" Dimensions="{nCells}">
      <DataItem NumberType="UInt" Precision="4" Format="XML" Dimensions="3 2">{i} 0 1 1 1 {nCells}</DataItem>
      <DataItem NumberType="{number_type}" Precision="{prec}" Format="{data_format}" Dimensions="{i+1} {nCells}">{prefix}{colon_or_nothing}/{dataName}{ext}</DataItem>
     </DataItem>
    </Attribute>"""
        xdmf += """
   </Grid>
  </Grid>"""
    xdmf += """
 </Domain>
</Xdmf>
"""
    with open(prefix + ".xdmf", "w") as fid:
        fid.write(xdmf)
    print(f"done writing {prefix}.xdmf")


def write_mesh_xdmf(
    prefix,
    nNodes,
    nCells,
    lDataName,
    lData,
    node_per_element,
    reduce_precision,
    to_hdf5,
):
    precisionDict = {"int64": 8, "int32": 4, "float64": 8, "float32": 4}
    numberTypeDict = {
        "int64": "UInt",
        "int32": "UInt",
        "float64": "Float",
        "float32": "Float",
    }
    topology = "Tetrahedron" if node_per_element == 4 else "Triangle"
    if to_hdf5:
        colon_or_nothing = ".h5:"
        ext = ""
        data_format = "HDF"
    else:
        colon_or_nothing = ""
        ext = ".bin"
        data_format = "Binary"

    xdmf = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
  <Grid Name="puml mesh" GridType="Uniform">
   <Topology TopologyType="{topology}" NumberOfElements="{nCells}">
    <DataItem NumberType="Int" Precision="8" Format="{data_format}" Dimensions="{nCells} {node_per_element}">{prefix}{colon_or_nothing}/connect{ext}</DataItem>
   </Topology>
   <Geometry name="geo" GeometryType="XYZ" NumberOfElements="{nNodes}">
    <DataItem NumberType="Float" Precision="8" Format="{data_format}" Dimensions="{nNodes} 3">{prefix}.h5:/geometry</DataItem>
   </Geometry>"""
    for k, dataName in enumerate(lDataName):
        prec = 4 if reduce_precision else precisionDict[lData[k].dtype.name]
        number_type = numberTypeDict[lData[k].dtype.name]
        xdmf += f"""
    <Attribute Name="{dataName}" Center="Cell">
      <DataItem NumberType="{number_type}" Precision="{prec}" Format="{data_format}" Dimensions="1 {nCells}">{prefix}{colon_or_nothing}/{dataName}{ext}</DataItem>
    </Attribute>"""

        xdmf += """
  </Grid>
 </Domain>
</Xdmf>
"""
    with open(prefix + ".xdmf", "w") as fid:
        fid.write(xdmf)
    print(f"done writing {prefix}.xdmf")


def write_binaries(
    prefix, lDataName, xyz, connect, lData, lidt, reduce_precision, to_hdf5
):
    dtypeDict = {
        "int64": "i8",
        "int32": "i4",
        "float64": "float64",
        "float32": "float32",
    }
    reducePrecisionDict = {
        "i8": "i4",
        "i4": "i4",
        "float64": "float32",
        "float32": "float32",
    }
    nCells, node_per_element = connect.shape
    if to_hdf5:
        import h5py

        with h5py.File(prefix + ".h5", "w") as h5f:
            h5f.create_dataset("/connect", (nCells, node_per_element), dtype="uint64")
            h5f["/connect"][:, :] = connect[:, :]
            h5f.create_dataset("/geometry", xyz.shape, dtype="d")
            h5f["/geometry"][:, :] = xyz[:, :]
            for k, dataName in enumerate(lDataName):
                hdname = "/" + dataName
                mydtype = dtypeDict[lData[k].dtype.name]
                if reduce_precision:
                    mydtype = reducePrecisionDict[mydtype]
                if len(lData[0].shape) == 1 and len(lidt) == 1:
                    h5f.create_dataset(hdname, (nCells,), dtype=str(mydtype))
                    h5f[hdname][:] = lData[k][:]
                else:
                    h5f.create_dataset(hdname, (len(lidt), nCells), dtype=str(mydtype))
                    for i, idt in enumerate(lidt):
                        h5f[hdname][i, :] = lData[k][idt, :]
        print(f"done writing {prefix}.h5")
    else:
        os.makedirs(prefix, exist_ok=True)
        with open(f"{prefix}/geometry.bin", "wb") as fid:
            xyz.tofile(fid)
        with open(f"{prefix}/connect.bin", "wb") as fid:
            connect.tofile(fid)
        for k, dataName in enumerate(lDataName):
            mydtype = dtypeDict[lData[k].dtype.name]
            if reduce_precision:
                mydtype = reducePrecisionDict[mydtype]
            with open(f"{prefix}/{dataName}.bin", "wb") as fid:
                lData[k].astype(mydtype).tofile(fid)
        print(f"done writing binary files in {prefix}")


def write_seissol_output(
    prefix,
    xyz,
    connect,
    lDataName,
    lData,
    dt,
    lidt,
    reduce_precision=False,
    to_hdf5=True,
):
    """
    Write hdf5/xdmf files output, readable by ParaView using SeisSol data
    prefix: file
    xyz: geometry array
    connect: connect array
    lDataName: list of array e.g. ['ASl', 'Vr']
    lData: list of numpy data array
    dt: sampling time of output
    lidt: list of time steps to be written
    reduce_precision: convert double to float and i64 to i32 if True
    """
    nNodes = xyz.shape[0]
    nCells, node_per_element = connect.shape
    if dt == 0.0:
        write_mesh_xdmf(
            prefix,
            nNodes,
            nCells,
            lDataName,
            lData,
            node_per_element,
            reduce_precision,
            to_hdf5,
        )
    else:
        write_timeseries_xdmf(
            prefix,
            nNodes,
            nCells,
            lDataName,
            lData,
            dt,
            node_per_element,
            lidt,
            reduce_precision,
            to_hdf5,
        )
    write_binaries(
        prefix, lDataName, xyz, connect, lData, lidt, reduce_precision, to_hdf5
    )
