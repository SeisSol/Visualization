import numpy as np
import os


def dataLocation(prefix, name, backend):
    if backend == "hdf5":
        colon_or_nothing = ".h5:"
        ext = ""
    else:
        colon_or_nothing = ""
        ext = ".bin"
    return f"{prefix}{colon_or_nothing}/{name}{ext}"


def write_timeseries_xdmf(
    prefix,
    xyz,
    connect,
    dictData,
    dictTime,
    reduce_precision,
    backend,
):
    precisionDict = {"int64": 8, "int32": 4, "float64": 8, "float32": 4}
    numberTypeDict = {
        "int64": "UInt",
        "int32": "UInt",
        "float64": "Float",
        "float32": "Float",
    }

    data_format = "HDF" if backend == "hdf5" else "Binary"
    nNodes = xyz.shape[0]
    nCells, node_per_element = connect.shape
    lDataName = list(dictData.keys())
    lData = list(dictData.values())
    topology = "Tetrahedron" if node_per_element == 4 else "Triangle"
    xdmf = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>"""
    geometry_location = dataLocation(prefix, "geometry", backend)
    connect_location = dataLocation(prefix, "connect", backend)

    for i, ctime in enumerate(list(dictTime.keys())):
        index = dictTime[ctime]
        xdmf += f"""
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
   <Grid Name="step_{index}" GridType="Uniform">
    <Topology TopologyType="{topology}" NumberOfElements="{nCells}">
     <DataItem NumberType="Int" Precision="8" Format="{data_format}" Dimensions="{nCells} {node_per_element}">{connect_location}</DataItem>
    </Topology>
    <Geometry name="geo" GeometryType="XYZ" NumberOfElements="{nNodes}">
     <DataItem NumberType="Float" Precision="8" Format="{data_format}" Dimensions="{nNodes} 3">{geometry_location}</DataItem>
    </Geometry>
    <Time Value="{ctime}"/>"""
        for k, dataName in enumerate(lDataName):
            prec = 4 if reduce_precision else precisionDict[lData[k].dtype.name]
            number_type = numberTypeDict[lData[k].dtype.name]
            data_location = dataLocation(prefix, dataName, backend)
            xdmf += f"""
    <Attribute Name="{dataName}" Center="Cell">
     <DataItem ItemType="HyperSlab" Dimensions="{nCells}">
      <DataItem NumberType="UInt" Precision="4" Format="XML" Dimensions="3 2">{i} 0 1 1 1 {nCells}</DataItem>
      <DataItem NumberType="{number_type}" Precision="{prec}" Format="{data_format}" Dimensions="{i+1} {nCells}">{data_location}</DataItem>
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
    xyz,
    connect,
    dictData,
    reduce_precision,
    backend,
):
    precisionDict = {"int64": 8, "int32": 4, "float64": 8, "float32": 4}
    numberTypeDict = {
        "int64": "UInt",
        "int32": "UInt",
        "float64": "Float",
        "float32": "Float",
    }

    data_format = "HDF" if backend == "hdf5" else "Binary"
    nNodes = xyz.shape[0]
    nCells, node_per_element = connect.shape
    topology = "Tetrahedron" if node_per_element == 4 else "Triangle"
    lDataName = list(dictData.keys())
    lData = list(dictData.values())
    geometry_location = dataLocation(prefix, "geometry", backend)
    connect_location = dataLocation(prefix, "connect", backend)

    xdmf = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
  <Grid Name="puml mesh" GridType="Uniform">
   <Topology TopologyType="{topology}" NumberOfElements="{nCells}">
    <DataItem NumberType="Int" Precision="8" Format="{data_format}" Dimensions="{nCells} {node_per_element}">{connect_location}</DataItem>
   </Topology>
   <Geometry name="geo" GeometryType="XYZ" NumberOfElements="{nNodes}">
    <DataItem NumberType="Float" Precision="8" Format="{data_format}" Dimensions="{nNodes} 3">{geometry_location}</DataItem>
   </Geometry>"""
    for k, dataName in enumerate(lDataName):
        prec = 4 if reduce_precision else precisionDict[lData[k].dtype.name]
        number_type = numberTypeDict[lData[k].dtype.name]
        data_location = dataLocation(prefix, dataName, backend)
        xdmf += f"""
    <Attribute Name="{dataName}" Center="Cell">
      <DataItem NumberType="{number_type}" Precision="{prec}" Format="{data_format}" Dimensions="1 {nCells}">{data_location}</DataItem>
    </Attribute>"""

    xdmf += """
  </Grid>
 </Domain>
</Xdmf>
"""
    with open(prefix + ".xdmf", "w") as fid:
        fid.write(xdmf)
    print(f"done writing {prefix}.xdmf")


def write_data(
    prefix,
    xyz,
    connect,
    dictData,
    dictTime,
    reduce_precision,
    backend,
    compression_level,
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
    lDataName = list(dictData.keys())
    lData = list(dictData.values())

    if backend == "hdf5":
        import h5py

        compression_options = {}
        if compression_level:
            compression_options = {
                "compression": "gzip",
                "compression_opts": compression_level,
            }

        with h5py.File(prefix + ".h5", "w") as h5f:
            h5f.create_dataset(
                "/connect",
                (nCells, node_per_element),
                dtype="uint64",
                **compression_options,
            )
            h5f["/connect"][:, :] = connect[:, :]
            h5f.create_dataset("/geometry", xyz.shape, dtype="d", **compression_options)
            h5f["/geometry"][:, :] = xyz[:, :]
            for k, dataName in enumerate(lDataName):
                hdname = "/" + dataName
                mydtype = dtypeDict[lData[k].dtype.name]
                if reduce_precision:
                    mydtype = reducePrecisionDict[mydtype]
                if len(lData[k].shape) == 1:
                    lData[k] = lData[k][np.newaxis, :]
                h5f.create_dataset(
                    hdname,
                    (len(dictTime), nCells),
                    dtype=str(mydtype),
                    **compression_options,
                )
                for i, idt in enumerate(list(dictTime.values())):
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
            if len(lData[k].shape) == 1:
                lData[k] = lData[k][np.newaxis, :]
            with open(f"{prefix}/{dataName}.bin", "wb") as fid:
                if not dictTime:
                    lData[k][:].astype(mydtype).tofile(fid)
                else:
                    for i, idt in enumerate(list(dictTime.values())):
                        lData[k][idt, :].astype(mydtype).tofile(fid)
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
    backend="hdf5",
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
    from warnings import warn

    warn("write_seissol_output is deprecated. Please use write instead")
    nNodes = xyz.shape[0]
    nCells, node_per_element = connect.shape
    dictData = {}
    for name, value in zip(lDataName, lData):
        dictData[name] = value
    dictTime = {}
    if dt:
        for index in lidt:
            dictTime[dt * index] = index

    if dt == 0:
        lTime = [-1.0]
    write(prefix, xyz, connect, dictData, dictTime, reduce_precision, backend)


def write(
    prefix,
    xyz,
    connect,
    dictData,
    dictTime,
    reduce_precision=False,
    backend="hdf5",
    compression_level=4,
):
    """
    Write hdf5/xdmf files output, readable by ParaView using SeisSol data
    prefix: file
    xyz: geometry array
    connect: connect array
    dictData: dictionnary with dataname as keys and numpy arrays as values
    dictTime: dictionnary with time values as keys and indices as values.
               for writing a puml mesh use an empty dictionnary
    reduce_precision: convert double to float and i64 to i32 if True
    backend: data format ("hdf5" or "raw")
    """
    nNodes = xyz.shape[0]
    nCells, node_per_element = connect.shape
    if backend not in ("hdf5", "raw"):
        raise ValueError("Invalid backend. Must be 'hdf5' or 'raw'.")
    if compression_level < 0 or compression_level > 9:
        raise ValueError("compression_level has to be in 0-9")

    # check that dictTime is compatible with dataArray
    if dictTime:
        max_index = max(dictTime.values())
    for dataName, dataArray in dictData.items():
        lenArray = 1 if len(dataArray.shape) == 1 else dataArray.shape[0]
        if dictTime:
            assert (
                max_index < lenArray
            ), f"max index of dictTime ({max_index}) larger than array size ({lenArray}) for array {dataName}"
        else:
            assert (
                lenArray == 1
            ), f"array size ({lenArray}) for array {dataName} is not 1 and dictTime is empty"

    if not dictTime:
        write_mesh_xdmf(
            prefix,
            xyz,
            connect,
            dictData,
            reduce_precision,
            backend,
        )
    else:
        write_timeseries_xdmf(
            prefix,
            xyz,
            connect,
            dictData,
            dictTime,
            reduce_precision,
            backend,
        )
    write_data(
        prefix,
        xyz,
        connect,
        dictData,
        dictTime,
        reduce_precision,
        backend,
        compression_level,
    )
