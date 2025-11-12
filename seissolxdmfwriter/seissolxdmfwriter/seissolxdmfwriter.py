import numpy as np
import os
from tqdm import tqdm
import sys

known_1d_arrays = ["locationFlag", "fault-tag", "partition", "clustering", "global-id"]


def dataLocation(prefix, name, backend):
    if backend == "hdf5":
        colon_or_nothing = ".h5:"
        ext = ""
    else:
        colon_or_nothing = ""
        ext = ".bin"
    return f"{prefix}{colon_or_nothing}/{name}{ext}"


def compile_dictDataTypes(dictData, reduce_precision):
    precisionDict = {"int64": 8, "int32": 4, "float64": 8, "float32": 4}
    numberTypeDict = {
        "int64": "UInt",
        "int32": "UInt",
        "uint32": "UInt",
        "uint64": "UInt",
        "float64": "Float",
        "float32": "Float",
    }
    dictDataTypes = {}
    for dataName, dataArray in dictData.items():
        prec = 4 if reduce_precision else precisionDict[dataArray.dtype.name]
        number_type = numberTypeDict[dataArray.dtype.name]
        dictDataTypes[dataName] = (prec, number_type)
    return dictDataTypes


def write_timeseries_xdmf(
    prefix,
    nNodes,
    nCells,
    node_per_element,
    dictDataTypes,
    timeValues,
    reduce_precision,
    backend,
):
    bn_prefix = os.path.basename(prefix)
    data_format = "HDF" if backend == "hdf5" else "Binary"
    lDataName = list(dictDataTypes.keys())
    lDataTypes = list(dictDataTypes.values())
    topology = "Tetrahedron" if node_per_element == 4 else "Triangle"
    xdmf = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">"""
    geometry_location = dataLocation(bn_prefix, "geometry", backend)
    connect_location = dataLocation(bn_prefix, "connect", backend)

    for i, ctime in enumerate(timeValues):
        xdmf += f"""
   <Grid Name="step_{i}" GridType="Uniform">
    <Topology TopologyType="{topology}" NumberOfElements="{nCells}">
     <DataItem NumberType="Int" Precision="8" Format="{data_format}" Dimensions="{nCells} {node_per_element}">{connect_location}</DataItem>
    </Topology>
    <Geometry name="geo" GeometryType="XYZ" NumberOfElements="{nNodes}">
     <DataItem NumberType="Float" Precision="8" Format="{data_format}" Dimensions="{nNodes} 3">{geometry_location}</DataItem>
    </Geometry>
    <Time Value="{ctime}"/>"""
        for k, dataName in enumerate(list(dictDataTypes.keys())):
            data_location = dataLocation(bn_prefix, dataName, backend)
            prec, number_type = dictDataTypes[dataName]
            if dataName in known_1d_arrays:
                xdmf += f"""
    <Attribute Name="{dataName}" Center="Cell">
     <DataItem NumberType="{number_type}" Precision="{prec}" Format="{data_format}" Dimensions="1 {nCells}">{data_location}</DataItem>
    </Attribute>"""
            else:
                xdmf += f"""
    <Attribute Name="{dataName}" Center="Cell">
     <DataItem ItemType="HyperSlab" Dimensions="{nCells}">
      <DataItem NumberType="UInt" Precision="4" Format="XML" Dimensions="3 2">{i} 0 1 1 1 {nCells}</DataItem>
      <DataItem NumberType="{number_type}" Precision="{prec}" Format="{data_format}" Dimensions="{i+1} {nCells}">{data_location}</DataItem>
     </DataItem>
    </Attribute>"""
        xdmf += """
   </Grid>"""
    xdmf += """
  </Grid>
 </Domain>
</Xdmf>
"""
    with open(prefix + ".xdmf", "w") as fid:
        fid.write(xdmf)
    print(f"done writing {prefix}.xdmf")
    full_path = os.path.abspath(f"{prefix}.xdmf")
    print(f"full path: {full_path}")


def write_mesh_xdmf(
    prefix,
    nNodes,
    nCells,
    node_per_element,
    dictDataTypes,
    reduce_precision,
    backend,
):
    bn_prefix = os.path.basename(prefix)
    data_format = "HDF" if backend == "hdf5" else "Binary"
    lDataName = list(dictDataTypes.keys())
    lDataTypes = list(dictDataTypes.values())
    topology = "Tetrahedron" if node_per_element == 4 else "Triangle"
    xdmf = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>"""
    geometry_location = dataLocation(bn_prefix, "geometry", backend)
    connect_location = dataLocation(bn_prefix, "connect", backend)

    xdmf += f"""
  <Grid Name="puml mesh" GridType="Uniform">
    <Topology TopologyType="{topology}" NumberOfElements="{nCells}">
     <DataItem NumberType="Int" Precision="8" Format="{data_format}" Dimensions="{nCells} {node_per_element}">{connect_location}</DataItem>
    </Topology>
    <Geometry name="geo" GeometryType="XYZ" NumberOfElements="{nNodes}">
     <DataItem NumberType="Float" Precision="8" Format="{data_format}" Dimensions="{nNodes} 3">{geometry_location}</DataItem>
    </Geometry>"""
    for k, dataName in enumerate(list(dictDataTypes.keys())):
        data_location = dataLocation(bn_prefix, dataName, backend)
        prec, number_type = dictDataTypes[dataName]
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
    full_path = os.path.abspath(f"{prefix}.xdmf")
    print(f"full path: {full_path}")


def output_type(input_array, reduce_precision):
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
    mydtype = dtypeDict[input_array.dtype.name]
    if reduce_precision:
        mydtype = reducePrecisionDict[mydtype]
    return mydtype


def write_one_arr_hdf5(h5f, ar_name, ar_data, compression_options):
    h5f.create_dataset(
        f"/{ar_name}", ar_data.shape, dtype=ar_data.dtype, **compression_options
    )
    if len(ar_data.shape) == 1:
        h5f[f"/{ar_name}"][:] = ar_data
    else:
        h5f[f"/{ar_name}"][:, :] = ar_data
    return ar_data.shape


def infer_n_elements(sx, filtered_cells):
    if isinstance(filtered_cells, slice) and filtered_cells == slice(None):
        return sx.ReadNElements()
    else:
        return len(filtered_cells)


def write_data_from_seissolxdmf(
    prefix,
    sx,
    non_temporal_array_names,
    array_names,
    time_indices,
    reduce_precision,
    backend,
    compression_level,
    filtered_cells,
):
    def read_non_temporal(sx, ar_name, filtered_cells):
        if ar_name == "geometry":
            return sx.ReadGeometry()
        elif ar_name == "connect":
            return sx.ReadConnect()[filtered_cells, :]
        else:
            return sx.Read1dData(ar_name, sx.nElements, isInt=True)[filtered_cells]

    nel = infer_n_elements(sx, filtered_cells)
    if backend == "hdf5":
        import h5py

        compression_options = {}
        if compression_level:
            compression_options = {
                "compression": "gzip",
                "compression_opts": compression_level,
            }

        with h5py.File(prefix + ".h5", "w") as h5f:
            for ar_name in non_temporal_array_names:
                my_array = read_non_temporal(sx, ar_name, filtered_cells)
                write_one_arr_hdf5(h5f, ar_name, my_array, compression_options)
            for ar_name in array_names:
                for i, idt in enumerate(
                    tqdm(
                        time_indices,
                        file=sys.stdout,
                        desc=ar_name,
                        dynamic_ncols=False,
                    )
                ):
                    try:
                        my_array = sx.ReadData(ar_name, idt)[filtered_cells]
                    except IndexError:
                        print(
                            f"time step {idt} of {ar_name} is corrupted, replacing with nans"
                        )
                        my_array = np.full(nel, np.nan)
                    if i == 0:
                        h5f.create_dataset(
                            f"/{ar_name}",
                            (len(time_indices), nel),
                            dtype=str(output_type(my_array, reduce_precision)),
                            **compression_options,
                        )
                    if my_array.shape[0] == 0:
                        print(
                            f"time step {idt} of {ar_name} is corrupted, replacing with nans"
                        )
                        my_array = np.full(nel, np.nan)
                    h5f[f"/{ar_name}"][i, :] = my_array[:]
        print(f"done writing {prefix}.h5")
    else:
        os.makedirs(prefix, exist_ok=True)
        for ar_name in non_temporal_array_names:
            my_array = read_non_temporal(sx, ar_name, filtered_cells)
            with open(f"{prefix}/{ar_name}.bin", "wb") as fid:
                my_array.tofile(fid)
        for ar_name in array_names:
            with open(f"{prefix}/{ar_name}.bin", "wb") as fid:
                for i, idt in enumerate(
                    tqdm(
                        time_indices,
                        file=sys.stdout,
                        desc=ar_name,
                        dynamic_ncols=False,
                    )
                ):
                    my_array = sx.ReadData(ar_name, idt)[filtered_cells]
                    if my_array.shape[0] == 0:
                        print(
                            f"time step {idt} of {ar_name} is corrupted, replacing with nans"
                        )
                        my_array = np.full(nel, np.nan)
                    if i == 0:
                        mydtype = output_type(my_array, reduce_precision)
                    my_array[:].astype(mydtype).tofile(fid)
        print(f"done writing binary files in {prefix}")


def write_data(
    prefix,
    dicDataNonTemporal,
    dictData,
    dictTime,
    reduce_precision,
    backend,
    compression_level,
):
    if dictTime:
        time_indices = list(dictTime.values())
    else:
        time_indices = [0]
    if backend == "hdf5":
        import h5py

        compression_options = {}
        if compression_level:
            compression_options = {
                "compression": "gzip",
                "compression_opts": compression_level,
            }

        with h5py.File(prefix + ".h5", "w") as h5f:
            for ar_name, my_array in dicDataNonTemporal.items():
                write_one_arr_hdf5(h5f, ar_name, my_array, compression_options)
            for ar_name, my_array in dictData.items():
                if len(my_array.shape) == 1:
                    my_array = my_array[np.newaxis, :]
                for i, idt in enumerate(time_indices):
                    if i == 0:
                        h5f.create_dataset(
                            f"/{ar_name}",
                            (len(time_indices), my_array.shape[1]),
                            dtype=str(output_type(my_array, reduce_precision)),
                            **compression_options,
                        )
                    h5f[f"/{ar_name}"][i, :] = my_array[idt, :]
        print(f"done writing {prefix}.h5")
    else:
        os.makedirs(prefix, exist_ok=True)
        for ar_name, my_array in dicDataNonTemporal.items():
            with open(f"{prefix}/{ar_name}.bin", "wb") as fid:
                my_array.tofile(fid)
        for ar_name, my_array in dictData.items():
            if len(my_array.shape) == 1:
                my_array = my_array[np.newaxis, :]
            mydtype = output_type(my_array, reduce_precision)
            with open(f"{prefix}/{ar_name}.bin", "wb") as fid:
                if not dictTime:
                    my_array[:].astype(mydtype).tofile(fid)
                else:
                    for i, idt in enumerate(time_indices):
                        my_array[idt, :].astype(mydtype).tofile(fid)
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

    dictDataTypes = compile_dictDataTypes(dictData, reduce_precision)

    dicDataNonTemporal = {"geometry": xyz, "connect": connect}
    to_move = [name for name in dictData.keys() if name in known_1d_arrays]

    for name in to_move:
        dicDataNonTemporal[name] = dictData.pop(name)

    if backend not in ("hdf5", "raw"):
        raise ValueError(f"Invalid backend {backend}. Must be 'hdf5' or 'raw'.")
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
            nNodes,
            nCells,
            node_per_element,
            dictDataTypes,
            reduce_precision,
            backend,
        )
    else:
        write_timeseries_xdmf(
            prefix,
            nNodes,
            nCells,
            node_per_element,
            dictDataTypes,
            dictTime.keys(),
            reduce_precision,
            backend,
        )

    write_data(
        prefix,
        dicDataNonTemporal,
        dictData,
        dictTime,
        reduce_precision,
        backend,
        compression_level,
    )


def write_from_seissol_output(
    prefix,
    sx,
    var_names,
    time_indices,
    reduce_precision=False,
    backend="hdf5",
    compression_level=4,
    filtered_cells=slice(None),
):
    """
    Write hdf5/xdmf files output, readable by ParaView from a seissolxdmf object
    prefix: file
    sx: seissolxdmf object
    var_names: list of variables to extract
    time_indices: list of times indices to extract
    reduce_precision: convert double to float and i64 to i32 if True
    backend: data format ("hdf5" or "raw")
    """
    if backend not in ("hdf5", "raw"):
        raise ValueError(f"Invalid backend {backend}. Must be 'hdf5' or 'raw'.")
    if compression_level < 0 or compression_level > 9:
        raise ValueError("compression_level has to be in 0-9")

    non_temporal_array_names = ["geometry", "connect"]
    to_move = [name for name in var_names if name in known_1d_arrays]
    dictDataTypes = {}
    for name in to_move:
        var_names.remove(name)
        non_temporal_array_names.append(name)
        dictDataTypes[name] = (4, "UInt")

    for name in var_names:
        (
            dataLocation,
            data_prec,
            MemDimension,
        ) = sx.GetDataLocationPrecisionMemDimension(name)
        data_prec = 4 if reduce_precision else data_prec
        dictDataTypes[name] = (data_prec, "Float")

    write_data_from_seissolxdmf(
        prefix,
        sx,
        non_temporal_array_names,
        var_names,
        time_indices,
        reduce_precision,
        backend,
        compression_level,
        filtered_cells,
    )

    nel = infer_n_elements(sx, filtered_cells)
    write_timeseries_xdmf(
        prefix,
        sx.ReadNNodes(),
        nel,
        sx.ReadNodesPerElement(),
        dictDataTypes,
        [sx.ReadTimes()[k] for k in time_indices],
        reduce_precision,
        backend,
    )
