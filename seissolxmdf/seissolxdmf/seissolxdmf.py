import numpy as np
import os
import lxml.etree as ET


def ReadHdf5DatasetChunk(absolute_path, hdf5var, firstElement, nchunk, idt=-1):
    """ Read block of data in hdf5 format
    idt!=-1 loads only one time step """
    import h5py

    lastElement = firstElement + nchunk

    oneDtMem = True if idt != -1 else False
    h5f = h5py.File(absolute_path, "r")
    if h5f[hdf5var].ndim == 2:
        if oneDtMem:
            myData = h5f[hdf5var][idt, firstElement:lastElement]
        else:
            myData = h5f[hdf5var][:, firstElement:lastElement]
    else:
        myData = h5f[hdf5var][firstElement:lastElement]
    h5f.close()
    return myData


def GetDtype(data_prec, isInt):
    if data_prec == 4:
        if isInt:
            return np.dtype("i4")
        else:
            return np.dtype("<f")
    else:
        if isInt:
            return np.dtype("i8")
        else:
            return np.dtype("d")


def ReadSimpleBinaryFile(absolute_path, MemDimension, data_prec, isInt, idt=-1):
    """Read block of data in binary format (posix)
    idt!=-1 loads only one time step 
    this function is a special case of ReadSimpleBinaryFileChunk
    but kept for performance reasons """
    oneDtMem = True if idt != -1 else False
    data_type = GetDtype(data_prec, isInt)

    fid = open(absolute_path, "r")
    if oneDtMem:
        fid.seek(idt * MemDimension * data_prec, os.SEEK_SET)
        myData = np.fromfile(fid, dtype=data_type, count=MemDimension)
    else:
        myData = np.fromfile(fid, dtype=data_type)
        ndt = np.shape(myData)[0] // MemDimension
        myData = myData.reshape((ndt, MemDimension))
    fid.close()
    return myData


def ReadSimpleBinaryFileChunk(absolute_path, MemDimension, data_prec, isInt, ndt, firstElement, nchunk, idt=-1):
    """Read block of data in binary format (posix)
    same as ReadSimpleBinaryFile: but reads a subset of the second dimension
    idt!=-1 loads only one time step """
    oneDtMem = True if idt != -1 else False
    data_type = GetDtype(data_prec, isInt)

    fid = open(absolute_path, "r")
    if oneDtMem:
        assert idt < ndt
        fid.seek((idt * MemDimension + firstElement) * data_prec, os.SEEK_SET)
        myData = np.fromfile(fid, dtype=data_type, count=nchunk)
    else:
        myData = np.zeros((ndt, nchunk))
        for idt in range(0, ndt):
            fid.seek((idt * MemDimension + firstElement) * data_prec, os.SEEK_SET)
            myData[idt, :] = np.fromfile(fid, dtype=data_type, count=nchunk)
    fid.close()
    return myData


def GetDataLocationPrecisionNElementsMemDimension(xdmfFilename, attribute):
    """ Common function called by ReadTopologyOrGeometry """
    tree = ET.parse(xdmfFilename)
    root = tree.getroot()
    for Property in root.findall(".//%s" % (attribute)):
        nElements = int(Property.get("NumberOfElements"))
        break
    for Property in root.findall(".//%s/DataItem" % (attribute)):
        dataLocation = Property.text
        data_prec = int(Property.get("Precision"))
        MemDimension = [int(val) for val in Property.get("Dimensions").split()]
        break
    return [dataLocation, data_prec, nElements, MemDimension]


def GetDataLocationPrecisionMemDimension(xdmfFilename, dataName):
    """ Common function called by ReadData """

    def get(prop):
        dataLocation = prop.text
        data_prec = int(prop.get("Precision"))
        dims = prop.get("Dimensions").split()
        if len(dims) == 1:
            MemDimension = int(prop.get("Dimensions").split()[0])
        else:
            MemDimension = int(prop.get("Dimensions").split()[1])
        return [dataLocation, data_prec, MemDimension]

    tree = ET.parse(xdmfFilename)
    root = tree.getroot()
    for Property in root.findall(".//Attribute"):
        if Property.get("Name") == dataName:
            for prop in Property.findall(".//DataItem"):
                if prop.get("Format") in ["HDF", "Binary"]:
                    return get(prop)
                path = prop.get("Reference")
                if path is not None:
                    ref = tree.xpath(path)[0]
                    return get(ref)
    raise NameError("%s not found in dataset" % (dataName))


def ReadTopologyOrGeometry(xdmfFilename, attribute):
    """ Common function to read either connect or geometry """
    path = os.path.join(os.path.dirname(xdmfFilename), "")
    dataLocation, data_prec, nElements, MemDimension = GetDataLocationPrecisionNElementsMemDimension(xdmfFilename, attribute)
    # 3 for surface, 4 for volume
    dim2 = MemDimension[1]
    splitArgs = dataLocation.split(":")
    isHdf5 = True if len(splitArgs) == 2 else False
    isInt = True if attribute == "Topology" else False
    if isHdf5:
        filename, hdf5var = splitArgs
        myData = ReadHdf5DatasetChunk(path + filename, hdf5var, 0, dim2)
    else:
        myData = ReadSimpleBinaryFile(path + dataLocation, dim2, data_prec, isInt)
        # due to zero padding in SeisSol for memory alignement, the array read may be larger than the actual data
        myData = myData[0:nElements, :]
    return myData


def ReadConnect(xdmfFilename):
    """ Read the connectivity matrice defining the cells """
    return ReadTopologyOrGeometry(xdmfFilename, "Topology")


def ReadGeometry(xdmfFilename):
    """ Read the connectivity matrice defining the cells """
    return ReadTopologyOrGeometry(xdmfFilename, "Geometry")


def ReadNdt(xdmfFilename):
    """ read number of time steps in the file """
    tree = ET.parse(xdmfFilename)
    root = tree.getroot()
    ndt = 0
    for Property in root.findall(".//Grid"):
        if Property.get("GridType") == "Uniform":
            ndt = ndt + 1
    if ndt == 0:
        raise NameError("ndt=0,( no GridType=Uniform found in xdmf)")
    else:
        return ndt


def ReadNElements(xdmfFilename):
    """ read number of cell elements of the mesh """
    tree = ET.parse(xdmfFilename)
    root = tree.getroot()
    for Property in root.findall("Domain/Grid/Grid/Topology"):
        path = Property.get("Reference")
        if path is None:
            return int(Property.get("NumberOfElements"))
        else:
            ref = tree.xpath(path)[0]
            return int(ref.get("NumberOfElements"))
    raise NameError("nElements could not be determined")


def ReadTimeStep(xdmfFilename):
    """ reading the time step (dt) in the xdmf file """
    tree = ET.parse(xdmfFilename)
    root = tree.getroot()
    i = 0
    for Property in root.findall("Domain/Grid/Grid/Time"):
        if i == 0:
            dt = float(Property.get("Value"))
            i = 1
        else:
            dt = float(Property.get("Value")) - dt
            return dt
    raise NameError("time step could not be determined")


def Read1dData(xdmfFilename, dataName, nElements, isInt=False):
    """ Read 1 dimension array (used by ReadPartition) """
    path = os.path.join(os.path.dirname(xdmfFilename), "")
    dataLocation, data_prec, MemDimension = GetDataLocationPrecisionMemDimension(xdmfFilename, dataName)
    splitArgs = dataLocation.split(":")
    isHdf5 = True if len(splitArgs) == 2 else False
    if isHdf5:
        filename, hdf5var = splitArgs
        myData = ReadHdf5DatasetChunk(path + filename, hdf5var, 0, MemDimension)
    else:
        filename = dataLocation
        myData = ReadSimpleBinaryFile(path + dataLocation, nElements, data_prec, isInt)
    return myData


def ReadPartition(xdmfFilename):
    """ Read partition array """
    nElements = ReadNElements(xdmfFilename)
    partition = Read1dData(xdmfFilename, "partition", nElements, isInt=True)
    return partition


def ReadData(xdmfFilename, dataName, idt=-1):
    """ Load a data array named 'dataName' (e.g. SRs)
    if idt!=-1, only the time step idt is loaded
    else all time steps are loaded   """

    nElements = ReadNElements(xdmfFilename)
    return ReadDataChunk(xdmfFilename, dataName, firstElement=0, nchunk=nElements, idt=idt)


def ReadDataChunk(xdmfFilename, dataName, firstElement, nchunk, idt=-1):
    """ Load a chunk of a data array named 'dataName' (e.g. SRs)
    That is instead of loading 0:nElements, load firstElement:firstElement+nchunk
    This function is used for generating in parallel Ground motion estimate maps
    if idt!=-1, only the time step idt is loaded
    else all time steps are loaded   """
    path = os.path.join(os.path.dirname(xdmfFilename), "")
    dataLocation, data_prec, MemDimension = GetDataLocationPrecisionMemDimension(xdmfFilename, dataName)
    splitArgs = dataLocation.split(":")
    isHdf5 = True if len(splitArgs) == 2 else False
    oneDtMem = True if idt != -1 else False
    if isHdf5:
        filename, hdf5var = splitArgs
        myData = ReadHdf5DatasetChunk(path + filename, hdf5var, firstElement, nchunk, idt)
    else:
        ndt = ReadNdt(xdmfFilename)
        myData = ReadSimpleBinaryFileChunk(path + dataLocation, MemDimension, data_prec, isInt=False, ndt=ndt, firstElement=firstElement, nchunk=nchunk, idt=idt)
    return myData


def LoadData(xdmfFilename, dataName, nElements, idt=0, oneDtMem=False, firstElement=-1):
    """ Do the same as ReadDataChunk. here for backward compatibility """
    dataLocation, data_prec, MemDimension = GetDataLocationPrecisionMemDimension(xdmfFilename, dataName)
    if not oneDtMem:
        idt = -1
    if firstElement == -1:
        firstElement = 0
    myData = ReadDataChunk(xdmfFilename, dataName, firstElement, nElements, idt)
    return [myData, data_prec]
