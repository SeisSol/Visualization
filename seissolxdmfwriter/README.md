seissolxdmfwriter
=================
A python module to write seissol outputs readable by paraview.

```python
import seissolxdmfwriter as sxw
import seissolxdmf as sx
import numpy as np

fn = "test-fault.xdmf"
# Read data from input file using seissolxdmf
sx = sx.seissolxdmf(fn)
geom = sx.ReadGeometry()
connect = sx.ReadConnect()
dt = sx.ReadTimeStep()
outputTimes = sx.ReadTimes()

SRs = sx.ReadData("SRs")
SRd = sx.ReadData("SRd")
SR = np.sqrt(SRs**2 + SRd**2)

# Write the 0,4 and 8th times steps of array SRs and SR in SRtest-fault.xdmf/SRtest-fault.h5
dictTime = {outputTimes[i]: i for i in [0, 4, 8]}
sxw.write(
    "test-fault",
    geom,
    connect,
    {"SRs": SRs, "SR": SR},
    dictTime,
    reduce_precision=True,
    backend="hdf5",
)

# Finally, the module can be use to write data directly from seissolxdmf, limiting
# the memory requirements

sxw.write_from_seissol_output(
    'test-fault-sx',
    sx,
    ['SRs', 'SRd','fault-tag', 'partition'],
    [3,4],
    reduce_precision=True,
    backend="hdf5",
    compression_level=4,
)

```

The module also encapsulates `seissol_output_extractor`, which can be used to extract and process data from SeisSol output files, allowing selection of variables, time steps, spatial ranges, and output format.
Here is an example of use:

```bash
# extracts PSR, Vr and partition, at 2nd and 4th time steps and at simulation time 0.5, from test-fault.xdmf and write into test_new-fault.xdmf
# use seissol_output_extractor --h for additionnal info about the arguments
seissol_output_extractor test-fault.xdmf --time "i2,i4,0.5" --variable PSR Vr partition --add2prefix "_new"
```
