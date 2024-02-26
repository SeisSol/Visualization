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
```

