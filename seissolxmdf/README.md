seissolxdmf
===============
A python reader for SeisSol xdmf output (posix or hdf5) and hdf5 meshes.
Below is an simple example, illustrating the use of the module:
```python
import seissolxdmf as sx
fn = 'test-fault.xdmf'
# Number of cells
nElements = sx.ReadNElements(fn)
# Read time step
dt = sx.ReadTimeStep(fn)
# Read number of time steps
ndt = sx.ReadNdt(fn)
# load geometry array as a numpy array of shape ((nodes, 3))
geom = sx.ReadGeometry(fn)
# load connectivity array as a numpy array of shape ((nElements, 3 or 4))
# The connectivity array gives for each cell a list of vertex ids.
connect = sx.ReadConnect(fn)
# load SRs as a numpy array of shape ((ndt, nElements))
SRs = sx.ReadData(fn, 'SRs')
# load the 8th time ste of the SRs array as a numpy array of shape (nElements)
SRs = sx.ReadData(fn, 'SRs', 8)
```
