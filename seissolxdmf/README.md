seissolxdmf
===============
A python reader for SeisSol xdmf output (posix or hdf5) and hdf5 meshes.
Below is an simple example, illustrating the use of the module:
```python
import seissolxdmf
fn = 'test-fault.xdmf'
# initiate class
sx = seissolxdmf.seissolxdmf(fn)
# Number of cells
nElements = sx.ReadNElements()
# Number of vertices
nNodes = sx.ReadNNodes()
# Number of nodes per elements
nodesPerElement = sx.ReadNodesPerElement()
# Read time step
dt = sx.ReadTimeStep()
# Read number of time steps
ndt = sx.ReadNdt()
# Returns a list of the output time values
outputTimes = sx.ReadTimes()
# load geometry array as a numpy array of shape ((nodes, 3))
geom = sx.ReadGeometry()
# load connectivity array as a numpy array of shape ((nElements, 3 or 4))
# The connectivity array gives for each cell a list of vertex ids.
connect = sx.ReadConnect()
# Check, whether variable "SRs" exists in the SeisSol output
assert "SRs" in sx.ReadAvailableDataFields()
# load SRs as a numpy array of shape ((ndt, nElements))
SRs = sx.ReadData('SRs')
# load the 9th time step of the SRs array as a numpy array of shape (nElements)
SRs = sx.ReadData('SRs', 8)
```
