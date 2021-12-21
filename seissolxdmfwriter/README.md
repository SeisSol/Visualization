seissolxdmfwriter
===============
Functions to write seissol outputs readable by paraview.

```python
import seissolxdmfwriter as sxw
import seissolxdmf as sx
fn = 'test-fault.xdmf'
# Read data from input file using seissolxdmf
sx = sx.seissolxdmf(fn)
geom = sx.ReadGeometry()
connect = sx.ReadConnect()
dt = sx.ReadTimeStep()
SRs = sx.ReadData('SRs')
SRd = sx.ReadData('SRs')
SR = np.sqrt(SRs**2 + SRd**2)

# Write the 0,4 and 8th times steps of array SRs and SR in SRtest-fault.xdmf/SRtest-fault.h5
sxw.write_seissol_output('test-fault', geom, connect, ['SRs', 'SR'], [SRs, SR], dt, [0, 4, 8])
```
