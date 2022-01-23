![testing](https://github.com/sstucker/pyscanpatterns/actions/workflows/test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/scanpatterns.svg)](https://badge.fury.io/py/scanpatterns)

# PyScanPatterns
Parametric scan patterns for laser scanning microscopy.

Galvo drive signals, line camera/PMT exposure triggers, and frame/digitizer triggers are generated given a scan type and parameters.

## Installation
```
pip install scanpatterns
```

## Patterns

So far, the following patterns are possible:
* Raster scan
* Raster scan with repeated A-lines
* Raster scan with repeated B-lines
* Bidirectional raster scan
* Figure-8 scan
* Rhodonea rose scan with any number of petals

Patterns are created by calling the `generate` method of any `LineScanPattern` instance.

The resultant scan signals can be accessed via the `x`, `y`, `line_trigger` and `frame_trigger` properties.

![Classic raster](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/raster.png)
```python
RasterScanPattern(16, 16, 1, samples_on=1, samples_off=10)
```

![Stepped raster](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rasterstep.png)
```python
RasterScanPattern(16, 16, 1, samples_on=1, samples_off=10, fast_axis_step=True, slow_axis_step=True)
```

![B-line repeated raster](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rasterrpt.png)
```python
RasterScanPattern(16, 16, 1, samples_on=1, samples_off=10, bline_repeat=2)
```

![Rectangular raster](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rectraster.png)
```python
RasterScanPattern(16, 16, 1, samples_on=1, samples_off=10, fov=[1.5, 4.5], fast_axis_step=True, slow_axis_step=True)
```

![Rotated raster](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rotraster.png)
```python
RasterScanPattern(16, 16, 1, samples_on=1, samples_off=10, fast_axis_step=True, slow_axis_step=True, rotation_rad=np.pi/4)
```

![Bidirectional raster](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/biraster.png)
```python
RasterScanPattern(15, 15, 1, samples_on=1, samples_off=10, bidirectional=True, slow_axis_step=True)
```

![Sample figure-8 scan](https://github.com/sstucker/PyScanPattern/blob/master/img/fig8.png)
```python
Figure8ScanPattern(1.0, 16, 76000)
```

![Rose p=3](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rose3.png)
```python
RoseScanPattern(3, 1, 16, 1, samples_on=1, samples_off=10)
```

![Rose p=5](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rose5.png)
```python
RoseScanPattern(5, 1, 16, 1, samples_on=1, samples_off=10)
```
