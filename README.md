![testing](https://github.com/sstucker/pyscanpatterns/actions/workflows/test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/scanpatterns.svg)](https://badge.fury.io/py/scanpatterns)

# PyScanPatterns
Parametric scan patterns for imaging applications.

Galvo drive signals, line camera/PMT exposure triggers, and frame/digitizer triggers are generated given a scan type and parameters. The exposure pulse is defined by the `samples_on` and `samples_off` parameters and this period (`samples_on + samples_off`) is used to create a pattern with a fixed number of exposures for a given `max_line_rate`. 

Used for an SD-OCT imaging system, but should work for ultrasound and laser scanning microscopy applications.

## Installation
```
pip install scanpatterns
```

## Features

The following patterns are possible:
* Raster scan
* Line scan
* Raster scan with repeated A-lines
* Raster scan with repeated B-lines
* Bidirectional raster scan
* Circle scans
* Figure-8 scan (modified lissajous)
* Rhodonea rose scan with odd number of petals

Patterns are created by calling the `generate` method of any `LineScanPattern` instance or by passing
the arguments of `generate` to the constructor.

The resultant scan signals can be accessed via the `x`, `y`, `line_trigger` and `frame_trigger` properties.

Other properties of a pattern such as `pattern_rate`, `frame_rate`, and the arguments used to create the pattern
are also available.

## Gallery

![Classic raster](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/raster.png)
```python
RasterScanPattern(16, 16, 76000, samples_on=1, samples_off=10)
```

![Stepped raster](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/rasterstep.png)
```python
RasterScanPattern(16, 16, 76000, samples_on=1, samples_off=10, fast_axis_step=True, slow_axis_step=True)
```

![B-line repeated raster](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/rasterrpt.png)
```python
RasterScanPattern(16, 16, 76000, samples_on=1, samples_off=10, bline_repeat=2)
```

![Rectangular raster](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/rectraster.png)
```python
RasterScanPattern(16, 16, 76000, fov=[1.5, 4.5], samples_on=1, samples_off=10, fast_axis_step=True, slow_axis_step=True)
```

![Rotated raster](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/rotraster.png)
```python
RasterScanPattern(16, 16, 76000, samples_on=1, samples_off=10, fast_axis_step=True, slow_axis_step=True, rotation_rad=np.pi/4)
```

![Bidirectional raster](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/biraster.png)
```python
RasterScanPattern(15, 15, 76000, samples_on=1, samples_off=10, bidirectional=True, slow_axis_step=True)
```

![Line scan](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/line.png)
```python
RasterScanPattern(15, 1, 76000, samples_on=1, samples_off=10)
```

![Bidirectional line](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/biline.png)
```python
RasterScanPattern(15, 1, 76000, samples_on=1, samples_off=10, bidirectional=True, rotation_rad=np.pi/8)
```

![Circle](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/circle.png)
```python
CircleScanPattern(64, 1.0, 76000, samples_on=1)
```

![Sample figure-8 scan](https://github.com/sstucker/PyScanPattern/blob/master/img/fig8.png)
```python
Figure8ScanPattern(1.0, 16, 76000)
```

![Rose p=3](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/rose3.png)
```python
RoseScanPattern(3, 1, 16, 76000, samples_on=1, samples_off=10)
```

![Rose p=5](https://raw.githubusercontent.com/sstucker/pyscanpatterns/master/img/rose5.png)
```python
RoseScanPattern(5, 1, 16, 76000, samples_on=1, samples_off=10)
```
