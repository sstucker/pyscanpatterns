# PyScanPattern
Parametric scan patterns for laser scanning microscopy.

Galvo drive signals, line camera/PMT exposure triggers, and frame/digitizer triggers are generated given a scan type and parameters.

## Installation
```
pip install scanpatterns
```

## Patterns

So far, the following patterns are implemented:
* Raster scan
* Raster scan with repeated A-lines
* Raster scan with repeated B-lines
* Figure-8 scan
* Rhodonea rose scan with any number of petals

Patterns are created by calling the `generate` method of any `LineScanPattern` instance.

The resultant scan signals can be accessed via the `x`, `y`, `line_trigger` and `frame_trigger` properties.

![Sample raster scan](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/raster.png)
```python
RasterScanPattern(16, 16,76000)
```

![Sample figure-8 scan](https://github.com/sstucker/PyScanPattern/blob/master/img/fig8.png)
```python
Figure8ScanPattern(1.0, 16, 76000)
```

![Sample rose scan](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rose3.png)
```python
Figure8ScanPattern(3, 1.0, 16, 76000)
```

![Sample rose scan](https://raw.githubusercontent.com/sstucker/PyScanPattern/master/img/rose5.png)
```python
Figure8ScanPattern(5, 1.0, 16, 76000)
```