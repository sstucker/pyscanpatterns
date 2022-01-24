import unittest
from scanpatterns import *  

def _measure_fov(raster: RasterScanPattern):
    points = np.array([raster.x[raster.line_trigger.astype(bool)],
                       raster.y[raster.line_trigger.astype(bool)]])
    max_x = np.argmax(points[0, :])
    min_x = np.argmin(points[0, :])
    max_y = np.argmax(points[1, :])
    min_y = np.argmin(points[1, :])
    x = points[0, max_x] - points[0, min_x]
    y = points[1, max_y] - points[1, min_y]
    return (x, y)
    

def _print_line_scan_pattern(pattern: LineScanPattern):
    print('Pattern type:', pattern.__class__.__name__)
    print('Number of samples:', len(pattern.x))
    print('Number of A-lines:', pattern.total_number_of_alines)
    print('Dimensions:', pattern.dimensions)
    print('Sample rate:', pattern.sample_rate, 'Hz')
    print('Pattern rate:', pattern.pattern_rate, 'Hz')


def _print_raster_pattern(raster: RasterScanPattern):
    _print_line_scan_pattern(raster)
    print('Ideal FOV:', raster.fov)
    print('Actual FOV:', _measure_fov(raster))
    print('Ideal exposure fraction:', raster.exposure_fraction)
    print('Actual exposure fraction:', raster.true_exposure_fraction)
    print('Flyback duty:', raster.flyback_duty)
    print('Fast-axis step:', raster.fast_axis_step)
    print('Slow-axis step:', raster.slow_axis_step)
    print('Bidirectional:', raster.bidirectional)


def _test_signal_len(test: unittest.TestCase, pat: LineScanPattern):
    test.assertTrue(all([len(pat.x) == len(l) for l in [pat.x, pat.y, pat.line_trigger, pat.frame_trigger]]),
                    msg='Scan signals of ' + pat.__class__.__name__ + ' vary in length! Lengths: '+ str(len(pat.x))
                    + ' ' + str(len(pat.y)) + ' ' + str(len(pat.line_trigger)) + ' ' + str(len(pat.frame_trigger)))

def _test_fov(test: unittest.TestCase, pat: RasterScanPattern, tolerance=0.1):
    x, y = _measure_fov(pat)
    err_x = abs(pat.fov[0] - x) / pat.fov[0]
    err_y = abs(pat.fov[1] - y) / pat.fov[1]
    test.assertTrue(err_x < tolerance, msg='Fast axis FOV has error of ' + str(err_x * 100)[0:5] + '%!')
    test.assertTrue(err_y < tolerance, msg='Slow axis FOV has error of ' + str(err_y * 100)[0:5] + '%!')
    

class scanpattern_test(unittest.TestCase):
    
    def test_figure_pat(self):
        pat = Figure8ScanPattern()
        pat.generate(1, 64, 76000, rotation_rad=np.pi / 4)
        _test_signal_len(self, pat)
        _print_line_scan_pattern(pat)
        print()
        
    def test_rose(self):
        pat = RoseScanPattern()
        pat.generate(3, 10, 128, 76000)
        _test_signal_len(self, pat)
        _print_line_scan_pattern(pat)
        print()
        
    def test_circle(self):
        pat = CircleScanPattern()
        pat.generate(64, 2.0, 76000)
        _test_signal_len(self, pat)
        _print_line_scan_pattern(pat)
        print()

    def test_raster(self):
        patterns = [
            RasterScanPattern(64, 64, 76000),
            RasterScanPattern(128, 128, 76000),
            RasterScanPattern(256, 256, 76000),
            RasterScanPattern(128, 128, 76000, fov=[10, 10]),
            RasterScanPattern(128, 128, 76000, flyback_duty=0.1),
            RasterScanPattern(128, 128, 76000, flyback_duty=0.2),
            RasterScanPattern(128, 128, 76000, flyback_duty=0.3),
            RasterScanPattern(128, 128, 76000, flyback_duty=0.4),
            RasterScanPattern(128, 128, 76000, flyback_duty=0.5),
            RasterScanPattern(128, 128, 76000, exposure_fraction=0.9),
            RasterScanPattern(128, 128, 76000, exposure_fraction=0.8),
            RasterScanPattern(128, 128, 76000, exposure_fraction=0.7),
            RasterScanPattern(128, 128, 76000, exposure_fraction=0.6),
            RasterScanPattern(128, 128, 76000, exposure_fraction=0.5),
            RasterScanPattern(128, 128, 76000, fast_axis_step=True),
            RasterScanPattern(128, 128, 76000, slow_axis_step=True),
            RasterScanPattern(128, 128, 76000, slow_axis_step=True, fast_axis_step=True),
            RasterScanPattern(128, 128, 76000, aline_repeat=2, bline_repeat=1),
            RasterScanPattern(128, 128, 76000, aline_repeat=2, bline_repeat=2),
            RasterScanPattern(128, 128, 76000, aline_repeat=1, bline_repeat=2),
            RasterScanPattern(128, 128, 76000, aline_repeat=1, bline_repeat=2),
            RasterScanPattern(128, 128, 76000, bidirectional=True),
            RasterScanPattern(128, 128, 76000, bidirectional=True, slow_axis_step=True),
            RasterScanPattern(65, 65, 76000, bidirectional=True, slow_axis_step=True),
            RasterScanPattern(65, 65, 76000, bidirectional=True),
            ]
        for pat in patterns:
            _test_signal_len(self, pat)
            _print_raster_pattern(pat)
            _test_fov(self, pat, 0.05)  # 5% FOV error tolerance
            print('------------')
            print('Test: PASSED')
            print('------------')
            print()

    
if __name__ == '__main__':
    
    result = unittest.main()
