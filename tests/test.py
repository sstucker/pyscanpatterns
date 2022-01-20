import unittest
from scanpatterns import *  

def _print_line_scan_pattern(pattern: LineScanPattern):
    print('Pattern type:', pattern.__class__.__name__)
    print('Number of samples', len(pattern.x))
    print('Number of A-lines', pattern.total_number_of_alines)
    print('Dimensions', pattern.dimensions)
    print('Sample rate', pattern.sample_rate, 'Hz')
    print('Pattern rate', pattern.pattern_rate, 'Hz')


def _test_signal_len(test: unittest.TestCase, pat: LineScanPattern):
    test.assertTrue(all([len(pat.x) == len(l) for l in [pat.x, pat.y, pat.line_trigger, pat.frame_trigger]]),
                    msg='Scan signals of ' + pat.__class__.__name__ + ' vary in length! Lengths:'+ str(len(pat.x))
                    + ' ' + str(len(pat.y)) + ' ' + str(len(pat.line_trigger)) + ' ' + str(len(pat.frame_trigger)))


class scanpattern_test(unittest.TestCase):
    
    def test_figure_pat(self):
        pat = Figure8ScanPattern()
        pat.generate(1, 64, 76000, rotation_rad=np.pi / 4)
        _test_signal_len(self, pat)
        
        _print_line_scan_pattern(pat)
        print()
        
    def test_rose(self):
        rose = RoseScanPattern()
        rose.generate(3, 10, 128, 76000)
        _print_line_scan_pattern(rose)
        print()

    def test_raster(self):
        raster = RasterScanPattern(256, 256, 76000, exposure_fraction=0.7)
        _print_line_scan_pattern(raster)
        print()

    def test_bidirectional_raster(self):
        raster = BidirectionalRasterScanPattern(64, 64, 76000)
        _print_line_scan_pattern(raster)
        print()

    def test_bidirectional_raster(self):
        raster = BlineRepeatedRasterScan(64, 64, 76000)
        _print_line_scan_pattern(raster)
        print()
        
    
if __name__ == '__main__':
    
    result = unittest.main()
