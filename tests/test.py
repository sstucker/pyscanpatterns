import unittest
from scanpattern import *  

def _print_line_scan_pattern(pattern: LineScanPattern):
    print('Pattern type:', pattern.__class__.__name__)
    print('Number of samples', len(pattern.x))
    print('Number of A-lines', pattern.total_number_of_alines)
    print('Dimensions', pattern.dimensions)
    print('Sample rate', pattern.sample_rate, 'Hz')
    print('Pattern rate', pattern.pattern_rate, 'Hz')


class scanpattern_test(unittest.TestCase):
    
    def test_figure_eight(self):
        eight = Figure8ScanPattern()
        eight.generate(1, 64, 76000, rotation_rad=np.pi / 4)
        _print_line_scan_pattern(eight)
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