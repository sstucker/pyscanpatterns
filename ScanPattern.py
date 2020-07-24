# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:09:59 2020
@author: sstucker
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class ScanPattern:

    def set_geometry(self, params):
        """
        Define the geometry of the pattern, overwriting old geometry and recreating the pattern
        :param params: List of parameters
        :return: 0 on success
        """
        raise NotImplementedError()

    def get_sample_rate(self):
        raise NotImplementedError()

    def get_signals(self):
        raise NotImplementedError()

    def get_trigger(self):
        raise NotImplementedError()

    def get_x(self):
        raise NotImplementedError()

    def get_y(self):
        raise NotImplementedError()

    def get_number_of_alines(self):
        """
        :return: The total number of line acquisitions per pattern
        """


class RasterScanPattern(ScanPattern):

    def __init__(self, pattern_rate=10, dac_samples_per_second=1000000, init_params=[]):
        """
        :param rate: Rate in hz to complete a scan
        :param dac_samples_per_second: Sample rate of generated drive signals in samples per second
        """

        self._debug = False

        self._x = np.array([])
        self._y = np.array([])
        self._cam = np.array([])
        self._fs = dac_samples_per_second
        self._pattern_rate = pattern_rate
        self._params = []

        # Raster specific properties
        self._alines = 100  # Default values
        self._blines = 1

        if len(init_params) > 0:
            self._params = init_params
            self.set_geometry(self._params)
            self._generate_raster_scan()
        else:
            self._generate_raster_scan(self._alines, self._blines)  # Some default pattern

    def get_raster_dimensions(self):
        """
        :return: [number of a-lines, number of b-lines]
        """
        return [self._alines, self._blines]

    def get_number_of_alines(self):
        return int(self._alines * self._blines)

    def set_geometry(self, params):
        """
        :param params:
            params[0]: Number of A-lines in raster scan
            params[1]: Number of B-lines in raster scan
            params[2]: Percentage of B-line scan time to be used for flyback signal. Default 5%
            params[3]: Percentage of entire fast-axis length to trigger exposures along. The ends of the fast-
            axis are prone to distortions as the galvos change direction
            params[4]: Line exposure time in microseconds. Too short a time cannot properly be converted
            depending on the sample rate. In these cases, the camera's fixed exposure time should be configured
            params[5]:  List [FOV width (fast axis), FOV height (slow axis)]
            params[6]: List [A-line spacing (fast axis), B-line spacing (slow axis)]
        :return: 0 on success
        """
        self._params = params
        try:
            self._alines = params[0]
            self._blines = params[1]
        except IndexError:
            return -1  # Not enough parameters to define the scan pattern
        if len(self._params) >= 7:
            self._generate_raster_scan(alines=self._params[0],
                                       blines=self._params[1],
                                       flyback_duty=self._params[2],
                                       exposure_width=self._params[3],
                                       exposure_time_us=self._params[4],
                                       fov=self._params[5],
                                       spacing=self._params[6])
        elif len(self._params) == 6:
            self._generate_raster_scan(alines=self._params[0],
                                       blines=self._params[1],
                                       flyback_duty=self._params[2],
                                       exposure_width=self._params[3],
                                       exposure_time_us=self._params[4],
                                       fov=self._params[5])
        elif len(self._params) == 5:
            self._generate_raster_scan(alines=self._params[0],
                                       blines=self._params[1],
                                       flyback_duty=self._params[2],
                                       exposure_width=self._params[3],
                                       exposure_time_us=self._params[4])
        elif len(self._params) == 4:
            self._generate_raster_scan(alines=self._params[0],
                                       blines=self._params[1],
                                       flyback_duty=self._params[2],
                                       exposure_width=self._params[3])
        elif len(self._params) == 3:
            self._generate_raster_scan(alines=self._params[0],
                                       blines=self._params[1],
                                       flyback_duty=self._params[2])
        elif len(self._params) == 2:
            self._generate_raster_scan(alines=self._params[0], blines=self._params[1])

        else:
            return -1

        return 0

    def get_sample_rate(self):
        return self._fs

    def get_signals(self):
        return [self._x, self._y, self._cam]

    def get_trigger(self):
        return self._cam

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def _generate_raster_scan(self, alines, blines, flyback_duty=0.25, exposure_width=0.8, exposure_time_us=100, fov=None, spacing=None):
        """
        Generates raster pattern. If no fov or spacing is passed, constrains FOV to -1, 1 voltage units
        :param alines: Number of A-lines in raster scan
        :param blines: Number of B-lines in raster scan
        :param flyback_duty: Percentage of B-line scan time to be used for flyback signal. Default 5%
        :param exposure_width: Percentage of entire fast-axis length to trigger exposures along. The ends of the fast-
            axis are prone to distortions as the galvos change direction
        :param exposure_time_us: Line exposure time in microseconds. Too short a time cannot properly be converted
            depending on the sample rate. In these cases, the camera's fixed exposure time should be configured
        :param fov: [FOV width (fast axis), FOV height (slow axis)] 2D field of view
        :param spacing: [A-line spacing (fast axis), B-line spacing (slow axis)] 2D spacing. Overrides fov definition
            if both are passed
        :return: -1 if error, 0 if successful
        """

        exposure_time_in_samples = int(self._fs * exposure_time_us * 10**-6)

        pattern_period = 1 / self._pattern_rate
        samples_per_pattern = int(self._fs / self._pattern_rate)
        samples_per_bline = int(samples_per_pattern / blines)
        samples_per_flyback = int(flyback_duty * samples_per_bline)

        flyback_y = (1 / blines) * flyback_duty

        tx = np.linspace(np.pi * flyback_duty, 2 * np.pi * blines + np.pi * flyback_duty, samples_per_pattern)
        x = signal.sawtooth(tx, width=flyback_duty)

        ty = np.linspace(np.pi * flyback_y, 2 * np.pi + np.pi * flyback_y, samples_per_pattern)
        y = signal.sawtooth(ty, width=flyback_y)

        scan_axis_starts = np.arange(int(samples_per_flyback/2), samples_per_pattern, samples_per_bline)
        scan_axis_ends = np.arange(samples_per_bline - int(samples_per_flyback/2), samples_per_pattern, samples_per_bline)
        
        scan_length_starts = scan_axis_starts + ((samples_per_bline - samples_per_flyback) * ((1 - exposure_width) / 2))
        scan_length_ends = scan_axis_ends - ((samples_per_bline - samples_per_flyback) * ((1 - exposure_width) / 2))

        scan_exposure_starts = []
        for i in range(blines):
            scan_exposure_starts.append(np.linspace(scan_length_starts[i], scan_length_ends[i], alines))
        scan_exposure_starts = np.around(scan_exposure_starts).astype(int).flatten()

        exposures = np.zeros(samples_per_pattern)
        for exposure in scan_exposure_starts:
            exposure = int(exposure)
            exposures[exposure:exposure + exposure_time_in_samples] = np.ones(exposure_time_in_samples)

        self._cam = exposures
        self._x = x
        self._y = y
        print('Generated', alines, 'by', blines, 'unidirectional raster pattern with', samples_per_pattern, 'samples per pattern')

        if self._debug:

            f1 = plt.figure(1)
            plt.plot(x, label='x-galvo')
            plt.plot(y, label='y-galvo')
            plt.plot(exposures*.5, label='exposure-onset')

            plt.scatter(scan_exposure_starts, np.zeros(np.shape(scan_exposure_starts)))

            plt.legend()

            f2 = plt.figure(2)
            plt.plot(x, y)
            plt.scatter(x[0], y[0], label='initial position')
            plt.scatter(x[scan_exposure_starts], y[scan_exposure_starts], label='exposures')
            plt.legend()

            plt.show()

        return 0


