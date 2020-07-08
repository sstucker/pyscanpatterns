# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:09:59 2020

@author: sstucker
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class ScanPattern:

    def __init__(self, geometry, axial=False, pattern_rate=5, order=3, dac_samples_per_second=100000):
        """
        :param geometry: The type of scan pattern to generate. Options include:
            'raster': Conventional raster scan (C-scan)
            'circle': Circular scan
            'lemniscate': A figure-8 with two perpendicular B-scans
            'rose': Odd petaled rhodonea function with n-perpendicular B-scans
        :param axial: Whether or not to generate a Z-coordinate signal in the 3rd axis
        :param rate: Rate in hz to complete a C scan
        :param order: for use with 'rose' patterns, number of B-scans [3, 5, 7, 9]
        :param ss: Sample rate of generated drive signals in samples per second
        """

        self.pattern_rate = pattern_rate
        self.fs = dac_samples_per_second

        self._x = np.array([])
        self._y = np.array([])
        self._trigger = np.array([])
        if axial:
            self._z = np.array([])

        if geometry is 'raster':
            self._generate_raster_scan
        elif geometry is 'circe':
            pass
        elif geometry is 'lemniscate':
            pass
        elif geometry is 'rose':
            pass
        else:
            print('Geometry mode ', geometry, ' not understood. Initializing in Raster mode.')

    def _generate_raster_scan(self, alines, blines, flyback_duty=0.25, exposure_width=0.8, fov=None, spacing=None):
        """
        Generates raster pattern. If no fov or spacing is passed, constrains FOV to -1, 1 voltage units
        :param alines: Number of A-lines in raster scan
        :param blines: Number of B-lines in raster scan
        :param flyback_duty: Percentage of B-line scan time to be used for flyback signal. Default 5%
        :param exposure_width: Percentage of entire fast-axis length to trigger exposures along. The ends of the fast-
            axis are prone to distortions as the galvos change direction
        :param fov: [a, b] 2D field of view
        :param spacing: [a, b] 2D spacing. Overrides fov definition if both are passed
        :return: -1 if error, 0 if successful
        """

        exposure_time_in_samples = 100

        pattern_period = 1 / self.pattern_rate
        samples_per_pattern = int(self.fs / self.pattern_rate)
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
        scan_exposure_starts = np.around(scan_exposure_starts).astype(int)

        exposures = np.zeros(samples_per_pattern)
        for exposure in scan_exposure_starts:
            print(exposure)
            exposures[exposure] = 1

        f1 = plt.figure(1)
        plt.plot(x, label='x-galvo')
        plt.plot(y, label='y-galvo')
        plt.plot(exposures*.5, label='exposure-onset')
        # plt.scatter(scan_axis_starts, np.ones(np.shape(scan_axis_starts)))
        # plt.scatter(scan_axis_ends, -1 * np.ones(np.shape(scan_axis_starts)))
        # plt.scatter(scan_length_starts, np.ones(np.shape(scan_axis_starts)))
        # plt.scatter(scan_length_ends, -1 * np.ones(np.shape(scan_axis_starts)))

        plt.scatter(scan_exposure_starts, np.zeros(np.shape(scan_exposure_starts)))

        plt.legend()

        f2 = plt.figure(2)
        plt.plot(x, y)
        plt.scatter(x[0], y[0], label='initial position')
        plt.scatter(x[scan_exposure_starts], y[scan_exposure_starts], label='exposures')
        plt.legend()

        plt.show()


myPat = ScanPattern('raster')
myPat._generate_raster_scan(10, 10)

