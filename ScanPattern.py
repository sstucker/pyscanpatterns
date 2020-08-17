# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:09:59 2020
@author: sstucker
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class ScanPattern:

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

    def get_total_number_of_alines(self):
        """
        :return: The total number of line acquisitions per pattern
        """
        raise NotImplementedError()

    def generate(self):
        raise NotImplementedError()


class RasterScanPattern(ScanPattern):

    def __init__(self, max_trigger_rate=75000, dac_samples_per_second=40000, debug=False):
        """
        :param max_trigger_rate: The maximum rate that the camera can be triggered in Hz. The scan pattern and its sample
            rate will be determined to achieve but not exceed this rate. Default 76 kHz.
        :param debug: Bool to enable verbose console output. Default False
        """

        self._debug = debug
        self._max_rate = max_trigger_rate

        self._x = np.array([])
        self._y = np.array([])
        self._cam = np.array([])
        self._fs = dac_samples_per_second
        self._scan_exposure_starts = []

        self._alines = 100  # Default RasterScanPattern values
        self._blines = 1

    def get_raster_dimensions(self):
        """
        :return: [number of a-lines, number of b-lines]
        """
        return [self._alines, self._blines]

    def get_total_number_of_alines(self):
        return int(self._alines * self._blines)

    def get_sample_rate(self):
        return self._fs

    def get_signals(self):
        return [self._x, self._y, self._cam]

    def get_exposure_starts(self):
        return self._scan_exposure_starts

    def get_trigger(self):
        return self._cam

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def generate(self, alines, blines, flyback_duty=0.15, exposure_width=0.7,
                             fov=None, samples_on=None, samples_off=2):
        """
        Generates raster pattern. If no fov or spacing is passed, constrains FOV to -1, 1 voltage units
        :param alines: Number of A-lines in raster scan
        :param blines: Number of B-lines in raster scan
        :param flyback_duty: Percentage of B-line scan time to be used for flyback signal. Default 5%
        :param exposure_width: Percentage of entire fast-axis length to trigger exposures along. The ends of the fast-
            axis are prone to distortions as the galvos change direction
        :param fov: [FOV width (fast axis), FOV height (slow axis)] 2D field of view in spatial units
        :param samples_on: Line exposure time in samples. Must exceed 0. Default None: same as samples off (50% duty)
        :param samples_off: The number of low samples to generate between exposure trigger pulses. Limited by dac rate.
            Default 2.
        :return: -1 if error, 0 if successful
        """
        start = time.time()

        self._alines = int(alines)
        self._blines = int(blines)

        if fov is None:
            fov = [1, 1]

        if samples_on is None:
            samples_on = samples_off

        # Determine signal sizes and sample rate to achieve max camera trigger rate
        samples_per_bline = (self._alines * (samples_off + samples_on)) * (1 / exposure_width)
        samples_per_bline += samples_per_bline * flyback_duty
        samples_per_bline = int(samples_per_bline)

        samples_per_pattern = samples_per_bline * self._blines
        samples_per_flyback = int(flyback_duty * samples_per_bline)

        trigger_period_samples = samples_on + samples_off
        self._fs = int(trigger_period_samples * self._max_rate)

        flyback_y = (1 / self._blines) * flyback_duty

        # Generate exposure trigger pulse
        bline_pulse = np.tile(np.concatenate([np.zeros(samples_off), np.ones(samples_on)]), self._alines)
        bline_pulse_len = len(bline_pulse)

        # Generate sawtooth signals
        tx = np.linspace(np.pi * flyback_duty, 2 * np.pi * self._blines + np.pi * flyback_duty, samples_per_pattern)
        x = signal.sawtooth(tx, width=flyback_duty)

        ty = np.linspace(np.pi * flyback_y, 2 * np.pi + np.pi * flyback_y, samples_per_pattern)
        y = signal.sawtooth(ty, width=flyback_y)

        scan_axis_starts = np.arange(int(samples_per_flyback / 2), samples_per_pattern, samples_per_bline)
        scan_axis_ends = np.arange(samples_per_bline - int(samples_per_flyback / 2), samples_per_pattern,
                                   samples_per_bline)

        scan_length_starts = scan_axis_starts + ((samples_per_bline - samples_per_flyback) * ((1 - exposure_width) / 2))
        scan_length_ends = scan_axis_ends - ((samples_per_bline - samples_per_flyback) * ((1 - exposure_width) / 2))

        exposures = np.zeros(samples_per_pattern)
        for exposure_start in scan_length_starts.astype(int):
            exposures[exposure_start:exposure_start+bline_pulse_len] = bline_pulse

        self._cam = exposures
        # Scale X and Y by FOV
        self._x = x * (fov[0] / 2)
        self._y = y * (fov[1] / 2)

        dac_period_us = (1 / self._fs) * 10 ** 6
        trig_period_samples = samples_on + samples_off
        if self._debug:
            print('Generated', alines, 'by', blines, 'unidirectional raster pattern with', samples_per_pattern,
                  'samples per pattern')
            print('DAC sample rate:', self._fs, 'hz')
            print('Trigger rate:', str(1 / ((1 / self._fs) * trig_period_samples))[0:9], 'lines-per-second')
            print('Trigger period:', str(dac_period_us * trig_period_samples)[0:9], 'us')
            print('Trigger width:', str(dac_period_us * samples_on)[0:9], 'us')
            print('P2P voltages:')
            print('   Trigger:', max(self._cam), 'to', min(self._cam))
            print('   X-galvo:', max(self._x), 'to', min(self._x))
            print('   Y-galvo:', max(self._y), 'to', min(self._y))
            print('Elapsed', time.time() - start, 's')

        if self._debug:

            f1 = plt.figure(1)
            plt.plot(x, label='x-galvo')
            plt.plot(y, label='y-galvo')
            plt.plot(exposures*.5, label='cam trigger')

            plt.scatter(scan_axis_starts, np.zeros(np.shape(scan_axis_starts)))
            plt.scatter(scan_axis_ends, np.zeros(np.shape(scan_axis_ends)))

            plt.legend()

            f2 = plt.figure(2)
            plt.plot(x, y)
            plt.scatter(x[0], y[0], label='initial position')
            plt.scatter(x[scan_length_starts.astype(int)], y[scan_length_starts.astype(int)], label='pulse starts')
            plt.scatter(x[scan_length_ends.astype(int)], y[scan_length_ends.astype(int)], label='pulse ends')
            for i, exp in enumerate(scan_length_starts.astype(int)):
                plt.annotate(str(i+1), [x[exp], y[exp]])
            plt.legend()

            plt.show()

        return 0


class Figure8ScanPattern(ScanPattern):

    def __init__(self, max_trigger_rate=75000, dac_samples_per_second=40000, debug=False):

        self._debug = debug
        self._max_rate = max_trigger_rate

        self._x = np.array([])
        self._y = np.array([])
        self._cam = np.array([])
        self._fs = dac_samples_per_second

        self._aline_width = 1  # Default RasterScanPattern values
        self._a_per_b = 64

    def get_raster_dimensions(self):
        """
        :return: [number of a-lines, number of b-lines (always 2 for a figure-8)]
        """
        return [self._a_per_b, 2]

    def get_total_number_of_alines(self):
        return int(self._a_per_b * 2)

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

    def generate(self, aline_width, aline_per_b, samples_on=2, samples_off=None):

        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self._aline_width = aline_width
        self._a_per_b = int(aline_per_b)

        # The fraction of a figure 8 half cycle used for the exposure such that aline width is 1
        b_frac = 0.23856334924933

        def xfunc(ts):
            return aline_width * np.cos(ts)

        def yfunc(ts):
            return aline_width * np.cos(ts) * np.sin(ts)

        # Start and end of B-line in radian time units
        bstart = np.pi / 2 - (np.pi * b_frac) / 2
        bend = np.pi / 2 + (np.pi * b_frac) / 2

        # The distance along the B-line
        bdist = np.sqrt((xfunc(bstart) - xfunc(bend)) ** 2 + (yfunc(bstart) - yfunc(bend)) ** 2)  # mm

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        # Build B-line trigger
        bline = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self._a_per_b)
        bpad = np.zeros(np.ceil((len(bline) - len(bline) * b_frac) / (2 * b_frac)).astype(int))
        trigger = np.concatenate([bpad, bline, bpad, bpad, bline, bpad])

        b1idx = np.concatenate([np.zeros(len(bpad)), np.ones(len(bline)), np.zeros(3 * len(bpad) + len(bline))]).astype(
            bool)
        b2idx = np.concatenate([np.zeros(3 * len(bpad) + len(bline)), np.ones(len(bline)), np.zeros(len(bpad))]).astype(
            bool)

        b1x = np.linspace(xfunc(bstart), xfunc(bend), len(bline))
        b1y = np.linspace(yfunc(bstart), yfunc(bend), len(bline))
        b2x = np.linspace(xfunc(bstart + np.pi), xfunc(bend + np.pi), len(bline))
        b2y = np.linspace(yfunc(bstart + np.pi), yfunc(bend + np.pi), len(bline))

        t = np.linspace(0, 2 * np.pi, len(trigger))
        x = xfunc(t)
        y = yfunc(t)
        x[b1idx] = b1x
        y[b1idx] = b1y
        x[b2idx] = b2x
        y[b2idx] = b2y

        self._x = x
        self._y = y
        self._cam = trigger

        if self._debug:

            print('Generated Figure-8 pattern with', len(trigger),
                  'samples per pattern')
            print('DAC sample rate:', self._fs, 'hz')
            print('P2P voltages:')
            print('   Trigger:', max(self._cam), 'to', min(self._cam))
            print('   X-galvo:', max(self._x), 'to', min(self._x))
            print('   Y-galvo:', max(self._y), 'to', min(self._y))

            plt.figure(1)
            plt.plot(trigger)
            plt.plot(x)
            plt.plot(y)

            plt.figure(2)
            ax2 = plt.axes()
            ax2.set_aspect('equal')
            ax2.plot(x, y)
            ax2.scatter(x[trigger.astype(bool)], y[trigger.astype(bool)])

            plt.show()

if __name__ == "__main__":
    raster = RasterScanPattern(debug=True)
    raster.generate(10, 1, fov=[0.05, 0.05])
    fig8 = Figure8ScanPattern(debug=True)
    fig8.generate(0.05, 64)