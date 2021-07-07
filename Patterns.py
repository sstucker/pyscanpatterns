# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:09:59 2020
@author: sstucker
"""
import numpy as np

def sigmoidspace(start, stop, n, b=1):
    return start + (stop - start) / (1 + np.exp(-b * np.linspace(-10, 10, n)))


def rotfunc(xs, ys, rotation_rad):
    rot = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                    [np.sin(rotation_rad), np.cos(rotation_rad)]])
    xy = np.column_stack([xs, ys])
    prod = np.matmul(xy, rot)
    return prod[:, 0], prod[:, 1]


class LineScanPattern:
    """
    Template for all scan patterns for the generation of 2D or 3D images acquired with A-line scans. Generates
    normalized signals for the steering of galvos and the triggering of a line camera in order to acquire a given
    geometry.
    """

    def get_line_trig(self):
        return self._line_trig

    def get_frame_trig(self):
        return self._frame_trig

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_sample_rate(self):
        """
        :return: The approximate sample rate in hz that samples should be generated such that the pattern is
        completed at the rate specified
        """
        raise NotImplementedError()

    def get_pattern_rate(self):
        """
        :return: The approximate rate at which the pattern is completed by the scanner given by the sample rate and number of samples.
        """
        raise NotImplementedError()

    def get_dimensions(self):
        """
        :return: An array of pattern dimensions which orient the A-scans with respect to one another. i.e. a raster
        pattern's first and second dimension represent the height and width of the pattern in A-lines.
        """
        raise NotImplementedError()

    def get_total_number_of_alines(self):
        """
        :return: The total number of line acquisitions per pattern
        """
        raise NotImplementedError()

    def generate(self):
        """
        Generate the pattern from some number of parameters. The pattern object is unuseable until this is called
        :return: 0 on success
        """
        raise NotImplementedError()


class Figure8ScanPattern(LineScanPattern):

    def __init__(self, max_trigger_rate=75000):

        self._max_rate = max_trigger_rate

        self._x = np.array([])
        self._y = np.array([])
        self._line = np.array([])
        self._frame = np.array([])
        self._pattern_rate = None
        self._fs = None
        self._aline_width = None
        self._a_per_b = None

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_dimensions(self):
        return [self._a_per_b, 2]

    def get_total_number_of_alines(self):
        return int(self._a_per_b * 2)

    def get_sample_rate(self):
        return self._fs

    def get_signals(self):
        return [self._line, self._frame, self._x, self._y]

    def generate(self, aline_width=0.05, aline_per_b=32, samples_on=1, samples_off=None, rotation_rad=0):

        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self._aline_width = aline_width
        self._a_per_b = int(aline_per_b)

        # Build B-line trigger
        btrig = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self._a_per_b)

        # TODO parameterize these, especially H
        G = np.pi / 2.5  # Cylical point at which B-scans begin in the figure
        A = 1 / 1.6289944852363252  # Factor such that B-scans given a G are normalized in length
        AY = 1.223251  # Stretch factor along second axis to achieve perpendicular B-scans
        H = 5  # 1 / H is the proportion of the B-line length used for flyback in samples

        def fig8(t):
            x = A * np.cos(t)
            y = A * np.cos(t) * np.sin(t) * AY
            return [x, y]

        t_fb0 = sigmoidspace(-G, G, int(1 / H * len(btrig)), b=0.2)
        [x_fb0, y_fb0] = fig8(t_fb0)

        t_fb1 = sigmoidspace(-G + np.pi, G + np.pi, int(1 / H * len(btrig)), b=0.2)
        [x_fb1, y_fb1] = fig8(t_fb1)

        x_b1 = np.linspace(x_fb0[-1], x_fb1[0], len(btrig))
        y_b1 = np.linspace(y_fb1[-1], y_fb0[0], len(btrig))

        x_b0 = np.linspace(x_fb1[-1], x_fb0[0], len(btrig))
        x_b0 = np.linspace(x_fb1[-1], x_fb0[0], len(btrig))
        y_b0 = np.linspace(y_fb1[-1], y_fb0[0], len(btrig))

        b0_norm = [x_b0[-1] - x_b0[0], y_b0[-1] - y_b0[0]] / np.linalg.norm([x_b0[-1] - x_b0[0], y_b0[-1] - y_b0[0]])
        b1_norm = [x_b1[-1] - x_b1[0], y_b1[-1] - y_b1[0]] / np.linalg.norm([x_b1[-1] - x_b1[0], y_b1[-1] - y_b1[0]])

        x = np.concatenate([x_b0[0:-1], x_fb0[0:-1], x_b1[0:-1], x_fb1[0:-1]]) * self._aline_width
        y = np.concatenate([y_b0[0:-1], y_fb0[0:-1], y_b1[0:-1], y_fb1[0:-1]]) * self._aline_width

        x, y = rotfunc(x, y, rotation_rad)

        fb = np.zeros(len(x_fb0))
        line_trigger = np.concatenate([btrig, fb, btrig, fb])
        frame_start = np.zeros(len(line_trigger))
        frame_start[0:samples_on] = 1

        self._x = x
        self._y = y
        self._line = line_trigger
        self._frame = frame_start

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        self._pattern_rate = 1 / ((1 / self._fs) * len(line_trigger))


class RoseScanPattern(LineScanPattern):

    def __init__(self, max_trigger_rate=75950):

        self._max_rate = max_trigger_rate

        self._x = np.array([])
        self._y = np.array([])
        self._line = np.array([])
        self._frame = np.array([])
        self._pattern_rate = None
        self._fs = None

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_dimensions(self):
        return [self._a_per_b, 2]

    def get_total_number_of_alines(self):
        return int(self._a_per_b * 2)

    def get_sample_rate(self):
        return self._fs

    def get_signals(self):
        return [self._line, self._frame, self._x, self._y]

    def generate(self, p, aline_width, aline_per_b, samples_on=2, samples_off=None, rotation_rad=0):

        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        self._aline_width = aline_width
        self._a_per_b = int(aline_per_b)

        b_frac = 0.2
        H = 5  # 1 / H is the proportion of the B-line length used for flyback in samples

        p = int(p)
        if p % 2 == 0:
            return  # TODO implement even petaled roses
        #            k = 2*p
        #            p0 = 0
        #            period = 2 * np.pi
        else:
            k = p
            p0 = 0
            period = np.pi

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        b_trig = np.tile(np.append(np.ones(samples_on), np.zeros(samples_off)), self._a_per_b)

        t = np.array([])
        lt = np.array([])

        period_p = period / k

        period_b = period_p * b_frac
        period_fb = period_p * (1 - b_frac)

        for i in range(p):  # Generate each petal

            p_start = p0 + i * period_p

            t_fb = sigmoidspace(p_start, p_start + period_fb, int(1 / H * len(b_trig)), b=1)
            t_b = np.linspace(p_start + period_fb, p_start + period_fb + period_b, len(b_trig))

            t_p = np.concatenate([t_fb, t_b])

            trig_p = np.concatenate([np.zeros(len(t_fb)), b_trig])

            def xfunc(ts):
                return aline_width * np.cos(p * ts) * np.cos(ts)

            def yfunc(ts):
                return aline_width * np.cos(p * ts) * np.sin(ts)

            t = np.append(t, t_p)
            lt = np.append(lt, trig_p)

        t -= period_fb / 2

        x = xfunc(t)
        y = yfunc(t)

        x, y = rotfunc(x, y, rotation_rad)

        self._x = x
        self._y = y
        self._line = lt

        ft = np.zeros(len(lt))
        ft[0:samples_on] = 1
        self._frame = ft
        self._pattern_rate = 1 / ((1 / self._fs) * len(x))


class BidirectionalRasterScanPattern(LineScanPattern):
    
    def __init__(self, max_trigger_rate=76000):
        self._max_rate = max_trigger_rate

        self._pattern_rate = None

        self._x = np.array([])
        self._y = np.array([])
        self._line_trig = np.array([])
        self._frame_trig = np.array([])
        self._fs = None
        self._alines = None
        self._blines = None
        self._aline_repeat = None
        
    def get_dimensions(self):
        """
        :return: [number of a-lines, number of b-lines]
        """
        return [self._alines, self._blines]

    def get_total_number_of_alines(self):
        return int(self._alines * self._blines)

    def get_sample_rate(self):
        return self._fs

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_signals(self):
        return [self._line_trig, self._frame_trig, self._x, self._y]

    def get_aline_repeat(self):
        return self._aline_repeat

    def generate(self, alines=64, blines=64, exposure_percentage=0.7, flyback_duty=0.1, fov=None, samples_on=1, samples_off=None):
        
        # Generate a single B-scan
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        if fov is None:
            fov = [1, 1]

        self._alines = int(alines)
        self._blines = int(blines)

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples
        
        bline_trig = np.tile(np.concatenate([np.ones(samples_on), np.zeros(samples_off)]), self._alines)
        bline_pad_len = int((1 - exposure_percentage) / 2 * len(bline_trig))
        bline_trig = np.concatenate([np.zeros(bline_pad_len), bline_trig, np.zeros(bline_pad_len)])
        
        fast_axis_scan = np.array([])
        slow_axis_scan = np.array([])
        
        fast_range = (fov[0] + fov[0] * (1 - exposure_percentage)) / 2  # +/-
        slow_range = fov[1] / 2
        
        bpos = np.linspace(-slow_range, slow_range, self._blines)
        for i in range(blines):
            if i % 2 == 0:
                xs = np.linspace(fast_range, -fast_range, len(bline_trig) + 1)[1::]
            else:
                xs = np.linspace(-fast_range, fast_range, len(bline_trig) + 1)[1::]
            fast_axis_scan = np.append(fast_axis_scan, xs)
        
        for i in range(len(bpos) - 1): 
            slow_axis_scan = np.append(slow_axis_scan, np.concatenate([np.ones(len(bline_trig) - bline_pad_len) * bpos[i],
                                                                       np.linspace(bpos[i], bpos[i + 1], bline_pad_len + 1)[1::]]))
        slow_axis_scan = np.append(slow_axis_scan, np.ones(len(bline_trig) - bline_pad_len) * bpos[-1])
        
        
        # flyback_slow = sigmoidspace(slow_axis_scan[-1], slow_axis_scan[0], bline_pad_len)
        flyback_slow = np.linspace(slow_axis_scan[-1], slow_axis_scan[0], bline_pad_len)
        
        self._x = fast_axis_scan
        self._y = np.concatenate([slow_axis_scan, flyback_slow])
        self._line_trig = np.tile(bline_trig, self._blines)
        
        self._frame_trig = np.zeros(len(self._line_trig))
        self._frame_trig[0:samples_on] = 1
        
        self._pattern_rate = 1 / (len(self._x) * (1 / self._fs))

class RasterScanPattern(LineScanPattern):

    def __init__(self, max_trigger_rate=75900):
        """
        :param max_trigger_rate: The maximum rate that the camera can be triggered in Hz. The scan pattern and its
        rate will be determined to achieve but not exceed this rate. Default < 76 kHz.
        """

        self._max_rate = max_trigger_rate

        self._pattern_rate = None

        self._x = np.array([])
        self._y = np.array([])
        self._line_trig = np.array([])
        self._frame_trig = np.array([])
        self._fs = None
        self._alines = None
        self._blines = None
        self._aline_repeat = None

    def get_dimensions(self):
        """
        :return: [number of a-lines, number of b-lines]
        """
        return [self._alines, self._blines]

    def get_total_number_of_alines(self):
        return int(self._alines * self._blines)

    def get_sample_rate(self):
        return self._fs

    def get_pattern_rate(self):
        return self._pattern_rate

    def get_signals(self):
        return [self._line_trig, self._frame_trig, self._x, self._y]

    def get_aline_repeat(self):
        return self._aline_repeat

    def generate(self, alines=64, blines=1, flyback_duty=0.2, trigger_width=1,
                 fov=None, samples_on=2, samples_off=None, samples_park=1, samples_step=1, rotation_rad=0,
                 fast_axis_step=False, slow_axis_step=True, aline_repeat=1, trigger_blines=False):

        self._aline_repeat = aline_repeat

        # Generate a single B-scan
        samples_on = int(samples_on)

        if samples_off is None:
            samples_off = int(samples_on)
        else:
            samples_off = int(samples_off)

        if fov is None:
            fov = [1, 1]

        self._alines = int(alines)
        self._blines = int(blines)

        period_samples = samples_on + samples_off
        self._fs = self._max_rate * period_samples

        if aline_repeat < 2:
            aline_repeat = 1
        else:
            fast_axis_step = True

        aline_trig = np.concatenate([np.ones(samples_on), np.zeros(samples_off)])
        if fast_axis_step is True:
            aline_trig = np.concatenate(
                [np.zeros(samples_park), np.tile(aline_trig, aline_repeat), np.zeros(samples_step)])

        bline_trig = np.tile(aline_trig, self._alines)
        bline_pad = np.zeros(int((len(bline_trig) * (1 - trigger_width)) / (2 * trigger_width)))
        bline_padded = np.concatenate([bline_pad, bline_trig, bline_pad])
        flyback_pad = np.zeros(int((len(bline_padded) * flyback_duty) / (1 - flyback_duty)))
        bline_trig = np.concatenate([flyback_pad, bline_padded])

        fast_axis_flyback = np.linspace(-fov[0] / 2, fov[0] / 2, len(flyback_pad) + 2)[1:-1]

        if fast_axis_step is False:
            fast_axis_scan = np.linspace(fov[0] / 2, -fov[0] / 2, len(bline_padded))
        else:
            fast_axis_scan = np.array([])
            positions = np.linspace(fov[0] / 2, -fov[0] / 2, self._alines)
            fast_axis_scan = np.append(fast_axis_scan, positions[0] * np.ones(len(bline_pad)))
            for i in range(len(positions) - 1):
                fast_axis_scan = np.append(fast_axis_scan, positions[i] * np.ones(len(aline_trig) - samples_step))
                fast_axis_scan = np.append(fast_axis_scan,
                                           np.linspace(positions[i], positions[i + 1], samples_step + 2)[1:-1])
            fast_axis_scan = np.append(fast_axis_scan, positions[-1] * np.ones(len(aline_trig) + len(bline_pad)))
        bline_scan = np.concatenate([fast_axis_flyback, fast_axis_scan])

        line_trig = np.tile(bline_trig, self._blines)

        if self._alines > 1:
            x = np.tile(bline_scan, self._blines)
        else:
            x = np.zeros(len(line_trig))

        if trigger_blines:
            f = np.zeros(len(bline_trig))
            f[0:samples_on] = 1
            frame_trig = np.tile(f, self._blines)
        else:
            frame_trig = np.zeros(len(line_trig))
            frame_trig[0:samples_on] = 1

        if self._blines > 1:

            slow_axis_flyback = fast_axis_flyback

            if slow_axis_step is False:
                slow_axis_scan = np.concatenate([np.linspace(-fov[0] / 2, fov[0] / 2, len(flyback_pad) + 2)[1:-1],
                                                 np.linspace(fov[0] / 2, -fov[0] / 2,
                                                             len(blines_scan) - len(flyback_pad))])
            else:
                slow_axis_scan = np.array([])
                positions = np.linspace(fov[0] / 2, -fov[0] / 2, self._alines)
                for i in range(len(positions) - 1):
                    slow_axis_scan = np.append(slow_axis_scan, positions[i] * np.ones(len(fast_axis_scan)))
                    slow_axis_scan = np.append(slow_axis_scan,
                                               np.linspace(positions[i], positions[i + 1], len(fast_axis_flyback) + 2)[
                                               1:-1])
                slow_axis_scan = np.append(slow_axis_scan, positions[-1] * np.ones(len(fast_axis_scan)))

            y = np.concatenate([slow_axis_flyback, slow_axis_scan])

        else:

            y = np.zeros(len(x))

        self._x = x
        self._y = y
        self._line_trig = line_trig
        self._frame_trig = frame_trig

        self._x, self._y = rotfunc(self._x, self._y, rotation_rad)

        self._pattern_rate = 1 / (len(self._x) * (1 / self._fs))

        return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close('all')

    pat = BidirectionalRasterScanPattern()

    # TODO fix B = 1
    pat.generate(alines=16, blines=16, fov=[1,1])
    
    plt.figure(1)
    plt.subplot(3, 1, 2)
    plt.plot(pat.get_x())
    plt.plot(pat.get_y())
    plt.xlim(0, len(pat.get_x()))

    plt.subplot(3, 1, 3)
    plt.plot(pat.get_frame_trig())
    plt.plot(pat.get_line_trig())
    plt.xlim(0, len(pat.get_x()))

    ax = plt.subplot(3, 1, 1)
    plt.plot(pat.get_x(), pat.get_y(), '-k', linewidth=0.1, alpha=0.5)
    plt.scatter(pat.get_x()[pat.get_line_trig().astype(bool)[0:len(pat.get_x())]],
                pat.get_y()[pat.get_line_trig().astype(bool)[0:len(pat.get_y())]], s=0.2)
    ax.set_aspect('equal')

    plt.show()
