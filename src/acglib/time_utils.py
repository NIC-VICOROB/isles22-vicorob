import datetime
import time
import numpy as np

def format_time_interval(seconds, time_format=None):
    """Formats a time interval into a string.

    :param seconds: the time interval in seconds
    :param str time_format: (optional) Time format specification string (see `Time format code specification <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_ for information on how to format time)

    :Example:

    >>> format_time_interval(30)
    '00:30'
    >>> format_time_interval(300)
    '05:00'
    >>> format_time_interval(seconds=4000)
    '01:06:40'
    >>> format_time_interval(seconds=4000, time_format='%H hours and %M minutes')
    '01 hours and 06 minutes'
    """

    if time_format is None:
        if 0 < seconds < 3600:
            time_format = "%M:%S"
        elif 3600 < seconds < 24 * 3600:
            time_format = "%H:%M:%S"
        else:
            time_format = "%d days, %H:%M:%S"
    formatted_time = time.strftime(time_format, time.gmtime(seconds))
    return formatted_time


def get_timestamp(formatted=True, time_format='%Y-%m-%d_%H:%M:%S'):
    """Returns a formatted timestamp of the current system time.

    :param formatted: (default: True)
    :param time_format: (default: '%Y-%m-%d_%H:%M:%S') Time format specification string (see `Time format code specification <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_ for information on how to format time)

    :Example:

    >>> get_timestamp()
    '2019-12-01_00:00:01'
    >>> get_timestamp(formatted=False)
    1575562508.845833
    >>> get_timestamp(time_format='%d/%m/%Y')
    '01/12/2019'
    """
    now = datetime.datetime.now()
    return now.strftime(time_format) if formatted else now.timestamp()


class RemainingTimeEstimator:
    """Provides an estimation the remaining execution time.

    .. py:method:: update(iter_num)

        :return: (str) formatted estimated remaining time

    .. py:method:: elapsed_time()

        :return: (str) formatted elapsed time (time passed since start)

    :Example:

    >>> rta = RemainingTimeEstimator(3)
    >>> for i in range(3):
    >>>     time.sleep(1)
    >>>     print(rta.update(i))
    >>> print('Total ' + rta.elapsed_time())
    00:03
    00:02
    00:01
    Total 0:03
    """

    def __init__(self, total_iters):
        self.total_iters = total_iters
        self.start_time = time.time()

        self.iter_times = []
        self.last_iter = {'num': -1, 'time': time.time()}

    def update(self, iter_num):
        assert iter_num > self.last_iter['num'], 'Please avoid time travelling'
        current_iter = {'num': iter_num, 'time': time.time()}
        current_time_per_iter = \
            (current_iter['time'] - self.last_iter['time']) / (current_iter['num'] - self.last_iter['num'])

        # Keep iter times at max 100
        self.iter_times.append(current_time_per_iter)
        if len(self.iter_times) > 100:
            self.iter_times.pop(0)

        # Remove extreme times
        iter_times_filtered = self.iter_times
        if len(self.iter_times) > 3:
            low, high = np.percentile(self.iter_times, [10, 90])
            iter_times_filtered = [t for t in self.iter_times if low <= t <= high]

        self.last_iter = current_iter
        if iter_num >= self.total_iters - 1:
            return self.elapsed_time()
        return format_time_interval(np.mean(iter_times_filtered) * (self.total_iters - current_iter['num']))

    def elapsed_time(self):
        return format_time_interval(time.time() - self.start_time)