import sys


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='='):
    """Prints a progress bar.

    :param int iteration: current iteration number (starting from 0)
    :param int total: total number of iterations
    :param str prefix: prefix to print before the progress bar
    :param str suffix: suffix to print after the progress bar

    :Example:

    >>> print_progress_bar(4, 100, prefix='Dataset A', suffix='images loaded')
    Dataset A [==>......................] 5/100 (5.0%) images loaded

    It can be easily integrated in an existing for loop by using enumerate():

    >>> for i, file in enumerate(files):
    >>>     print_progress_bar(i, len(files))
    >>>     process_file(file) # ...
    """
    total = max(total - 1, 1)
    
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total) + 1
    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r {} [{}] {}/{} ({}%) {}'.format(prefix, bar, iteration, total, percent, suffix), end='\r')
    if iteration >= total:  # Print new line on completion
        print(' ')
    
    sys.stdout.flush()

class ProgressBar:
    def __init__(self, total):
        pass