import sys
import os
import time

from joblib import Parallel, delayed


def find_prime_numbers(upto):
    prime_list = [2, ]
    i = 2
    while len(prime_list) < upto:

        for j in range(2, i):
            if i % j == 0:
                break
            if j == i - 1:
                prime_list += [i]
        i += 1
    return prime_list[-1]


def main(proc_no, out_path='results/', array_id=1):
    proc_no = int(proc_no)

    # find first x prime numbers
    x = 5000

    # 120 runs = 24 * 5
    start = time.time()
    for _ in range(5):
        _ = find_prime_numbers(x)
    ref_time = 24 * (time.time() - start)
    print('Single process:     %.2f s' % ref_time)

    # 120 parallellized runs
    start = time.time()
    _ = (
        Parallel(n_jobs=proc_no)
        (delayed(find_prime_numbers)(i) for i in 120*[x])
    )
    quicker_time = time.time() - start

    print('Parallel processes: %.2f s' % quicker_time)
    print('Extra time:         %.2f s' % (quicker_time - ref_time / proc_no))
    print()
    print('Speedup: %.2f' % (ref_time/quicker_time))


if __name__ == "__main__":

    # (cluster) keyword arguments: array_id and out_path
    kwargs = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, val = arg.split('=')
            if key == 'array_id':
                kwargs[key] = int(val)
            else:
                kwargs[key] = val

    # arguments: anything needed for this script
    args = [arg for arg in sys.argv[1:] if not ('=' in arg)]

    main(*args, **kwargs)
