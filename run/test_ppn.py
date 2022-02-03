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


def main(job_no, out_path='results/', array_id=1):
    job_no = int(job_no)

    start = time.time()
    for _ in range(job_no):
        find_prime_numbers(1000)
    ref_time = time.time() - start

    start = time.time()
    _ = (
        Parallel(n_jobs=job_no)
        (delayed(find_prime_numbers)(i) for i in job_no*[1000])
    )
    quicker_time = time.time() - start
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
