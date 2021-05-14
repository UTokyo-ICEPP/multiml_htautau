from contextlib import contextmanager
import time


@contextmanager
def timer(reg, name):
    time_info = [
        time.process_time(),  # CPU time
        time.perf_counter(),  # real time
    ]

    yield time_info

    time_info[0] = time.process_time() - time_info[0]
    time_info[1] = time.perf_counter() - time_info[1]

    # print(f'total cpu/real time of [{name}] = {time_info[0]:.2f} s / {time_info[1]:.2f} s')
    reg[name] = time_info
