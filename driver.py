#!/usr/bin/env python3

import os
import sys

MKLROOT = os.getenv('MKLROOT', '/opt/intel/oneapi/mkl/2024.1')
os.environ['MKLROOT'] = MKLROOT
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Set CPU affinity
os.sched_setaffinity(0, {0})

from typing import Set
from jinja2 import Environment, FileSystemLoader
from importlib import import_module
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import argparse
import psutil


# This are the g++ flags I usually use, to my knowledge these should enable the highest level of optimizations.
# At some point I think it would be good if these were not hard-coded but user-defined (e.g. using environment variables).
CXX = 'g++'
OPTIM_FLAGS = '-march=native -mtune=native -Ofast -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin'
MKL_FLAGS = f'-DMKL_ILP64 -I{MKLROOT}/include'
CXXFLAGS = f'-std=c++17 {MKL_FLAGS} -Wall -Wextra -Wshadow -Wformat -Wnoexcept -Wcast-qual -Wunused -Wdouble-promotion \
-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti {OPTIM_FLAGS}'
MKL_LDFLAGS = f'-Wl,--start-group {MKLROOT}/lib/libmkl_intel_ilp64.a {MKLROOT}/lib/libmkl_sequential.a {MKLROOT}/lib/libmkl_core.a -Wl,--end-group'
LDFLAGS = f'{MKL_LDFLAGS} -lm'

TURBO_CLOCK = psutil.cpu_freq().max / 1000 # GHz
FLOPS_CYCLE = 32 # Assuming fp32

REFERENCE_KERNEL = 'gemm_mkl'

RNG = np.random.default_rng()


def random_ndarray(n_rows: int, n_cols: int) -> np.ndarray:
    return (RNG.random((n_rows, n_cols), dtype=np.float32) + 100).astype(np.float32)


def compile_and_load(kernel: str = '') -> Set[str]:
    """
    Compile the provided kernel, generate the ctypes binding and load the generated module.
    
    To be considered valid each kernel must:
        - Start with `gemm_`
        - Be declared in a `.h` file.
        - Be implemented in a `.cpp` file with the same name as the `.h` file (except for the extension, of course)
    
    Note: all kernels must have the same signature, this can be found for example in `kernels/gemm_jpi.h`.

    
    Parameters
    ----------
    kernel : `str`
        The kernel to be loaded or `''`, if no kernel is specified load everything from `kernels/`

    Returns
    -------
    kernels : `Set[str]`
        The list of loaded kernels, without file extension
    """
    base_names = set()
    all_files = set(os.listdir('kernels/'))

    for file in all_files:
        if file.endswith('.h') and file.startswith('gemm_'):
            source_name = file.replace('.h', '.cpp')
            assert source_name in all_files
            base_names.add(file.replace('.h', ''))

    if REFERENCE_KERNEL not in base_names:
        raise RuntimeError(f'Reference kernel {REFERENCE_KERNEL} not found')
    
    if len(kernel) > 0:
        if kernel in base_names:
            base_names = {kernel, REFERENCE_KERNEL}
        else:
            raise RuntimeError(f'{kernel} not found')

    print(f'Compiling {base_names}:\n')
    # Compile a shared object .so for each file .c with the same base name
    for obj in base_names:
        shared_obj = f'{obj}.so'
        binding = f'_{obj}.py'
        if os.path.exists(shared_obj):
            print(f'rm {shared_obj}')
            os.remove(shared_obj)
        
        if os.path.exists(binding):
            print(f'rm {binding}')
            os.remove(binding)

        cmd = f'{CXX} {CXXFLAGS} -fPIC -shared kernels/{obj}.cpp -o {shared_obj} {LDFLAGS}'
        print(cmd)
        if not os.system(cmd) == 0:
            raise RuntimeError(f'Error compiling {shared_obj}')


    # Load the jinja template
    env = Environment(loader=FileSystemLoader('./'))
    bindings_template = env.get_template('_bindings.pytemplate')
    
    for kernel in base_names:
        # Each template requires two things:
        #   - {{ shared_obj }} the name of the .so file (with the extension)
        #   - {{ kernel }} the name of the actual kernel to call
        data = {
            'shared_obj': f'{kernel}.so',
            'kernel': kernel
        }
        generated_code = bindings_template.render(data)
        module = f'_{kernel}'
        with open(f'{module}.py', 'wt') as f:
            f.write(generated_code)
        
        # Flush everything to disk
        os.sync()

        # Try to load the generated module
        globals()[kernel] = import_module(module)

    return base_names


def benchmark(plot: str, n_repeat: int, target_kernel: str,
              start: int, stop: int, step: int):
    base_names = compile_and_load(target_kernel)

    kernel_sizes = {}
    kernel_gflops = {}


    def _run_benchmark(m: int, n: int, k: int):
        # Leading dimension of A
        lda = m
        # Leading dimension of B
        ldb = k
        # Leading dimension of C
        ldc = m
        a = random_ndarray(m, k)
        b = random_ndarray(k, n)
        
        flops = 2 * m * n * k

        # Compute C_expected  
        c_expected = np.zeros((m, n)).astype(np.float32)
        ground_truth_gemm = getattr(globals()[REFERENCE_KERNEL], REFERENCE_KERNEL)
        ground_truth_gemm(m, n, k, a, lda, b, ldb, c_expected, ldc)
        
        for kernel in base_names:
            gemm = getattr(globals()[kernel], kernel)
            
            # Take the best latency out of n_repeat samples
            latency = float('+inf')

            for _ in range(n_repeat):
                c = np.zeros((m, n)).astype(np.float32)
                cur_latency = gemm(m, n, k, a, lda, b, ldb, c, ldc)

                latency = latency if cur_latency >= latency else cur_latency

                if not np.allclose(c_expected, c, rtol=1.e-5, atol=1.e-5, equal_nan=True):
                    raise RuntimeError(f'"{kernel}" evaluation failed! M={m}, N={n}, K={k}')

            if kernel not in kernel_sizes:
                kernel_sizes[kernel] = []

            kernel_sizes[kernel].append(k)

            if kernel not in kernel_gflops:
                kernel_gflops[kernel] = []

            kernel_gflops[kernel].append(flops / (1e9 * latency))
    

    print(f'\nBenchmarking kernels {base_names} REPEAT={n_repeat} m=n=k=[{start}:{stop}:{step}]. This can take some time...\n')
    for k in range(start, stop, step):
        m = k
        n = k
        _run_benchmark(m, n, k)
    
    print('Perfomance recap (GFLOPS):')
    recap = [['Kernel', 'Mean', 'Max', 'Min']]
    for kernel in kernel_sizes:
        gflops = kernel_gflops[kernel]
        mean = round(sum(gflops) / len(gflops), 1)
        max_val = round(max(gflops), 1)
        min_val = round(min(gflops), 1)
        recap.append([kernel, mean, max_val, min_val])

    print(tabulate(recap) + '\n')

    if plot == 'none':
        return
    
    plt.axhline(y=(TURBO_CLOCK * FLOPS_CYCLE), color='r', linestyle='--')
    plt.xlabel('m = n = k')
    plt.ylabel('GFLOPS')
    plt.title('Performance (single-threaded)')

    if plot == 'bar':
        width = 0.25
        multi = 0
        labels = kernel_sizes[REFERENCE_KERNEL]
        x = np.arange(len(labels))
        for kernel, gflops in kernel_gflops.items():
            rects = plt.bar(x + (width * multi), gflops, width, label=kernel)
            multi += 1
            plt.bar_label(rects, padding=3, fmt='%.1f')
        
        plt.xticks(x + (width * 0.5), labels)

    elif plot == 'line':
        for kernel in kernel_sizes:
            sizes = kernel_sizes[kernel]
            gflops = kernel_gflops[kernel]
            plt.plot(sizes, gflops, '.-', label=kernel)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(description='Run your favourite GEMM kernels against the current state of the art, how close can you get?')
    cli_parser.add_argument('--repeat', type=int, default=2,
                            help='how many times a kernels should be executed for each set of M, N, K parameters')
    cli_parser.add_argument('--plot', choices=['bar', 'line', 'off'], default='bar',
                            help='plot to display at the end')
    cli_parser.add_argument('--clean', action='store_true',
                            help='remove all the intermediate objects')
    cli_parser.add_argument('--kernel', type=str, default='',
                            help='kernel to benchmark, if empty benchmark all of them')
    cli_parser.add_argument('--range', type=str, default='2048,2049,1',
                            help='size of the matrices (m=n=k) to be tested in the form "start,stop,range"')
    args = cli_parser.parse_args()
    
    if args.clean:
        print('Cleaning workspace...')
        files = os.listdir()
        for f in files:
            if ((f.startswith('_gemm') and f.endswith('.py')) or
                 (f.startswith('gemm') and f.endswith('.so'))):
                print(f'rm {f}')
                os.remove(f)
        sys.exit()

    range_params = args.range.split(',')
    start, stop, step = (int(param, base=10) for param in range_params)
    assert start >= 0 and stop >= 0 and step >= 1

    benchmark(args.plot, args.repeat, args.kernel,
              start, stop, step)

