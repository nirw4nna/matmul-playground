#!/usr/bin/env python3

import os
import sys

NUM_THREADS = os.getenv('NUM_THREADS', '1')
IS_PARALLEL = int(NUM_THREADS) > 1

MKLROOT = os.getenv('MKLROOT', '/opt/intel/oneapi/mkl/2024.1')
os.environ['MKLROOT'] = MKLROOT
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['GOTO_NUM_THREADS'] = NUM_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_THREADS

if IS_PARALLEL:
    # Set OMP policy to not move threads around
    # Note: this appears to harm performance for some reason (probably due MKL/OpenMP interactions).
    #       Try these without MKL at some point.
    
    # os.environ['OMP_PROC_BIND'] = 'close'
    # os.environ['OMP_PROC_BIND'] = 'true'
    # os.environ['OMP_PLACES'] = 'cores'
    pass
else:
    # Set CPU affinity
    os.sched_setaffinity(0, {0})

from typing import Set, Tuple, List
from jinja2 import Environment, FileSystemLoader
from importlib import import_module
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import argparse
import psutil
from cache import compute_auto


# These are the flags I usually use, to my knowledge they should enable the highest possible optimisations.
# At some point I think it would be good if these were not hard-coded but user-defined (e.g. using environment variables).
CXX = 'g++'
OPTIM_FLAGS = '-march=native -mtune=native -Ofast -ffp-contract=fast -funroll-loops -flto=auto -fuse-linker-plugin'
MKL_FLAGS = f'-DMKL_ILP64 -I{MKLROOT}/include'

CXXFLAGS = f'-std=c++17 {MKL_FLAGS} -Wall -Wextra -Wshadow -Wformat -Wnoexcept -Wcast-qual -Wunused -Wdouble-promotion \
-Wlogical-op -Wcast-align -fno-exceptions -fno-rtti {"-fopenmp " if IS_PARALLEL else ""}{OPTIM_FLAGS}'

MKL_LDFLAGS = f'-Wl,--start-group {MKLROOT}/lib/libmkl_intel_ilp64.a {MKLROOT}/lib/{"libmkl_sequential.a" if not IS_PARALLEL else "libmkl_gnu_thread.a"} \
{MKLROOT}/lib/libmkl_core.a -Wl,--end-group'

LDFLAGS = f'{MKL_LDFLAGS} -lm'

TURBO_CLOCK = psutil.cpu_freq().max / 1000 # GHz
FLOPS_CYCLE = 32 # Assuming fp32

REFERENCE_KERNEL = 'gemm_mkl'

RNG = np.random.default_rng()


def random_ndarray(n_rows: int, n_cols: int) -> np.ndarray:
    return (RNG.random((n_rows, n_cols), dtype=np.float32) + 100).astype(np.float32)


def compile_and_load(kernels: List[str], param: str, param_start: int, param_stop: int, param_step: int) -> Set[str]:
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
    
    param : `str`
        Extra argument that has to be passed to the compiler

    param_start, param_stop, param_step : `int`
        Range of values that `param` takes. A new module is generated
        for each value of `param` and is identified by `{name}_{param}={value}`.

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
    
    if len(kernels) > 0:
        found_kernels = {REFERENCE_KERNEL}
        for kernel in kernels:
            if kernel in base_names:
                found_kernels.add(kernel)
            else:
                raise RuntimeError(f'{kernel} not found')

        base_names = found_kernels

    # Load the jinja template
    env = Environment(loader=FileSystemLoader('./'))
    bindings_template = env.get_template('_bindings.pytemplate')
    
    def _compile_and_load(objs: Set[str], compiler_arg: str = ''):
        has_arg = len(compiler_arg) > 0
        extra_flag = f'-D{compiler_arg}' if has_arg else ''
        identifier = f'_{compiler_arg.replace("=", "_")}' if has_arg else ''

        # Compile a shared object .so for each file .c with the same base name
        for obj in objs:
            shared_obj = f'{obj}{identifier}.so'
            module = f'_{obj}{identifier}'
            py_module = f'{module}.py'

            if os.path.exists(shared_obj):
                print(f'rm {shared_obj}')
                os.remove(shared_obj)
            
            if os.path.exists(py_module):
                print(f'rm {py_module}')
                os.remove(py_module)

            cmd = f'{CXX} {CXXFLAGS} {extra_flag} -fPIC -shared kernels/{obj}.cpp -o {shared_obj} {LDFLAGS}'
            print(cmd)
            if not os.system(cmd) == 0:
                raise RuntimeError(f'Error compiling {shared_obj}')
            
            # Make sure the output of the compiler is flushed to disk
            os.sync()

            data = {
                'shared_obj': shared_obj,
                'kernel': obj
            }

            generated_code = bindings_template.render(data)
            with open(py_module, 'wt') as f:
                f.write(generated_code)
            
            # Flush everything to disk
            os.sync()

            # Try to load the generated module
            globals()[f'{obj}{identifier}'] = import_module(module)

            
    print(f'Compiling {base_names}:\n')

    if not param == 'None':
        # Compile and load the reference kernel only once
        objs = {obj for obj in base_names if obj != REFERENCE_KERNEL}
        for v in range(param_start, param_stop, param_step):
            _compile_and_load(objs, f'{param}={v}')
        _compile_and_load({REFERENCE_KERNEL})
    else:
        _compile_and_load(base_names)

    return base_names


def benchmark(plot: str, n_repeat: int, kernels: List[str],
              size_start: int, size_stop: int, size_step: int,
              finetune_param: str,
              finetuning_start: int, finetuning_stop: int, finetuning_step: int):
    finetuning = not finetune_param == 'None'

    if finetuning:
        analytical_values = compute_auto(dtype_size=4, mr=8, nr=12, unroll=4)
        print(f'Analytical values: {analytical_values}')
        val = analytical_values[finetune_param]
        finetuning_start += val
        finetuning_stop += val
        
    base_names = compile_and_load(kernels, finetune_param,
                                  finetuning_start, finetuning_stop, finetuning_step)

    kernel_sizes = {}
    kernel_gflops = {}

    
    def _run_benchmark(m: int, n: int, k: int):
        lda = m; ldb = k; ldc = m

        a = random_ndarray(m, k)
        b = random_ndarray(k, n)
        
        flops = 2 * m * n * k

        # Compute C_expected  
        c_expected = np.zeros((m, n)).astype(np.float32)
        ref_gemm = getattr(globals()[REFERENCE_KERNEL], REFERENCE_KERNEL)
        ref_gemm(m, n, k, a, lda, b, ldb, c_expected, ldc)
        
        def _benchmark_kernel(module: str, kernel: str):
            gemm = getattr(globals()[module], kernel)
                
            # Take the best latency out of n_repeat samples
            latency = float('+inf')

            for _ in range(n_repeat):
                c = np.zeros((m, n)).astype(np.float32)

                cur_latency = gemm(m, n, k, a, lda, b, ldb, c, ldc)

                latency = latency if cur_latency >= latency else cur_latency

                if not np.allclose(c_expected, c, rtol=1.e-5, atol=1.e-5, equal_nan=True):
                    raise RuntimeError(f'"{kernel}" evaluation failed! M={m}, N={n}, K={k}')

            if module not in kernel_sizes:
                kernel_sizes[module] = []

            kernel_sizes[module].append(k)
            
            if module not in kernel_gflops:
                kernel_gflops[module] = []
            
            kernel_gflops[module].append(flops / (1e9 * latency))


        for kernel in base_names:
            if finetuning:
                for v in range(finetuning_start, finetuning_stop, finetuning_step):
                    # Benchmark the reference kernel only once
                    if kernel == REFERENCE_KERNEL:
                        _benchmark_kernel(REFERENCE_KERNEL, REFERENCE_KERNEL)
                        break

                    _benchmark_kernel(f'{kernel}_{finetune_param}_{v}', kernel)
            else:
                _benchmark_kernel(kernel, kernel)
    

    print(f'\nBenchmarking kernels {base_names} REPEAT={n_repeat} m=n=k=[{size_start}:{size_stop}:{size_step}]. This can take some time...\n')
    for k in range(size_start, size_stop, size_step):
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
    
    if not IS_PARALLEL:
        # Plot the theoretical upper-bound only in single-threaded mode otherwise it's too unrealistic
        plt.axhline(y=(TURBO_CLOCK * FLOPS_CYCLE), color='r', linestyle='--')

    # TODO: if parallel plot also Gflops / thread
    plt.xlabel('m = n = k')
    plt.ylabel('GFLOPS')
    plt.title(f'Performance ({"single-threaded" if not IS_PARALLEL else f"NUM_THREADS={NUM_THREADS}"})')

    if plot == 'bar':
        width = 0.25
        multi = 0
        labels = kernel_sizes[REFERENCE_KERNEL]
        x = np.arange(len(labels))
        for kernel, gflops in kernel_gflops.items():
            rects = plt.bar(x + (width * multi), gflops, width, label=kernel)
            multi += 1
            plt.bar_label(rects, padding=3, fmt='%.1f')
        
        plt.xticks(x + (width * (multi - 1)) * 0.5, labels)

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
    cli_parser.add_argument('--kernels', type=str, default='',
                            help='comma-separated list of kernels to benchmark, if empty benchmark all of them')
    cli_parser.add_argument('--range', type=str, default='2048,2049,1',
                            help='size of the matrices (m=n=k) to be tested in the form "start,stop,step"')
    cli_parser.add_argument('--finetune', choices=['KC', 'MC', 'NC', 'None'], default='None',
                            help='fine-tune the selected parameter')
    cli_parser.add_argument('--finetuning-range', type=str, default='-50,50,10',
                            help='range of values to test for the selected parameter (the baseline is computed analytically) in the form "start,stop,step"')
    
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

    def _parse_range(range: str) -> Tuple[int, int, int]:
        range_params = range.split(',')
        return (int(param, base=10) for param in range_params)

    start, stop, step = _parse_range(args.range)
    assert start >= 0 and stop >= 0 and step >= 1

    finetune_start, finetune_stop, finetune_step = _parse_range(args.finetuning_range)
    assert finetune_start < finetune_stop and finetune_step >= 1

    kernels = [kernel for kernel in args.kernels.split(',') if not kernel == '']
    
    benchmark(args.plot, args.repeat, kernels,
              start, stop, step,
              args.finetune,
              finetune_start, finetune_stop, finetune_step)

