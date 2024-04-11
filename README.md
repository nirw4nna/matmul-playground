# matmul-playground
The goal of this project is to provide a playground to develop and test different GEMM algorithms against well-known implementations like Intel MKL and OpenBLAS.

All the kernels make the following assumptions:
- the core computation is: $C := A^T B$ where $C$ is $m$ x $n$, $A$ is $m$ x $k$ and $B$ is $k$ x $n$
- the $C$ matrix is zero-initialized (this assumption can be relaxed)
- the matrices are all stored in **column-major order**
- no error checking on the input matrices, meaning the shapes must be compliant
- the total number of FLOPS of each GEMM is computed as $2 * m * n * k$

## Quickstart
Before you start make sure to have the following tools installed and available:
- A C++ compiler that supports C++17
- Python >= 3.x
- Intel MKL (can be found [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html))

The code has been tested with the following setup:
- Linux 5.15 x86_64
- gcc 11.4.0
- Python 3.10.12
- MKL 2024.1 (if you are using a different version you can specify that by setting the environment variable `MKLROOT=/path/to/intel/oneapi/mkl/version` before executing the script)


Once all the dependencies are installed:
1. Create a new Python virtual environment and source it
```
python3 -m venv venv
source venv/bin/activate
```
2. Install Python requirements
```
pip install -r requirements.txt
```
3. (Optional) Make `driver.py` executable
```
chmod +x driver.py
```
4. Start benchmarking the kernels
```
./driver.py
```
It's also possible to use this tool for **finetuning** the parameters that control the cache blocking (`MC`, `NC` and `KC`). To do so you can specify the option `--finetune=[MC | NC | KC]`, the script will compute some defaults based on the hardware you are currently on and then with the `--finetuning-range=start,stop,step` option you can test different values, using the so-called 'analytical' parameters as a baseline.

### Notes for non-Linux users
If you are running on a different operating system remember to update the `_CacheParams` values inside `cache.py`. This is because currently I'm using `sysconf` to fetch all the information I need and I don't know a cross-platform way of doing this.

If you don't know how to fetch this information you can start from the default values provided in `platform.h` and use the finetuning mechanism to do search.


### Notes on benchmarking
The numbers you will get from this tool are not ment to be 100% accurate but they are a good approximation. This means that one should not fixate on the numbers per-se but rather on closing the gap with MKL.

The theoretical peak performance (the red dotted line) is computed assuming the CPU is spending 100% of its time computing FMAs at its peak clock frequency. This is obviously never going to happen, it's only there to put an upper-bound on the plot.

Regarding the actual methodology used, it should be 'fair' for the basic use case but inconsistencies can arise if you're using very large matrices or you run the benchmark for too long.
There are several reasons for this:
- The CPU, especially if you're on a laptop or have a poor cooler, can start throttling
- Cache behavior is not deterministic
- The OS can be unpredictable

To try and prevent some of these things one can:
- Disable turbo boost
```
echo 1 >> /sys/devices/system/cpu/intel_pstate/no_turbo
```
- Use the 'performance' CPU scaling governor
```
sudo cpupower frequency-set -g performance
```
- Increase process priority (requires to run the program with `sudo nice`)
- Disable address space randomization
```
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
```

Out of these, the two most important ones are probably the first two since they make sure that the CPU keeps a *constant* frequency across the board. Then, if you want to benchmark a plethora of different sizes just remember to split them accross multiple runs, to keep the OS from preempting your process.

## Supported Features
| Extension | Data type | Supported |
|-----------|-----------|-----------|
| `AVX`     | FP32      | no        |
| `AVX2`    | FP32      | yes       |
| `AVX512`  | FP32      | no        |  
