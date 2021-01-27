# XFLOWRL

XFLOWRL is a graph-net based neural network optimisation extension of [TASO](https://github.com/jiazhihao/TASO).
The basic idea was to substitute TASO's *cost-based backtracking search* 
and replace it with a RL-based optimisation. Potentially, this should be
able to generalise to large unseen graphs and find better performing solutions
than the backtracking search.

XFLOWRL consists of three main parts:

1. An extension of the TASO library exposing a low-level RL environment
2. A gym-style high-level environment written in Python, interacting with the XFLOWRL environment
3. A hierarchical PPO agent interacting with this environment, iteratively applying graph substitutions to a target graph.

As of today, XFLOWRL is able to learn generalisable policies for subgraph transformation that
match the performance of the original TASO implementation. However, training takes
a long time, and the results are just as good as TASO, not better.

Some experiments hint that TASO is actually finding a close to optimal solution so
that it seems infeasible to improve on this. This will be discussed in the [EXPERIMENTS.md](EXPERIMENTS.md) section.

## Setup

XFLOWRL interacts with TASO and thus depends on the TASO library. TASO on the other
hand depends on a working CUDA and CuDNN installation.

### Installing Python

The experiments used **Python 3.6.10**, though later versions might also work. It is strongly
advised to use a virtual environment such as [pyenv](https://github.com/pyenv/pyenv-installer)
for installation.

Most dependencies should get automatically installed with the libraries. To exactly mirror the 
installed packages of the current experiment environment, run

```bash
pip install -r pip.freeze
```

_Note: This command may fail if you have not yet installed the TASO package._

<details>
  <summary>2021 Experiments Requirements</summary>
  ## Requirements
- Python: 3.7.0
- TensorFlow: 2.3.2
- Cuda: 10.1 (Used 18.04 (LTS), the cuBLAS library needs to be manually installed, it doesn't get installed into `/usr/local/cuda10-1` by default)
- CuDNN: 7.6.5.32-1+cuda10.1
- Nvidia Driver: >= 418.39
</details>

### Installing CUDA

CUDA can be obtained from [the NVIDIA website](https://developer.nvidia.com/cuda-downloads). 
CuDNN can be downloaded after registering for a NVIDIA developer account.

At the time of the initial experiments, we used **Cuda 10.2** and **CuDNN 7.6.5.32-1**.

Please also install the development files, i.e. **libcudnn-dev**.

This project assumes that CUDA is available at `/usr/local/cuda`, which should
be the default.

### Installing TASO

Obtain TASO from [their github repository](https://github.com/jiazhihao/TASO).

Most importantly, set the `TASO_HOME` variable:

```bash
export TASO_HOME=/path/to/taso
``` 

Also, for some builds to succeed I had to set the `LD_LIBRARY_PATH` variable:

```bash
export LD_LIRBARY_PATH=/usr/local/lib
```

At the time of the experiments, we were working on commit `a310b60f`:

```bash
git checkout a310b60f
```

We also had to make slight adjustments to the base library for our interaction with
TASO to work. For these, apply the two patches to the TASO library:

```bash
pushd /path/to/taso
cp /path/to/xflowrl/TASO_*.patch .
patch -p0 < TASO_01_get_pointer_address.patch
patch -p0 < TASO_02_run_memorysafe.patch
popd
```

In `TASO/include/ops.h` there are two constants that might have to be changed, 
depending on the available GPU memory. One is `MAX_TENSOR_SIZE` and one is
`WORK_SPACE_SIZE`. The first patch changes these to 128MB and 512MB, respectively.
This should work on the GPU on the `ran` machine of the Cambridge University Computer Lab
(GeForce GTX 780). On larger GPUs, this could be reverted to the original settings
(512MB and 1GB).

After applying these changes, just follow the [installation instructions](https://github.com/jiazhihao/TASO/blob/master/INSTALL.md).

### Installing XFLOWRL

The installation of XFLOWRL is very similar to that of TASO. First, make sure the environment
variables are set:

```bash
export TASO_HOME=/path/to/taso
export LD_LIRBARY_PATH=/usr/local/lib
```

Then, go to `taso_ext`, create a `build` folder, and run `cmake`:

```bash
cd taso_ext
mkdir build
cd build
cmake ..
make -j 10
sudo make install
```

The last two commands compile the extension and install it to the `LD_LIBRARY_PATH`.

At this path, you should now have two `.so` files:

```plain
ls -alp /usr/local/lib/
# ...
# -rw-r--r--  1 root root   448920 Apr  7 13:16 libtaso_rl.so
# -rw-r--r--  1 root root  1173704 Apr  7 13:39 libtaso_runtime.so
# ...
```

The extension has been built. Now the python package can be installed:

```
# In the project root:
pip install -e .
```

## Updating XFLOWRL

After making changes to the XFLOWRL code, it might have to be rebuilt. 

Any changes to the `taso_ext` folder (i.e. `graph_feedback.cc` or
`graph_feedback.h`) need a fresh cmake build:

```bash
cd taso_ext/build
make -j 10
sudo make install
```

Any changes to cython files (i.e. files in `xflowrl/_cython`) will need to be
re-compiled:

```
# In the project root:
pip install -e .
```

Any changes to pure Python files should work without re-installing, thanks to
the `pip install -e .` command. If you used `python setup.py install`, like TASO
does, the install command has to be executed after any changes.

## Running XFLOW RL

There are example scripts that instantiate a XFLOWRL instance and train
on a set of graphs: 

```bash
python xflowrl/examples/run_onnx_xflowrl.py
```

Details of this are explained in the [EXPERIMENTS.md](EXPERIMENTS.md) section.