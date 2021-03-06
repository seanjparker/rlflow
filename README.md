# RLFlow

RLFlow is a graph-net based neural network optimisation extension of [TASO](https://github.com/jiazhihao/TASO).
The basic idea was to substitute TASO's *cost-based backtracking search* 
and replace it with a RL-based optimisation. Potentially, this should be
able to generalise to large unseen graphs and find better performing solutions
than the backtracking search.

RLFlow consists of three main parts:

1. An extension of the TASO library exposing a low-level RL environment
2. A gym-style high-level environment written in Python, interacting with the RLFlow environment
3. Extensible agent API interacting with this environment, iteratively applying graph substitutions to a target graph.

The project also contains model-based and model-free agents as well as scripts to perform
different types of analysis of the agents' performance.

As of today, RLFlow is able to learn generalisable policies for subgraph transformation that
match the performance of the original TASO implementation in ConvNets and slightly outperform TASO on Transformer style
networks.

## Setup

RLFlow interacts with TASO and thus depends on the TASO library. TASO on the other
hand depends on a working CUDA and CuDNN installation.

### Installing Python

The experiments used **Python 3.7.0**, though later versions might also work. It is strongly
advised to use a virtual environment such as [pyenv](https://github.com/pyenv/pyenv-installer)
for installation.

Most dependencies should get automatically installed with the libraries. To exactly mirror the 
installed packages of the current experiment environment, run

```bash
pip install -r pip.freeze
```

_Note: This command will fail if you have not yet installed the TASO package._

<details>
  <summary>Experiments prerequisites</summary>
  <h5>Versions</h5>
    <ul>
    <li> Python: 3.7.0 </li>
    <li> TensorFlow: 2.3.2 </li>
    <li> Cuda: 10.1 (Used 18.04 (LTS), the cuBLAS library needs to be manually installed, it doesn't get installed into `/usr/local/cuda10-1` by default) </li>
    <li> CuDNN: 7.6.5.32-1+cuda10.1 </li>
    <li> Nvidia Driver: >= 418.39 </li>
    </ul>
</details>

### Installing CUDA

CUDA can be obtained from [the NVIDIA website](https://developer.nvidia.com/cuda-downloads). 
CuDNN can be downloaded after registering for a NVIDIA developer account.

At the time of the initial experiments, we used **Cuda 10.1** and **CuDNN 7.6.5.32-1+cuda10.1**.

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

At the time of the experiments, we were working on commit `0b01194`:

```bash
git checkout 0b01194
```

We also had to make slight adjustments to the base library for our interaction with
TASO to work. For these, apply the two patches to the TASO library:

```bash
pushd /path/to/taso
cp /path/to/xflowrl/TASO_*.patch .
patch -p0 < TASO_01_get_pointer_address.patch
patch -p0 < TASO_02_run_memorysafe.patch
patch -p0 < TASO_03_get_costs.patch
popd
```

In `TASO/include/ops.h` there are two constants that might have to be changed, 
depending on the available GPU memory. One is `MAX_TENSOR_SIZE` and one is
`WORK_SPACE_SIZE`. The first patch changes these to 128MB and 512MB, respectively.
This should work on the GPU on the `ran` machine of the Cambridge University Computer Lab
(GeForce GTX 780).

After applying these changes, just follow the [installation instructions](https://github.com/jiazhihao/TASO/blob/master/INSTALL.md).

### Installing RLFlow

The installation of RLFlow is very similar to that of TASO. First, make sure the environment
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

## Updating RLFlow

After making changes to the RLFlow code, it might have to be rebuilt. 

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

## Running RLFlow

There are example scripts that instantiate a RLFlow instance and train
on a set of graphs: 

```bash
python xflowrl/examples/run_onnx_xflowrl.py
```
## Citation
TBD