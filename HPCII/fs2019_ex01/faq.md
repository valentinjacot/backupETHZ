# Frequently Asked Questions
This collection of frequently asked questions shall be used a resource of information before posting a question on [piazza](https://piazza.com/class/jromqmagjbd30c).

_Please consult this FAQ pool first before you ask a question. In case your question is not adressed in this collection, post it in [piazza](https://piazza.com/class/jromqmagjbd30c). Do **not** adress questions to the TAs unless there is a reason to do so._

## Table of Contents
- [Eigen library on Euler](#the-eigen-library-on-the-euler-cluster)
- [Installing Python packages on Euler](#installing-python-packages-on-euler)
- [How to use an interactive node on Euler](#how-to-use-an-interactive-node-on-euler)
- [Working with Piz Daint supercomputer](#working-with-piz-daint-supercomputer)
- [Git tutorial](#git-tutorial)

## The Eigen library on the Euler cluster

The Eigen library is a templated library and does not provide shared object
files but only header files.  In general, if you need additional libraries, you
should use
```
module load <package>
```
You can find the available `packages` by executing
```
module avail
```

In case of the Eigen library, the location of the header files is not added to
your environment when loading the library.  Consequently, the compiler can not
find the headers and will complain at compile time.  You can fix the problem by
specifying the include path manually
```
g++ -I/cluster/apps/eigen/3.2.1/x86_64/gcc_4.8.2/serial/include/eigen3 mycode.cpp
```
If you plan to use the Eigen library more often, you can set the search path of
the `gcc` compiler in your `$HOME/.bashrc` file by adding the line
```
export CPATH=/cluster/apps/eigen/3.2.1/x86_64/gcc_4.8.2/serial/include/eigen3
```
at the end of the file.  This will set the search path for both `gcc` and
`g++`.  But remember, this will always load the Eigen version `3.2.1`.  You
must manually update the path if you want to compile using another version.


## Installing Python packages on Euler

If you need to install a Python package on Euler, you must install it in your
`$HOME` due to write permissions.  For example, you can use the following
command to install the `beautifulsoup` package with python2.7:
```
python -m pip install --user bs4
```
After that you can use `import bs4` in your Python code.


## How to use an interactive node on Euler

Sometimes it is convenient to checkout an interactive node on Euler, rather
than submitting jobs to queue.  Interactive means that you will get back a
shell with the requested resources and you can run your code interactively.
You can request an interactive node on Euler with the following command:
```
bsub -n 24 -W 03:00 -Is /bin/bash
```
As you already know, `bsub` will place your request on the job scheduler queue.
Instead of executing your program, you request a BASH shell in which you will
work once the requested resources have been allocated for you.  The above
example allocates 24 cores for 3 hours.  When you submit this request, you will
have to wait until the job has been allocated for you.  If the cluster is busy,
you might have to wait a while until you receive your interactive shell.

If you want a Haswell node, you can specify it explicitly
```
bsub -n 24 -W 03:00 -R "select[model==XeonE5_2680v3]" -Is /bin/bash
```





## Working with Piz Daint supercomputer

You will be granted access to the main supercomputer in the CSCS (Swiss Supercomputer Center in Lugano): Piz Daint. The credentials (a username and a password) will be distributed to you when time comes.
Piz Daint consists of two parts: a GPU-enabled cluster Cray XC50 and a multicore cluster Cray XC40 (former Piz Dora). You will only have to use the GPU part, that offers 5320 nodes with an Intel Xeon E5-2690 v3 CPU and an NVIDIA Tesla P100 GPU.
Detailed information about the machines and how to use them can be found on the [CSCS web site](https://www.cscs.ch/computers/dismissed/piz-daint-piz-dora/m) as well as the [user portal](https://user.cscs.ch/)

### Log in

Computational resources of CSCS are only accessible from the internal network,
so you will have to first login in frontend cluster Ela:
```
$ ssh your_username@ela.cscs.ch
```
Now you can access a login node of Piz Daint with the command `ssh daint`:
```
your_username@ela1:~> ssh daint
```
Once on Piz Daint, you can run the command "hostname" to verify that you are indeed on the desired machine.

### Compilation

Piz Daint uses module system to control the environment and to simplify usage of different tools and libraries. In order to compile CUDA code, you will have to load the following modules:
```
$ module load daint-gpu
$ module swap PrgEnv-cray PrgEnv-gnu
$ module load cudatoolkit
```

### Job submission

All CSCS systems use the SLURM batch system for the submission,
control and management of user jobs. SLURM provides a rich set of features for organizing your workload and provides an extensive array of tools for managing your resource usage, however in normal interaction with the batch system you only need to know a few basic commands:

* `srun` - submit an executable or run an executable from within the sbatch script
* `sbatch` - submit a batch script
* `squeue` - check the status of jobs on the system
* `scancel` - delete one of your jobs from the queue (e.g. scancel JOB_PID)

From the Piz Daint login node you can submit a hostname task to the jobs queue: `srun -C gpu hostname`. Use `squeue -u your_username` to check the status of your job, or squeue to see your job relative to all the other jobs in the queue. After the job is executed, you will see the output of the command on the screen. Observe that the hostname is now different: you see the name of the compute node which has executed your task. In order to have a more precise control over the parameters of your job (i.e. runtime,
number of nodes or tasks submitted, output files, etc.) you can prepare a SLURM script and submit it with `sbatch` command. A common script looks like this:
```
#!/ bin/ bash -l
# SBATCH --job - name = job_name
# SBATCH --time =01:00:00
# SBATCH --nodes =1
# SBATCH --ntasks -per - node =1
# SBATCH -- constraint =gpu

srun ./ your_executable
```
See more details about job submission [here](https://user.cscs.ch/access/running/).

### Interactive mode

You can also run your jobs on compute nodes interactively. To do so,
run `salloc -C gpu` and wait (typically a few minutes) until you get a shell prompt. Run `srun hostname` once again to make sure that you are on a compute node. Now you can use the command `srun your_application` without having to submit it in the queue.

### IMPORTANT NOTE

Note that the total node-hour budget for the students is 1600 node-hours
per month, which is about **20 node-hours per month per person**. The limits are not hard, i.e. you can exceed them, but the priority of your jobs will decrease and the queue waiting time will increase significantly. Be considerate about resource usage, especially don’t overuse interactive mode! 

**If you have any questions or issues, please DO NOT contact the CSCS help desk, always address your question on [piazza](https://piazza.com/class/jromqmagjbd30c) to the Teaching Assistants!**








