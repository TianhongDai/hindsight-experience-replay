# Hindsight Experience Replay (HER)
This is a pytorch implementation of [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). 

## Acknowledgement:
- [Openai Baselines](https://github.com/openai/baselines)

## Requirements
- python=3.5.2
- openai-gym=0.12.5 (mujoco200 is supported, but you need to use gym >= 0.12.5, it has a bug in the previous version.)
- mujoco-py=1.50.1.56 (~~**Please use this version, if you use mujoco200, you may failed in the FetchSlide-v1**~~)
- pytorch=1.0.0 (**If you use pytorch-0.4.1, you may have data type errors. I will fix it later.**)
- mpi4py

## TODO List
- [x] support GPU acceleration - although I have added GPU support, but I still not recommend if you don't have a powerful machine.
- [x] add multi-env per MPI.
- [x] add the plot and demo of the **FetchSlide-v1**.

## Instruction to run the code
If you want to use GPU, just add the flag `--cuda` **(Not Recommended, Better Use CPU)**.
1. train the **FetchReach-v1**:
```bash
mpirun -np 1 python -u train.py --env-name='FetchReach-v1' --n-cycles=10 2>&1 | tee reach.log
```
2. train the **FetchPush-v1**:
```bash
mpirun -np 8 python -u train.py --env-name='FetchPush-v1' 2>&1 | tee push.log
```
3. train the **FetchPickAndPlace-v1**:
```bash
mpirun -np 16 python -u train.py --env-name='FetchPickAndPlace-v1' 2>&1 | tee pick.log
```
4. train the **FetchSlide-v1**:
```bash
mpirun -np 8 python -u train.py --env-name='FetchSlide-v1' --n-epochs=200 2>&1 | tee slide.log
```

5. Hand envs
```bash
mpirun -np 8 python -u train.py --env-name='HandReach-v0' --n-epochs=200 2>&1 | tee logs/HandReach70.log

# train hierarchical policy
mpirun -np 8 python -u train_hier.py --env-name='HandManipulatePen-v0' --n-epochs=200 2>&1 | tee logs/hier_HandPen.log

mpirun -np 8 python -u train_hier.py --env-name='HandManipulatePenRotate-v0' --n-epochs=200 2>&1 | tee logs/hier_HandPenR.log

# relatively easy env, do this first
mpirun -np 8 python -u train_hier.py --env-name='HandManipulateBlockRotateZ-v0' --n-epochs=200 --c 5 2>&1 | tee logs/hier_HandBZ.log

mpirun -np 8 python -u train_hier.py --env-name='HandManipulateBlockRotateParallel-v0' --n-epochs=200 2>&1 | tee logs/hier_HandBP.log

mpirun -np 8 python -u train_hier.py --env-name='HandManipulateBlockRotateXYZ-v0' --n-epochs=200 2>&1 | tee logs/hier_HandBxyz.log

# use gpu, not work yet
CUDA_VISIBLE_DEVICES=1 mpirun -np 8 python -u train_hier.py --env-name='HandManipulatePenRotate-v0' --n-epochs=200 --cuda 2>&1 | tee logs/hier_HandPenR_cuda.log

# flat policy
mpirun -np 8 python -u train.py --env-name='HandManipulatePen-v0' --n-epochs=200 2>&1 | tee logs/flat_HandPen.log


mpirun -np 8 python -u train.py --env-name='HandManipulateBlockFull-v0' --n-epochs=200 2>&1 | tee logs/flat_HandFull1.log

# target is qpos
mpirun -np 8 python -u train.py --env-name='HandManipulateBlockPos-v0' --n-epochs=200 2>&1 | tee logs/flat_qpos1.log


```

### Play Demo
```bash

python demo.py --env-name=HandManipulateBlockPos-v0 --model_path="saved_models/HandManipulateBlockPos-v0_Dec01_16-36-35_hier_False"

```
### Download the Pre-trained Model
Please download them from the [Google Driver](https://drive.google.com/open?id=1dNzIpIcL4x1im8dJcUyNO30m_lhzO9K4), then put the `saved_models` under the current folder.

## Results
### Training Performance
It was plotted by using 5 different seeds, the solid line is the median value. 
![Training_Curve](figures/results.png)
### Demo:
**Tips**: when you watch the demo, you can press **TAB** to switch the camera in the mujoco.  

FetchPush-v1| FetchPickAndPlace-v1| FetchSlide-v1
-----------------------|-----------------------|-----------------------|
![](figures/push.gif)| ![](figures/pick.gif)| ![](figures/slide.gif)
