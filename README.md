# Hindsight Experience Replay (HER)
This is a pytorch implementation of [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). 

## Requirements
- python 3.5.2
- openai-gym
- mujoco-1.50.1.56
- pytorch-1.0.0
- mpi4py

## TODO List
- [ ] support GPU acceleration.
- [ ] add multi-env per MPI.

## Instruction to run the code
1. train the `FetchReach-v1`:
```bash
mpirun -np 1 python -u train.py --env-name='FetchReach-v1' --n-cycles=10 2>&1 | tee reach.log
```
2. train the `FetchPush-v1`:
```bash
mpirun -np 8 python -u train.py --env-name='FetchPush-v1' 2>&1 | tee push.log
```
3. train the `FetchPickAndPlace-v1`:
```bash
mpirun -np 16 python -u train.py --env-name='FetchPickAndPlace-v1' 2>&1 | tee pick.log
```

### Play Demo
```bash
python demo.py --env-name=<environment name>
```
### Download the Pre-trained Model
Please download them from the [Google Driver](https://drive.google.com/open?id=1dNzIpIcL4x1im8dJcUyNO30m_lhzO9K4), then put the `saved_models` under the current folder.

## Results
### Training Performance
![Training_Curve](figures/results.png)
### Demo:
**Note:** the new-version openai-gym has problem in rendering, so I use the demo of `Walker2d-v1`  
**Tips**: when you watch the demo, you can press **TAB** to switch the camera in the mujoco.  
![Demo](figures/demo.gif)

