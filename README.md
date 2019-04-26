# RL_Easy21
This folder contains the implementation of the [Easy21 assignment](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf) of David Silver's Reinforcement Learning course.

### Dependencies

The script makes use of the following dependencies, which are to be installed by the user using the standard <a href="https://pip.pypa.io/en/stable/">**pip**</a> module. These dependencies were tested under **Ubuntu 18.04** and using **Python3**.

- [dill](https://pypi.org/project/dill/)
- [pandas](https://pypi.org/project/pandas/)
- [seaborn](https://pypi.org/project/seaborn/)

- [OpenAI Gym](https://github.com/openai/gym)

### Installation
The environment is OpenAI Gym compatible. To install it, you need to run:
```
$ pip install -e gym-foo
```
Then, you can create an instance of the environment with:
```python
gym.make('gym_easy21:easy21-v0')
```
### run
The following scripts implements the 3 algorithms required by the assignment:
 - `mc.py`: Monte Carlo control
 - `sarsa_lambda.py`: TD Learning
 - `linear_approx.py`: Linear Function Approximation 
