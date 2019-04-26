# RL_Easy21
This folder contains the implementation of the [Easy21 assignment](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf) of David Silver's Reinforcement Learning course.

### Dependencies

The script makes use of the following dependencies, which are to be installed by the user using the standard <a href="https://pip.pypa.io/en/stable/">**pip**</a> module. These dependencies were tested under **Ubuntu 18.04** and using **Python3**.

- [dill](https://pypi.org/project/dill/)
- [pandas](https://pypi.org/project/pandas/)
- [seaborn](https://pypi.org/project/seaborn/)
- [OpenAI Gym](https://github.com/openai/gym)

### Installation
1. Create a virtual environment:

      ```
      $ virtualenv pyeasy21
      ```
2. Activate the virtual environment:

      ```
      $ source pyeasy21/bin/activate
      ```
3. Install all the dependencies necessary for running RL algorithms and plot results:

      ```
      $ pip install -r requirements.txt
      $ pip install -e gym-easy21
      ```
The `gym-easy21` environment is OpenAI Gym compatible. You can create an instance of the it with:
      ```python
      gym.make('gym_easy21:easy21-v0')
      ```

### Run
The following scripts implements the 3 algorithms required by the assignment:
 - `mc.py`: Monte Carlo control
 - `sarsa_lambda.py`: TD Learning
 - `linear_approx.py`: Linear Function Approximation
