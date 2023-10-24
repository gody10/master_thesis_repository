This Repository Contains :

- nyiso_hourly_prices.csv -> CSV containing all the historical price data that we need for the environment
- requirements.txt -> PIP package information in order to setup your environment.
- SmartHomeGymEnv_v2_deployment.py -> This is the gym environment python file required to train the Reinforcement Learning algorithms
- train_agent_on_environment.ipynb -> This is the notebook that trains the agent in several different environments
- test_agent_deployment.ipynb -> This is the notebook that uses already trained agents and tests their performance on a given day
- env_5_pr_5_df_experiments_marwil -> This folder contains checkpoints for all the available environment combinations for the MARWIL algorithm

BEFORE RUNNING:
1. Download and open the anaconda prompt (https://www.anaconda.com/download)
2. Execute the "conda create -n your_env_name python=3.10.12" command to create a virtual conda environment with the specific python version
3. Execute the "conda activate your_env_name" command to activate your virtual conda environment
4. Execute the "pip install -r requirements.txt" to install all the required packages with the correct versions
5. Run the jupyter notebooks

