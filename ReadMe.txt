# Master Thesis: Residential Energy Consumption Optimization through Reinforcement Learning

Welcome to the repository dedicated to my Master's thesis focusing on Reinforcement Learning (RL) applications in household energy optimization.

## Contents

- **nyiso_hourly_prices.csv**: Contains historical price data crucial for modeling and simulating energy market environments.
- **requirements.txt**: PIP package information essential for setting up the required Python environment.
- **SmartHomeGymEnv_v2_deployment.py**: Gymnasium environment definition essential for training RL algorithms tailored for energy market scenarios.
- **train_agent_on_environment.ipynb**: Notebook for training RL agents in various environmental contexts.
- **test_agent_deployment.ipynb**: Testing pre-trained RL agents' performance on specific days within the energy market.
- **env_5_pr_5_df_experiments_marwil/**: Folder housing checkpoints for different environment configurations used for training RL agents employing the MARWIL algorithm.

## Setup Instructions

### Before Running:

1. Download and open Anaconda Prompt ([Download Anaconda](https://www.anaconda.com/download)).
2. Execute: `conda create -n your_env_name python=3.10.12` to create a virtual Conda environment with the specified Python version.
3. Activate the virtual Conda environment: `conda activate your_env_name`.
4. Install required packages: `pip install -r requirements.txt`.
5. Run the Jupyter Notebooks to explore RL training and testing procedures.

This repository serves as a comprehensive guide and resource hub for understanding, experimenting with, and implementing RL models in the context of household energy optimization.

Feel free to explore the provided resources, engage in experiments, and contribute to the advancement of Reinforcement Learning methodologies within this wonderful field.
