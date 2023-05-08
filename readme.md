This is a implementation of BadRL on A2C.

The base learning algorithm A2C is from an open-souce implementation https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

# Install requirements
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt


We provide an example script to simply train and test the attacking algorithm on Breakout environment. Please run

bash trj.sh




