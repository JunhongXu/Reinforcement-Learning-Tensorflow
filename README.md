# Reinforcement Learning-Tensorflow

# Current Status:
1. Deep Deterministic Policy Gradient (DONE)
    
    Completed the Pendulum-v0 and ContinuousMountainCar OpenAI Gym tasks.
    
    log directory contains tensorboard log for each model (only have DDPG currently) and env trained by that model

    tmp directory contains OpenAI Gym training videos and statistics

    models directory contains saved tensorflow models and saved memory.

    To run the code, you can run

        $ python main.py

    To visualize training:

        $ tensorboard --logdir=log/DDPG/env_name

    

# Future plans:
1. Deep Q Learning (with dueling and target networks). Link:https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf and https://arxiv.org/pdf/1511.06581.pdf
2. Deep Deterministic Policy Gradient. Link: https://arxiv.org/pdf/1509.02971.pdf
3. A3C. Link: https://arxiv.org/pdf/1602.01783v2.pdf
