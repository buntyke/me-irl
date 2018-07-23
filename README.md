# Maximum Entropy Inverse Reinforcement Learning

ME-IRL implementation in grid world. It is based on the irl-imitation git repository available [here](https://github.com/stormmax/irl-imitation).

This implementation has an additional approx flag that approximates the state-transition dynamics. To test the working, run the following commands:


For true state transition dynamics:
```
$ python maxent_irl_gridworld.py --gamma=0.8 --n_trajs=1000 --l_traj=50 --wid 10 --hei 10 --rand_start --learning_rate=0.01 --n_iters=20
```

For monte-carlo approximation of state-transition dynamics:
```
$ python maxent_irl_gridworld.py --gamma=0.8 --n_trajs=1000 --l_traj=50 --wid 10 --hei 10 --rand_start --learning_rate=0.01 --n_iters=20 --approx
```


