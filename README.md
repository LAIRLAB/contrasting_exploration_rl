# Contrasting Exploration in Parameter and Action Space

## Citation

If you use this code, please cite the following paper

> Anirudh Vemula, Wen Sun, J. Andrew Bagnell. **Contrasting Exploration in Parameter and Action Space: A Zeroth Order Optimization Perspective**. *In the proceedings of Artificial Intelligence and Statistics (AISTATS). 2019*

## Reproducing Experiments

### Setup
Run

``` shell
pip install -r requirements.txt
./scripts/setup.sh
```

### Mujoco Experiments

To reproduce the results of the Mujoco Experiments (both Swimmer and HalfCheetah) with hyperparameter tuning, run

``` shell
./scripts/run_mujoco_experiments.sh
```

### LQR Experiments

To reproduce the results of the LQR experiments with hyperparameter tuning, run

``` shell
./scripts/run_lqr_experiments.sh
```

### Generating Plots

To generate plots used in the paper, run the following commands after running the experiments

``` shell
python scripts/plot_zooomed_results.py
python scripts/plot_lqr_results.py
```

## Dependencies
* Python 3.6.3
* Ray 0.5.2
* Numpy 1.15.1
* Matplotlib 2.2.3
* Mujoco-py 1.50.1.56
* Gym 0.10.5
* Scipy 1.1.0
* Jupyter 1.0.0
* Autograd
