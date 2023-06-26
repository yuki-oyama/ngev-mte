# Markovian traffic equilibrium assignment based on network generalized extreme value model
Python code for a link-based stochastic user equilibrium based on the network-GEV model and its solution algorithms for both primal and dual problems.

## Paper
For more details, please see our paper that has been published in Transportation Research Part B: Methodological.

Oyama, Y., Hara, Y., Akamatsu, T. (2022) [Markovian traffic equilibrium assignment based on network generalized extreme value model](https://www.sciencedirect.com/science/article/pii/S0191261521001934). Transportation Research Part B: Methodological 155: 135-159.

Please cite this paper if you find this code useful:
```
@article{oyama2022markovian,
  title={Markovian traffic equilibrium assignment based on network generalized extreme value model},
  author={Oyama, Yuki and Hara, Yusuke and Akamatsu, Takashi},
  journal={Transportation Research Part B: Methodological},
  volume={155},
  pages={135--159},
  year={2022},
  publisher={Elsevier}
}
```

## Example for Quick Start
Solve the NGEV equilibrium with Accelerated Gradient Projection method (the dual algorithm) in the Sioux Falls network.

```
python run.py --model_name 'NGEVMCA' --optimizers 'AGDBT'
```

## Model options
Loading model options:

- Logit-based Dial assignment
- Logit-based Markovian traffic assignment
- NGEV-based Dial assignment
- NGEV-based Markovian traffic assignment

('syntax' for the scale parameters specification)

Solution algorithm options:

- Method of Successive Averages (MSA)
- Partial Linearization (PL)
- Gradient Projection (GP/GD)
- Accelerated Gradient Projection (AGP/AGD)
