## ABIDES calibration using the Optuna framework

> Optuna is a hyperparameter optimization framework that can be used for agent-based model calibration. 
> It offers a number of functionalities including parallel distributed optimization, in-built sqllite storage and 
> the ability to easily construct the search spaces for the parameters to be tuned. 
> Optimization is done by choosing a suitable set of hyperparameter values from a given range. 
>Uses a sampler which implements the task of value suggestion based on a specified distribution.
>
- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.
- Algorithms for Hyper-Parameter Optimization: <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>
- Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures: <http://proceedings.mlr.press/v28/bergstra13.pdf>

## How to use:
Example:
```
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}
```

ABIDES: 
```
cd ../
python -u calibrate.py
```

ABIDES Functions:

```
def abides(agents, props, oracle, name):
    """ run an abides simulation using a list of agents. 
        Note this implementation assumes zero-latency.

    :param agents: list of agents in the ABM simulation
    :param props: simulation-specific properties
    :param oracle: the data oracle for the simulation
    :param name: simulation name
    :return: agent_states (saved states for each agent at the end of the simulation)
    """
```

```
def config(seed, params):
    """ create the list of agents for the simulation

    :param params: abides config parameters
    :return: list of agents given a set of parameters
    """
```

```
def objective(trial):
    """ The objective function to be optimized given parameters of the agent-based model

    :param trial: a single execution of the objective function
    :return: objective function
    """

    params = {
        'n_value': trial.suggest_int('n_value', 1, 100),
        'n_noise': trial.suggest_int('n_value', 1, 100)
    }

    # 1) get list of agents using params
    agents = config(SEED, params)

    # 2) run abides
    agents_saved_states = abides(name=trial.number,
                                 agents=agents,
                                 oracle=oracle(),
                                 latency_model=latency_model(num_agents=len(agents)),
                                 default_computation_delay=1000)# one millisecond

    # 3) run abides with the set of agents and properties and observe the resulting agent states
    exchange, gains = agents_saved_states[0], agents_saved_states[1:]

    return sum(gains)  # simple objective to maximize the gains of all agents
```