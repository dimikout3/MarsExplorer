# Mars Explorer

Mars Explorer is an openai-gym compatible environment designed and developed as an initial endeavor to bridge the gap between powerful Deep Reinforcement Learning methodologies and the problem of exploration/coverage of an unknown terrain.

<img src="utils/images_repo/intro.gif">

## Strong Generalization is the Key

Terrain diversification is one of the MarsExplorer kye attributes. For each episode, the general dynamics are determined by a specific automated process that has different levels of variation. These levels correspond to the randomness in the number, size, and positioning of obstacles, the terrain scalabality (size), the percentage of the terrain that the robot must explore to consider the problem solved and the bonus reward it will receive in that case. This procedural generation of terrains allows training in multiple/diverse layouts, forcing, ultimately, the RL algorithm to enable generalization capabilities, which are of paramount importance in real-life applicaiton where unforeseen cases may appear.

<img src="utils/images_repo/terrain.gif">

# Installation

## Quick Start

You can install MarsExplorer environment by using the following command:

```shell
$ git clone https://github.com/dimikout3/GeneralExplorationPolicy.git
$ pip install -e mars-explorer
```

## Full Installation 

If you want you can proceed with a full isntallation, that includes a pre-configured CONDA environment with the Ray/RLlib and all the dependancies. Thereby, enabling a safe and robust pipelining approach to training your own agent on exploration/coverage missions.

```shell
$ sh setup.sh
```

## Dependancies

You can have a better look at the dependencies at:

```shell
setup/environment.yml
```

# Testing

Please run the following command to make sure that everything works as expected:

```shell
$ python mars-explorer/tests/test.py
```

## Manual Control

We have included a manual control of the agent, via the corresponding arrow keys. Run the manual control environment via:

```shell
$ python mars-explorer/tests/manual.py
```
