# COP-Ising-Model
 
Authors: Jonah Post, Mio Poortvliet
##How to run

Run main.py to run all examples, in order.

If you don't want to execute from main.py, you need to specify your working directory to be the same as the root folder (```COP-Ising-Model```).

###Why is this project structured like this?
We intended to write a general Metropolis Algorithm file that will accept any physics system. This is why the project is structured like a library and is intended to be used by importing ```MetropolisAlgorithm``` from ```MetropolisAlgorithm.py```. Then you feed it your own Hamiltonian, similar to how we feed it ```IsingModel``` in the examples. 

##Features
- General implementation of ```MetropolisAlgorithm```.
- Contains an easy to modify example system ```IsingModel```.
- Works in any dimension, though becomes horribly slow after ```n=2```.
- Focus on speed, makes smart use of numpy and some numba.

##Dependencies
- Numpy
- Matplotlib
- Numba

See the report for more information on how to use this, numerical experiments and a reflection.