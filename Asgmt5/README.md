# Gauss-Newton Algorithm
 
 This is the third assignment of the lesson and it requires us to write a small program to solve a rotation matrix with Gauss-Newton algorithm.

## 1. Requirements
#### General
- Python (verified on 3.6.3)

#### Python Packages
- numpy (verified on 1.14.3)
- matplotlib (verified on 2.2.2)
- argparse (standard library, verified on 1.1)
- scipy (verified on 0.19.1)

## 2. Algorithm

![image](https://github.com/ImCharlesY/Visual-Localization-and-Perception/raw/master/Asgmt3/images/algo.png)

## 3. Assignment requirements
Write a small program to implement a Harris feature detector as the `Relevant Information` part describes.

- In any language (C++, java, python, Matlab).

- Implement each step described in the algorithm.

- Readme.txt about how to compile and run the program.

- A short report about the experimental results.

## 4. Run the scripts

### Install the requirements

```
pip install -r requirements.txt
```

### Command-line

```
python [script name].py -h
```

All the scripts use `argparse` module to make it easy to write user-friendly command-line interfaces. You can use option `-h` or `--help` to get a useful usage message for each script.

For this script, there are 7 other options that you can use in command line.

```
usage: Gauss_Newton.py [-h] [--maxits [MAXITS]] [--seed [RANDOM_SEED]]
                       [--num_init [NUM_INIT]] [--orth] [--result] [--legend]
                       [--plot]

optional arguments:
  -h, --help            show this help message and exit
  --maxits [MAXITS]     Maximum number of iterations of the algorithm to
                        perform. Default 256.
  --seed [RANDOM_SEED]  Random seed. Default 0
  --num_init [NUM_INIT]
                        Number of initial guesses. Default 10.
  --orth                Whether to orthogonalize the initial guesses.
  --result              Whether to save the results.
  --legend              Whether to show the legend in error graph.
  --plot                Whether to show the error graph.

```

Note: You can get the error graph in directory `./result/`. If you specify `--result` option, then you also get the solution in the same directory. 

For example:

```
python -m Gauss_Newton --num_init 100 --plot
```
