# Visual Localization and Perception
 
 This repository contains the implementations of assignments in Visual Localization and Perception Lesson (EE382).

## 1. Requirements
#### General
- Python (verified on 3.5.4)

#### Python Packages
- numpy (verified on 1.14.3)
- matplotlib (verified on 3.2.1)
- argparse (standard library, verified on 1.1)

## 2. Assignments
### 2.1. Assignment 1
#### 2.1.1 Relevant information
*Histogram Specialization*: Histogram specialization adjust the brightness of the original image to make it be close to the target image, which can address the illumination change to some extend and make feature matching easier.

*Algorithm*:

1. Calculate the cumulative histogram of the first image  <img src="http://latex.codecogs.com/svg.latex?f_1(I)" />  ;

2. Calculate the cumulative histogram of the second image  <img src="http://latex.codecogs.com/svg.latex?f_2(I^{'})" />  ;

3. Build a lookup table  <img src="http://latex.codecogs.com/svg.latex?I%5E%7B%27%7D%20%3D%20f_2%5E%7B-1%7D%20%5Ccirc%20f_1%28I%29" />  by finding the corresponding gray level  <img src="http://latex.codecogs.com/svg.latex?I_j^{'}" />  of each gray level  <img src="http://latex.codecogs.com/svg.latex?I_i" />  , where  <img src="http://latex.codecogs.com/svg.latex?I_j%5E%7B%27%7D%20%3D%20argmin_j%20%7Cf_1%28I_i%29%20-%20f_2%28I_j%5E%7B%27%7D%29%7C" />  ;

4. Map the new intensity of each pixel by finding the lookup table.

#### 2.1.2 Assignment requirements
Write a small program to implement a histogram specialization algorithm as the `Relevant Information` part describes.

- In any language (C++, java, python, Matlab).

- Implement each step described in the algorithm.

- Tests on at least 5 pairs of images with different exposure time.

- Readme.txt about how to compile and run the program.

- A short report about the experimental results.

## 3. Run the scripts

### Install the requirements

```
pip install -r requirements.txt
```

If everything works well, you can run `python ./Asgmt1/hist_spec.py` to apply histogram specialization to a pair of test images.

### Command-line

```
python [script name].py -h
```

All the scripts use `argparse` module to make it easy to write user-friendly command-line interfaces. You can use option `-h` or `--help` to get a useful usage message for each script.
