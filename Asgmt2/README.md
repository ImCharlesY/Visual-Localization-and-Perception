# Feature Detection
 
 This is the second assignment of the lesson and it requires us to write a small program to implement a the Harris feature detector.

## 1. Requirements
#### General
- Python (verified on 3.6.1)

#### Python Packages
- numpy (verified on 1.14.3)
- matplotlib (verified on 2.2.2)
- argparse (standard library, verified on 1.1)
- cv2 (verified on 3.4.3)

## 2. Relevant information
*Corner*: A corner is a point that can be interpreted as the junction of two edges in images. Its local neighborhood always stands in two dominant and different edge directions. 

*Harris Corner Detector* :In order to capture the corners from the image, researchers have proposed many different corner detectors including the Kanade-Lucas-Tomasi (KLT) operator and the Harris operator which are most simple, efficient and reliable for use in corner detection. 

## 3. Algorithm

+ 1. Compute corner response.

For each pixel (x,y) in the input image, we calculates a 2 * 2 gradient covariance matrix M(x,y) over a windowSize * windowSize neighborhood. Then, the Harris corner response is defined as follows:
<img src="http://latex.codecogs.com/svg.latex?dst_%7Bx%2Cy%7D%20%3D%20det%28M%29%20-%20k%5Ccdot%20tri%28M%29%5E2" />, where M is given by 

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/a617dda21e306dbfbdb7a186b1c203e3f3443867" /> 

and k is an empirical constant (0.04-0.06).

+ 2. Find region of pixels whose corner responses are larger the threshold.

We set the threshold by <img src="http://latex.codecogs.com/svg.latex?thresh%20%5Ccdot%20max%28dst_%7Bx%2Cy%7D%29" /> where `thresh` is a given parameter.

+ 3. Find local maximum in each region.

We regard two cliques with less than 100 adjacent pixels as different regions. Then we use the pixel with local maximum corner response to represent each region. These pixels are just the corners we want to detect.

## 4. Assignment requirements
Write a small program to implement a Harris feature detector as the `Relevant Information` part describes.

- In any language (C++, java, python, Matlab).

- Implement each step described in the algorithm.

- Readme.txt about how to compile and run the program.

- A short report about the experimental results.

## 5. Run the scripts

### Install the requirements

```
pip install -r requirements.txt
```

### Command-line

```
python [script name].py -h
```

All the scripts use `argparse` module to make it easy to write user-friendly command-line interfaces. You can use option `-h` or `--help` to get a useful usage message for each script.

For this script, there are 5 other options that you can use in command line.

|  option  |         description                |   default  |
| -------- |:----------------------------------:|:----------:|
|    -i    | the orginal image file name        | "test.jpg" |
|    -w    | the size of the sliding window     |      2     |
|    -k    | An empirical constant              |    0.04    |
|    -t    | The threshold for an optimal value |   0.001    |

You should put your images in directory `images/input/` . Using the above options to set appropriate parameters and apply __Harris Feature Detector__ to them. Then you can get the result graph in directory `images/output/` .  

For example:

```
python feat_detection.py -i test.jpg -w 2 -k 0.04 -t 0.001
```


