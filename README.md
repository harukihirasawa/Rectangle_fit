# Rectangle_fit.py README
#### Haruki Hirasawa 
##### November 27, 2014
#####Memorial University of Newfoundland, Dept. of Physics and Physical Oceanography
### Introduction
**rectangle_fit.py** is a python code designed for use in analyzing images of rod-like colloids. I was written by Haruki Hirasawa under the supervision of Dr. Anand Yethiraj of the Memorial University of Newfoundland, Department of Physics and Physical Oceanography.

### What does Rectangle_fit do?
**rectangle_fit.py** analyzes images of rod colloids as follows:
1. Find an appropriate threshold value for the image
2. Take the points whose intensity surpasses the threshold
3. Find the regions of connected bright pixels
4. Ignoring small regions, find the centroid and "major axis" of each pixel blob
5. Fit a bounding rectangle to the pixel blob using the technique described by D. Chaudhuri et. al. in "_Finding best-fitted rectangle for regions using a bisection method_" (2014)
6. Classify each rectangle based on a variety of criteria into three categories: One rod, Two Rods, Three or more Rods
7. By using the profiles of the rectangles along the major and minor axes, split any long rectangle as well as optimized any one rod rectangles.
8. Overlay fitted rectangles onto the original image and print
9. Calculate and print metadata to a .dat file
10. Output data into a .csv file
11. Print a number of histograms to characterize the overall tendencies of the rods in the image.


### Required Python Libraries
The most current version of the code was developed using the following libraries and versions:

* **numpy**  1.91
* **scipy** 0.14.0
* **matplotlib** 1.4.2
* **pillow** 2.6.0

### Using Rectangle_fit.py

