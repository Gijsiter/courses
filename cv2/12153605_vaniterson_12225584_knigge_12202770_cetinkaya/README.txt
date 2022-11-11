### 	README						###
	CV 2 
	Assignment 2
	
	12153605  	12225584 	12202770
	Gijs		Levi		Bunyamin
###							###


Requirements provided in requirements.txt

Code for plots is provided in the files themselves.

Section 3:
Figures 1 and 2 are from the assignment.

Figure 3 can be reproduced as follows:
1. in fundamental.py  in the function definition for normalized_eight_point() uncomment the code under the comment "print verifying the mean and average distance"
(provided in main)
2. in sfm.py load any set of images
3. get any point sets using get_homographies()
4. run estimation with normalization estimate_fundamental(X1, X2, method='normalized_eightpoint')

Figure 4:
(provided in main of sfm.py)
1. Load all images
2. estimate inliers using 3 different methods: eight-point, normalized-eight-point, nep-ransac
3. plot using plot_epipolar_lines()
4. (optional) repeat using cv2.findFundamentalMat() -> plot_epipolar_lines()

Figure 5:
Simply load all images in sfm.py and input into point_view_mat().
plot the resulting matrix != 0.

Figure 6-8:
Using provided PVM ans Stitch function, plot the mentioned selection of frames and settings.


Figure 9-11:
Using the colmap functions, plot the views.
Click on Reconstruction > Automatic reconstruction.
Run using the real-world-images.
