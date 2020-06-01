# Montreal-Crime-Analytics-AI-
Haitam Daif - 40007112

##How to run the program:
# Start by entering the cell size for which you desire to display the graph for
# Input a threshold size for which you desire the crime rates to be filtered for
# Input the start point coordinates from which you desire to start your search 
# Input the end point coordinates to which you desire to reach
# coordinates x range: -73.590 <= coordinates x <= -73.550
# coordinates y range: 45.490 <= coordinates y <= 45.530

##Libraries used in the program:
#Mathplot    => used to plot the 2d graph 
#Hist2d      => used to display the 2d graph with colors and text inside each cell
#math 	      => used to perform mathematical operations such as ceiling()
#numpy       => used to perform statistical operations
#shapefile   => used to read the shapefile which contains the points and values of each cell
#time	      => used to calculate time to run the program in seconds
#efaultdict => used to store data values like from the map (grid/graph)

##Test cases:
1#cellsize: 0.002
threshold: 50
-73.58 45.51 => starting point
-73.56 45.49 => ending point

2#cellsize: 0.002
threshold: 50
-73.58 45.51 => starting point
-73.5531 45.5033 => ending point

3#cellsize: 0.002
threshold: 55
-73.551 45.5287 => starting point
-73.5869 45.4914 => ending point

4#cellsize: 0.002
threshold: 70
-73.5875 45.5288 => starting point
-73.5508 45.493 => ending point

5#cellsize: 0.002
threshold: 50
-73.58 45.51 => starting point
-73.5631 45.5092 => ending point

