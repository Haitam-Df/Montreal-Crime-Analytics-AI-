import matplotlib
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import time

# -------------------------------------------------------
# Assignment 1
# Written by Haitam Daif 40007112
# For COMP 472 Section KX – Summer 2020
# --------------------------------------------------------

# open/read files
from matplotlib.colors import ListedColormap

file = 'Shape/crime_dt'
data = shp.Reader(file, encoding="ISO-8859-1")

# xMax = -73.55
# xMin = -73.59
# yMax = 45.53
# yMin = 45.49
xMin, yMin, xMax, yMax = data.bbox
# coordinates of Montreal center downtown
coords = []
x = []
y = []

# ask for user input
cellSize = float(input("What cell size would you like to display for? (0.002 or 0.003): "))
thresholdSize = float(input("What threshold would you which to display the data for (0-100)?: "))

# initialize rows and columns based on cell size
columns = math.ceil((yMax - yMin) / cellSize)  # take max-min and divide by cell size to get the numbers of cells
rows = math.ceil((xMax - xMin) / cellSize)  # the ceil function allows rounding to have an exact number of cells
grid = np.array([[0] * rows] * columns)  # creates an empty 2D array of 0's
binsize = columns

# count number of crimes in each cell of the grid
crimeDict = {}
biggestCrimeRate = 0

# appends points of each cell accordingly inside a 2D array
records = data.shapeRecords()

for i in range(len(records)):
    x = int((records[i].shape.__geo_interface__["coordinates"][0] - xMin) / cellSize)  # append the X's
    y = int((records[i].shape.__geo_interface__["coordinates"][1] - yMin) / cellSize)  # append the Y's
    grid[x][y] = grid[x][y] + 1  # append both to the grid to create a 2D array
    if (x, y) in crimeDict.keys():
        crimeDict[x, y] = crimeDict[x, y] + 1
    else:
        crimeDict[x, y] = 1
    if biggestCrimeRate < crimeDict[x, y]:
        biggestCrimeRate = crimeDict[x, y]

# appends the coordinates of each point on the graph
xpoint = []
ypoint = []
for i in range(len(records)):
    coords.append(records[i].shape.__geo_interface__["coordinates"])
    xpoint.append(coords[i][0])
    ypoint.append(coords[i][1])

# initializing graph and grid size
fig, ax1 = plt.subplots(1, 1)
cmap = plt.cm.jet
cmapcolors = ['blue', 'yellow']  # sets the colors used to define low/high crimes rates
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Customized cmap', cmapcolors, cmap.N)
flatGrid = grid.flatten()  # takes the values of the 2d grid and transform it into a 1D array
flatGrid = sorted(flatGrid, reverse=True)  # sort the 1D array in descending order
threshold = np.percentile(flatGrid, thresholdSize)  # gives a fraction of the data based on the percentage desired
bounds = [0, threshold, 280]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
index = len(flatGrid) - int(float(thresholdSize / 100) * len(flatGrid)) - 1  # displays the no of index (199 for 0.002)
high_risk_rate = flatGrid[index]  # cells that contain high risk rates
graph = plt.hist2d(xpoint, ypoint, bins=binsize, cmap=cmap, norm=norm)  # Displays the graph !

# display crime rate number in each cell of the grid
hist, xbins, ybins, im = ax1.hist2d(xpoint, ypoint, bins=binsize, cmap=cmap, norm=norm)
for i in range(len(ybins) - 1):
    for j in range(len(xbins) - 1):
        plt.text(xbins[j] + (cellSize / 2), ybins[i] + (cellSize / 2), int(hist.T[i, j]),
                 fontweight='bold', fontdict=dict(fontsize=5.5, ha='center', va='center'))
plt.colorbar(graph[3], ax=ax1)  # creates color bar graph on the right side of the graph

# computation of statistics ( Mean, Standard deviation, median )
mean = np.mean(flatGrid)
std = np.std(flatGrid)
median = np.median(flatGrid)
ax1.set_title("Mean: " + str(mean) + "\n Std: " + str(std))  # Display stats above graph

# starting A* algorithm

# will mark every crime cell that has a rate higher than the threshold which will help for the making of the algorithm

for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j] >= high_risk_rate:
            grid[i][j] = 1
        else:
            grid[i][j] = 0

grid = grid.transpose()

start_time = time.time()
def neighborsNode(point):
    nodeList = []
    xcell = point[0]  # ith cell in the graph on the x-axis
    ycell = point[1]  # ith cell in the graph on the x-axis

    # down
    if ycell-1 >= 0:  # on the y-axis go down one and make sure it doesn't go out of bounds
        if xcell == 0 or xcell == len(grid[0])-1:
            nodeList.append((xcell, ycell - 1))
        else:
            if not(grid[ycell-1][xcell] == 1 and grid[ycell-1][xcell-1] == 1):
                nodeList.append((xcell, ycell-1))
    # up
    if ycell+1 <= len(grid)-1:  # on the y-axis go up one and make sure it doesn't go out of bounds
        if xcell == 0 or xcell == len(grid[0])-1:
            nodeList.append((xcell, ycell+1))
        else:
            if not(grid[ycell][xcell] == 1 and grid[ycell][xcell-1] == 1):
                nodeList.append((xcell, ycell+1))

    # right
    if xcell+1 <= len(grid[0])-1:
        if ycell == 0 or ycell == len(grid)-1:
            nodeList.append((xcell+1, ycell))
        else:
            if not (grid[ycell][xcell] == 1 and grid[ycell-1][xcell] == 1):
                nodeList.append((xcell + 1, ycell))

    # left
    if xcell - 1 >= 0:
        if ycell == 0 or ycell == len(grid) - 1:
            nodeList.append((xcell - 1, ycell))
        else:
            if not (grid[ycell][xcell - 1] == 1 and grid[ycell - 1][xcell - 1] == 1):
                nodeList.append((xcell - 1, ycell))

    # right-lower diagonal
    if xcell + 1 <= len(grid[0]) - 1 and ycell - 1 >= 0:
        if grid[ycell - 1][xcell] == 0:
            nodeList.append((xcell + 1, ycell - 1))

    # left-lower diagonal
    if xcell - 1 >= 0 and ycell - 1 >= 0:
        if grid[ycell - 1][xcell - 1] == 0:
            nodeList.append((xcell - 1, ycell - 1))

    # right-upper diagonal
    if xcell + 1 <= len(grid[0]) - 1 and ycell + 1 <= len(grid) - 1:
        if grid[ycell][xcell] == 0:
            nodeList.append((xcell + 1, ycell + 1))

    # left-upper diagonal
    if xcell - 1 >= 0 and ycell + 1 <= len(grid)-1:
        if grid[ycell][xcell-1] == 0:
            nodeList.append((xcell-1, ycell+1))

    return nodeList

def cost(point1, point2):
    xcell1 = point1[0]
    ycell1 = point1[1]
    xcell2 = point2[0]
    ycell2 = point2[1]

    if xcell1 == xcell2 and ycell1 - ycell2 == 1:  # for 2 on top of the other the line between them
        if grid[ycell2][xcell2] == 1:  # if one of them is yellow
            return 1.3
        elif xcell1 - 1 >= 0 and ycell1 - 1 >= 0:  # make sure their not out of bound
            if grid[ycell1 - 1][xcell1 - 1] == 1:  # check that the one above it is also yellow or not
                return 1.3
            else:
                return 1
        else:
            return 1

    if xcell1 == xcell2 and ycell2 - ycell1 == 1:  # for 2 on top of each other
        if grid[ycell1][xcell1] == 1:
            return 1.3
        elif xcell1-1 >= 0:
            if grid[ycell1][xcell1-1] == 1:
                return 1.3
            else:
                return 1
        else:
            return 1

    if ycell1 == ycell2 and xcell2-xcell1 == 1:  # for 2 side to side
        if grid[ycell1][xcell1] == 1:
            return 1.3
        elif ycell1-1 >= 0:
            if grid[ycell1-1][xcell1] == 1:
                return 1.3
            else:
                return 1
        else:
            return 1

    if ycell1 == ycell2 and xcell1-xcell2 == 1:  # for 2 side to side
        if grid[ycell2][xcell2] == 1:
            return 1.3
        elif ycell2-1 >= 0:
            if grid[ycell2-1][xcell2] == 1:
                return 1.3
            else:
                return 1
        else:
            return 1

    if abs(xcell1 - xcell2) == 1 and abs(ycell1 - ycell2) == 1:  # if diagonal on blue area
        return 1.5

def pathfinder(startingNode, current):
    path = [current]  # insert current node in path
    while current in startingNode.keys():  # traceback nodes to get the path
        current = startingNode[current]
        path.insert(0, current)
    return path

def astar(start, goal):
    # f cost which represent the cost from any node to the finishing goal.
    def h(n):
        return abs(goal[0] - n[0]) + abs(goal[1] - n[1])

    previousNode = {}  # parent node are stored in here
    openList = [start]  # open list of visited nodes

    # g cost which represent the cost from the start point to any node.
    gn = defaultdict(lambda: float("inf"))
    gn[start] = 0
    print("grissel", gn[start])
    # f function records the cost from start to n plus from n to the goal.
    fn = defaultdict(lambda: float("inf"))
    fn[start] = h(start)

    # visit each node in open list
    while openList:
        lowestfn = math.inf
        current = None
        for point in openList:
            if fn[point] < lowestfn:
                lowestfn = fn[point]
                current = point

        # build path if goal node is reached
        if current == goal:
            return pathfinder(previousNode, current)

        # pop visited node from the open list
        openList.remove(current)
        neighbors = neighborsNode(current)
        for neighbor in neighbors:
            temp_gn = gn[current] + cost(current, neighbor)
            if temp_gn < gn[neighbor]:
                previousNode[neighbor] = current
                gn[neighbor] = temp_gn
                fn[neighbor] = gn[neighbor] + h(neighbor)
                if neighbor not in openList:
                    openList.append(neighbor)

    return None


validCoordinates = False
# ask user to prompt starting and ending point
while not validCoordinates:  # loops until good coordinates have been entered
    xstart, ystart = input("Please enter the start point:").split()
    xstart, ystart = float(xstart), float(ystart)

    xend, yend = input("Please enter the end point:").split()
    xend, yend = float(xend), float(yend)

    # sets the range from which coordinates are accepted
    if (-73.590 <= xstart <= -73.550) and (45.490 <= ystart <= 45.530)\
            and (-73.590 <= xend <= -73.550) and (45.490 <= yend <= 45.530):
        validCoordinates = True
    else:
        print("Invalid coordinates, please try again!")

# cast variables to integers
xstart = int(((xstart - (-73.59)) / cellSize) + 0.01)
ystart = int(((ystart - (45.49)) / cellSize) + 0.01)
xend = int(((xend - (-73.59)) / cellSize) + 0.01)
yend = int(((yend - (45.49)) / cellSize) + 0.01)

# calls A* to initialize path
optPath = astar((xstart, ystart), (xend, yend))

if optPath != None :
    # calculate the total cost
    totalcost = 0
    for i in range(len(optPath) - 1):
        totalcost += cost(optPath[i], optPath[i + 1])

    # fill path to be stored and find the cost of it
    pathDisplay = []
    xpath = []
    ypath = []

    for point in optPath:
        pathDisplay.append((round(point[0] * cellSize + (-73.59), 3), round(point[1] * cellSize + 45.49, 3)))
        xpath.append(round(point[0] * cellSize + (-73.59), 3))
        ypath.append(round(point[1] * cellSize + 45.49, 3))

    print("cost:", "{:.2f}".format(totalcost))

    plt.plot(xpath, ypath, color="red", linewidth=2)
else:
    print("Due to blocks, no path is found. Please change the map and try again")

end_time = time.time()
exectime = end_time - start_time
print("execution time: ", "{:.2f}".format(exectime))
if(exectime >10):
    print("You have passed the 10 seconds allowed for the execution of the algorithm!")
print("Program terminated.")
plt.show()
