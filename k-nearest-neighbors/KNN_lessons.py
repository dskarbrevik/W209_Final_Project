# Import bokeh packages
from bokeh.layouts import column
from bokeh.models import (CategoricalColorMapper, ColumnDataSource, 
CustomJS, Legend, Range, Range1d, Slider, Arrow, VeeHead, OpenHead, HoverTool, FixedTicker)
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, save, show, reset_output

# Import python packages
from IPython.display import Image
#import graphviz
import numpy as np
#import pydotplus 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

import os

os.chdir("C:\\Users\\skarb\\Desktop\\Github\\W209_Final_Project\\k-nearest-neighbors\\")


#np.set_printoptions(threshold=np.inf)

# Load iris data
iris = load_iris()

#np.random.seed(1)

#shuffle = np.random.permutation(np.arange(iris.data.shape[0]))
#iris.data, iris.target = iris.data[shuffle], iris.target[shuffle]

#small_data = iris.data
featureNames = iris.feature_names
#small_labels = iris.target
labelNames = iris.target_names


select_data = [1,12,18,20,36,55,61,73,89,95,103,112,127,144,148]
#select_data = [1,3,5,8,12,13,16,18,20,25,32,36,42,43,
#               51,52,55,61,62,63,73,79,85,87,89,90,95,96,
#               103,104,107,112,115,119,122,124,127,129,132,138,144,148]

small_data = np.zeros([15,2])
small_labels = np.zeros([15,])
for i,j in zip(list(range(15)),select_data):
    small_data[i][0] = iris.data[j][0]
    small_data[i][1] = iris.data[j][1]
    small_labels[i] = iris.target[j]

smallest_data = np.zeros([2,2])
smallest_labels = np.zeros([2,])
smallest_data[0] = iris.data[5,:2]
smallest_data[1] = iris.data[68,:2]
smallest_labels[0] = iris.target[5]
smallest_labels[1] = iris.target[68]

# Bucket the training data points by training label (i.e. setosa, versicolor or virginica)
# This will make it easier to generate a legend for the plot
label_0 = []
label_1 = []
label_2 = []
for i in range(len(small_labels)):
    if small_labels[i] == 0:
        label_0.append(i)
    elif small_labels[i] == 1:
        label_1.append(i)
    else:
        label_2.append(i)
        
# Define color palettes for plots (light for decision surfaces, dark for data points)
light_palette = [Category20[6][2*i + 1] for i in range(3)]
dark_palette = [Category20[6][2*i] for i in range(3)]

# Determine the limits of the plot
x_min = min([dataPoint[0] for dataPoint in small_data]) - 1
x_max = max([dataPoint[0] for dataPoint in small_data]) + 1
x_range = Range1d(x_min, x_max, bounds = (x_min, x_max))

y_min = min([dataPoint[1] for dataPoint in small_data]) - 1
y_max = max([dataPoint[1] for dataPoint in small_data]) + 1
y_range = Range1d(y_min, y_max, bounds = (y_min, y_max))

k_val=2
model = KNeighborsClassifier(n_neighbors=k_val) # create the classifier
X = [ [ dataPoint[0], dataPoint[1] ] for dataPoint in small_data]
y = small_labels
model.fit(X, y) # train the classifier


# Create a mesh grid based on the plot limits, then classify the mesh using the trained model
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 
                     np.arange(y_min, y_max, 0.01))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
zz = z.reshape(xx.shape)

# Create bokeh figure
bokeh_plot = figure(plot_width=500,
                    plot_height=500,
                    x_range = x_range,
                    y_range = y_range,
                    tools = "pan, box_zoom, wheel_zoom, reset, undo, redo"
                    )

# Add a custom hover tool to the plot
custom_hover = HoverTool(tooltips = [("Index", "$index"),
                                     ("(x, y)", "($x{0.00}, $y{0.00})"),
                                     ("Class", "@true_classes")
                                    ])
bokeh_plot.add_tools(custom_hover)

# Plot the mesh grid on the bokeh figure as an image
# Note: "Z" here is the mesh grid predictions, and it must be contained in a list (i.e. "[Z]" not "Z")
bokeh_plot.image(image=[zz],
                 alpha = 0,
                 x = x_min,           
                 y = y_min,
                 dw =(x_max - x_min),
                 dh =(y_max - y_min),
                 palette = light_palette
                )

# Plot data points in the label_0 bucket
source_0 = ColumnDataSource(data = dict(x = [small_data[i][0] for i in label_0],
                                        y = [small_data[i][1] for i in label_0],
                                        true_classes = [labelNames[small_labels[i]] for i in label_0]))

bokeh_plot.circle(x = source_0.data['x'],
                  y = source_0.data['y'], 
                  source = source_0,
                  size = 6,
                  fill_color = dark_palette[0],
                  line_color = dark_palette[0],
                  legend = labelNames[0]
                 )


# Plot data points in the label_1 bucket
source_1 = ColumnDataSource(data = dict(x = [small_data[i][0] for i in label_1],
                                        y = [small_data[i][1] for i in label_1],
                                        true_classes = [labelNames[small_labels[i]] for i in label_1]))

bokeh_plot.circle(x = source_1.data['x'],
                  y = source_1.data['y'], 
                  source = source_1,
                  size = 6,
                  fill_color = dark_palette[1],
                  line_color = dark_palette[1],
                  legend = labelNames[1],
                 )


# Plot data points in the label_2 bucket
source_2 = ColumnDataSource(data = dict(x = [small_data[i][0] for i in label_2],
                                        y = [small_data[i][1] for i in label_2],
                                        true_classes = [labelNames[small_labels[i]] for i in label_2]))

bokeh_plot.circle(x = source_2.data['x'],
                  y = source_2.data['y'], 
                  source = source_2,
                  size = 6,
                  fill_color = dark_palette[2],
                  line_color = dark_palette[2],
                  legend = labelNames[2],
                 )

# add lines connecting from unknown point to known points
#bokeh_plot.line(x=[smallest_data[0][0],5.8],
#                y=[smallest_data[0][1],3.8], 
#                line_dash="dashed",
#                line_color="grey",
#                line_width = 2)
#
#bokeh_plot.line(x=[smallest_data[1][0],5.8],
#                y=[smallest_data[1][1],3.8], 
#                line_dash="dashed",
#                line_color="grey",
#                line_width = 2)

#bokeh_plot.add_layout(Arrow(end=OpenHead(size=12, line_color="grey"), 
#                            line_color="grey",
#                            line_width = 2,
#                            line_dash="dashed",
#                            x_start=5.8, 
#                            y_start=3.8, 
#                            x_end=smallest_data[0][0], 
#                            y_end=smallest_data[0][1]))
#
#bokeh_plot.add_layout(Arrow(end=OpenHead(size=12, line_color="grey"), 
#                            line_color="grey",
#                            line_width = 2,
#                            line_dash="dashed",
#                            x_start=5.8, 
#                            y_start=3.8, 
#                            x_end=smallest_data[1][0], 
#                            y_end=smallest_data[1][1]))

# Label axes, place legend
bokeh_plot.xaxis.axis_label = featureNames[0]
bokeh_plot.yaxis.axis_label = featureNames[1]
bokeh_plot.legend.location = "bottom_left"
title = "k-NN decision boundary (where k = 2)"
bokeh_plot.title.text = title

output_file("knn_lesson3.html")
save(bokeh_plot)
reset_output()