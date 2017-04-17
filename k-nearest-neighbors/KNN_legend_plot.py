# Import bokeh packages
from bokeh.layouts import column
from bokeh.models import CategoricalColorMapper, ColumnDataSource, CustomJS, Legend, Range, Range1d, Slider, HoverTool
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


# Load iris data
iris = load_iris()
trainingData = iris.data
featureNames = iris.feature_names
trainingLabels = iris.target
labelNames = iris.target_names

# Bucket the training data points by training label (i.e. setosa, versicolor or virginica)
# This will make it easier to generate a legend for the plot
label_0 = []
label_1 = []
label_2 = []
for i in range(len(trainingLabels)):
    if trainingLabels[i] == 0:
        label_0.append(i)
    elif trainingLabels[i] == 1:
        label_1.append(i)
    else:
        label_2.append(i)
        
# Define color palettes for plots (light for decision surfaces, dark for data points)
light_palette = [Category20[6][2*i + 1] for i in range(3)]
dark_palette = [Category20[6][2*i] for i in range(3)]

# Determine the limits of the decision boundary plots
x_min = min([dataPoint[0] for dataPoint in trainingData]) - 1
x_max = max([dataPoint[0] for dataPoint in trainingData]) + 1
x_range = Range1d(x_min, x_max, bounds = (x_min, x_max))

y_min = min([dataPoint[1] for dataPoint in trainingData]) - 1
y_max = max([dataPoint[1] for dataPoint in trainingData]) + 1
y_range = Range1d(y_min, y_max, bounds = (y_min, y_max))


file_names = ["knn1.html","knn2.html","knn3.html","knn4.html","knn5.html",
              "knn6.html","knn7.html","knn8.html","knn9.html","knn10.html"]
nums = [1,2,3,4,5,6,7,8,9,10]
for j in nums:
    # Train a model on the first two features
    model = KNeighborsClassifier(n_neighbors=j) # create the classifier
    X = [ [ dataPoint[0], dataPoint[1] ] for dataPoint in trainingData]
    y = trainingLabels
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
    source_0 = ColumnDataSource(data = dict(x = [trainingData[i][0] for i in label_0],
                                            y = [trainingData[i][1] for i in label_0],
                                            true_classes = [labelNames[trainingLabels[i]] for i in label_0]))
    
    bokeh_plot.circle(x = source_0.data['x'],
                      y = source_0.data['y'], 
                      source = source_0,
                      size = 4,
                      fill_color = dark_palette[0],
                      line_color = dark_palette[0],
                      legend = labelNames[0]
                     )
    
    
    # Plot data points in the label_1 bucket
    source_1 = ColumnDataSource(data = dict(x = [trainingData[i][0] for i in label_1],
                                            y = [trainingData[i][1] for i in label_1],
                                            true_classes = [labelNames[trainingLabels[i]] for i in label_1]))
    
    bokeh_plot.circle(x = source_1.data['x'],
                      y = source_1.data['y'], 
                      source = source_1,
                      size = 4,
                      fill_color = dark_palette[1],
                      line_color = dark_palette[1],
                      legend = labelNames[1],
                     )
    
    
    # Plot data points in the label_2 bucket
    source_2 = ColumnDataSource(data = dict(x = [trainingData[i][0] for i in label_2],
                                            y = [trainingData[i][1] for i in label_2],
                                            true_classes = [labelNames[trainingLabels[i]] for i in label_2]))
    
    bokeh_plot.circle(x = source_2.data['x'],
                      y = source_2.data['y'], 
                      source = source_2,
                      size = 4,
                      fill_color = dark_palette[2],
                      line_color = dark_palette[2],
                      legend = labelNames[2],
                     )
    
    # Label axes, place legend
    bokeh_plot.xaxis.axis_label = featureNames[0]
    bokeh_plot.yaxis.axis_label = featureNames[1]
    bokeh_plot.legend.location = "bottom_left"
    title = "When K = {}".format(j)
    bokeh_plot.title.text = title

    output_file("knn{}.html".format(j))
    save(bokeh_plot)

