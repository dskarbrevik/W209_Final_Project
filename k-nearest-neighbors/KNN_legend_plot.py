# Import bokeh packages
from bokeh.layouts import column
from bokeh.models import CategoricalColorMapper, ColumnDataSource, CustomJS, Legend, Range, Range1d, Slider
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, save, show

# Import python packages
from IPython.display import Image
#import graphviz
import numpy as np
#import pydotplus 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

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
boundary_x_min = min([dataPoint[0] for dataPoint in trainingData]) - 1
boundary_x_max = max([dataPoint[0] for dataPoint in trainingData]) + 1
boundary_x_range = Range1d(boundary_x_min, boundary_x_max, bounds = (boundary_x_min, boundary_x_max))

boundary_y_min = min([dataPoint[1] for dataPoint in trainingData]) - 1
boundary_y_max = max([dataPoint[1] for dataPoint in trainingData]) + 1
boundary_y_range = Range1d(boundary_y_min, boundary_y_max, bounds = (boundary_y_min, boundary_y_max))


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
    xx, yy = np.meshgrid(np.arange(boundary_x_min, boundary_x_max, 0.01), 
                         np.arange(boundary_y_min, boundary_y_max, 0.01))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    
    # Create bokeh figure
    bokeh_plot = figure(plot_width=500,
                        plot_height=500,
                        x_range = boundary_x_range,
                        y_range = boundary_y_range
                        )
    
    # Plot the mesh grid on the bokeh figure as an image
    # Note: "Z" here is the mesh grid predictions, and it must be contained in a list (i.e. "[Z]" not "Z")
    bokeh_plot.image(image=[zz],
                     alpha = 0,
                     x = boundary_x_min,           
                     y = boundary_y_min,
                     dw =(boundary_x_max - boundary_x_min),
                     dh =(boundary_y_max - boundary_y_min),
                     palette = light_palette
                    )
    
    # Plot data points in the label_0 bucket
    bokeh_plot.circle([trainingData[i][0] for i in label_0], 
                      [trainingData[i][1] for i in label_0],
                      size = 4,
                      fill_color = dark_palette[0],
                      line_color = dark_palette[0],
                      legend = labelNames[0]
                     )
    
    # Plot data points in the label_1 bucket
    bokeh_plot.circle([trainingData[i][0] for i in label_1], 
                      [trainingData[i][1] for i in label_1],
                      size = 4,
                      fill_color = dark_palette[1],
                      line_color = dark_palette[1],
                      legend = labelNames[1]
                     )
    
    # Plot data points in the label_2 bucket
    bokeh_plot.circle([trainingData[i][0] for i in label_2], 
                      [trainingData[i][1] for i in label_2],
                      size = 4,
                      fill_color = dark_palette[2],
                      line_color = dark_palette[2],
                      legend = labelNames[2]
                     )
    
    # Label axes, place legend
    bokeh_plot.xaxis.axis_label = featureNames[0]
    bokeh_plot.yaxis.axis_label = featureNames[1]
    bokeh_plot.legend.location = "bottom_left"
    title = "When K = {}".format(j)
    bokeh_plot.title.text = title

    output_file("test{}.html".format(j))
    save(bokeh_plot)

