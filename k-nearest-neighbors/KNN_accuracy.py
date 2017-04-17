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
from sklearn.metrics import accuracy_score

import os

os.chdir("C:\\Users\\skarb\\Desktop\\Github\\W209_Final_Project\\k-nearest-neighbors\\")

# Load iris data
iris = load_iris()

np.random.seed(1)

# data starts sorted by class so randomize it.
shuffle = np.random.permutation(np.arange(iris.data.shape[0]))
iris.data, iris.target = iris.data[shuffle], iris.target[shuffle]

train_data = iris.data[:120,:2]
train_labels = iris.target[:120]
test_data = iris.data[120:,:2]
test_labels = iris.target[120:]


accuracy = []

for i in range(1,11):
    # Train a model on the first two features
    model = KNeighborsClassifier(n_neighbors=i) # create the classifier
    model.fit(train_data, train_labels) # train the classifier
    preds = model.predict(test_data)
    accuracy.append(accuracy_score(test_labels, preds))


for i in range(1,11):
    
    bokeh_plot = figure(plot_width=500,
                        plot_height=500,
                        x_range = Range1d(0, 13, bounds = (0, 13)),
                        y_range = Range1d(-0.04, 1.04, bounds = (-0.04, 1.04)),
                        tools = "pan, box_zoom, wheel_zoom, reset, undo, redo"
                        )

    bokeh_plot.circle([(j+1) for j in range(i)], 
                      [accuracy[j] for j in range(i)],
                      size = 4
                     )

    bokeh_plot.line([(j+1) for j in range(i)], 
                    [accuracy[j] for j in range(i)],
                    line_width=1)
    
    bokeh_plot.xaxis.axis_label = "K-value"
    bokeh_plot.yaxis.axis_label = "Accuracy"
    
    #title_num = i +1

    output_file("accuracy-{}.html".format(i))
    save(bokeh_plot)
