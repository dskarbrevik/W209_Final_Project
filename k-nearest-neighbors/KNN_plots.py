from bokeh.plotting import figure, output_file, show, reset_output
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from bokeh.charts import Scatter, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.sampledata.autompg import autompg as df
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import widgetbox, layout, gridplot, row
from bokeh.models.widgets import Slider
from bokeh.models.mappers import LinearColorMapper

#np.set_printoptions(threshold=np.inf)

# load the data
iris = datasets.load_iris()

# data starts sorted by class so randomize it.
shuffle = np.random.permutation(np.arange(iris.data.shape[0]))
iris.data, iris.target = iris.data[shuffle], iris.target[shuffle]

train_data = iris.data[:120,:2]
train_labels = iris.target[:120]
test_data = iris.data[120:,:2]
test_labels = iris.target[120:]

# split training and test data
#for x in range(len(iris.data)):	       
#    if np.random.rand() < .8:	           
#        train_data = np.append(train_data, iris.data[x])
#        train_labels = np.append(train_labels, iris.target[x])                
#    else:	            
#        test_data = np.append(test_data, iris.data[x])
#        test_labels = np.append(test_labels, iris.target[x])

#type(train_data)
#type(train_labels)
#type(test_data)
#type(test_labels)
#
#def KNN(k_values):
#    
#    for k in k_values:
#        kneighbor = KNeighborsClassifier() # create the classifier
#        kneighbor.fit(train_data, train_labels) # train the classifier
#        preds = kneighbor.predict(test_data) # test the classifier
#    
#    
#
#    print("\n")
#    print("Diagnoistics for k-nearest neighbor = 1 model:")
#    print("\n")
#    target_names = ["setosa", "versicolor", "virginica"]
#    print(classification_report(test_labels, preds, target_names=target_names))
#    
#k_values = [1,2,3,5,7,9]
#KNN(k_values)


#################################################
# SIMPLY TRAIN THE MODEL
#################################################


kneighbor = KNeighborsClassifier(n_neighbors=100) # create the classifier
kneighbor.fit(train_data[:,:2], train_labels) # train the classifier
#preds = kneighbor.predict(test_data) # test the classifier






# prepare some data
x = train_data[:, 0]
y = train_data[:, 1]


x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .001),
                     np.arange(y_min, y_max, .001))


Z = kneighbor.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

cmap_light = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

from bokeh.palettes import Category20
light_palette = [Category20[6][2*i + 1] for i in range(3)]
dark_palette = [Category20[6][2*i] for i in range(3)]

p = figure()
p.image(image=[Z],x=x_min, y=y_min, dw=(x_max-x_min), dh=(y_max-y_min), palette=light_palette)


#source = get_contour_data(xx,yy,Z)
#
#
#p = figure()
#p.multi_line(xs='xs', ys='ys', line_color='line_color', source=source)
colormap = {0: 'red', 1: 'green', 2: 'blue'}
colors = [dark_palette[x] for x in train_labels]
p.circle(x,y, color=colors, radius=.02)
##show(p)
#
slider = widgetbox(Slider(start=0, end=10, value=1, step=1, title="K = ?"))
##reset_output
l = row(slider,p,)
#
show(l)
# output to static HTML file
#output_file("lines.html")


