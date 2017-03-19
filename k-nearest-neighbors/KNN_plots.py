from bokeh.plotting import figure, output_file, show
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from bokeh.charts import Scatter, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.sampledata.autompg import autompg as df

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
#
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


kneighbor = KNeighborsClassifier() # create the classifier
kneighbor.fit(train_data[:,:2], train_labels) # train the classifier
#preds = kneighbor.predict(test_data) # test the classifier

#################################################
# MATPLOTLIB IMPLEMENTATION
#################################################
#cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#h = 0.01 #step size in mesh
#
## Plot the decision boundary. For that, we will assign a color to each
## point in the mesh [x_min, x_max]x[y_min, y_max].
#x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
#y_min, y_max = test_data[:, 1].min() - 1, test_data[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                     np.arange(y_min, y_max, h))
#preds = kneighbor.predict(np.c_[xx.ravel(), yy.ravel()])
#
## Put the result into a color plot
#preds = preds.reshape(xx.shape)
#
#plt.figure()
#plt.pcolormesh(xx, yy, preds, cmap=cmap_light)
#
## Plot also the training points
#plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=cmap_bold)
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
##plt.title("3-Class classification (k = %i, weights = '%s')"
##          % (n_neighbors, weights))
#
#
#plt.show()

###########################################

### Stole this def that extracts matplotlib data

def get_contour_data(X, Y, Z):
    cs = plt.contour(X, Y, Z)
    xs = []
    ys = []
    xt = []
    yt = []
    col = []
    text = []
    isolevelid = 0
    for isolevel in cs.collections:
        isocol = isolevel.get_color()[0]
        thecol = 3 * [None]
        theiso = str(cs.get_array()[isolevelid])
        isolevelid += 1
        for i in range(3):
            thecol[i] = int(255 * isocol[i])
        thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])

        for path in isolevel.get_paths():
            v = path.vertices
            x = v[:, 0]
            y = v[:, 1]
            xs.append(x.tolist())
            ys.append(y.tolist())
            xt.append(x[len(x) / 2])
            yt.append(y[len(y) / 2])
            text.append(theiso)
            col.append(thecol)

    source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': col,'xt':xt,'yt':yt,'text':text})
    return source

##############################################

#from bokeh.sampledata.iris import flowers as df


# prepare some data
x = train_data[:, 0]
y = train_data[:, 1]


x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))


Z = kneighbor.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

source = get_contour_data(xx,yy,Z)


p = figure()
p.multi_line(xs='xs', ys='ys', line_color='line_color', source=source)
colormap = {0: 'red', 1: 'green', 2: 'blue'}
colors = [colormap[x] for x in train_labels]
p.circle(x,y, color=colors, radius=.02)
show(p)

# output to static HTML file
#output_file("lines.html")


