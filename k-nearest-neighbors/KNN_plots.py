from bokeh.plotting import figure, output_file, show
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from bokeh.charts import Scatter, output_file, show
from bokeh.sampledata.autompg import autompg as df

np.set_printoptions(threshold=np.inf)

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

def KNN(k_values):
    
    for k in k_values:
        kneighbor = KNeighborsClassifier() # create the classifier
        kneighbor.fit(train_data, train_labels) # train the classifier
        preds = kneighbor.predict(test_data) # test the classifier
    
    

    print("\n")
    print("Diagnoistics for k-nearest neighbor = 1 model:")
    print("\n")
    target_names = ["setosa", "versicolor", "virginica"]
    print(classification_report(test_labels, preds, target_names=target_names))
    
k_values = [1,2,3,5,7,9]
KNN(k_values)





kneighbor = KNeighborsClassifier() # create the classifier
kneighbor.fit(train_data, train_labels) # train the classifier
preds = kneighbor.predict(test_data) # test the classifier

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

h = 0.01 #step size in mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
y_min, y_max = test_data[:, 1].min() - 1, test_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
preds = kneighbor.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
preds = preds.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, preds, cmap=cmap_light)

# Plot also the training points
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.title("3-Class classification (k = %i, weights = '%s')"
#          % (n_neighbors, weights))


plt.show()



from bokeh.sampledata.iris import flowers as df
df.shape
type(df)

test_labels.shape
test_labels[1:10]
colormap[0]
df[2]

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

# output to static HTML file
output_file("lines.html")

# create a new plot with a title and axis labels
colormap = {0: 'red', 1: 'green', 2: 'blue'}
colors = [colormap[x] for x in train_labels]
p = figure()
p.circle(x, y, color=colors, radius=.02)

# add a line renderer with legend and line thickness
#p.line(x, y, legend="Temp.", line_width=10)

# show the results
show(p)
