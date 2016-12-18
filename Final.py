from __future__ import division
import copy
import math
from sklearn import cluster
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn import decomposition
from sklearn import datasets


color_dict = {0: 'r', 1: 'g', 2: 'c', 3: 'm', 4: 'y', 5: 'b', 6: 'k'}


x_train = np.loadtxt(r"C:\Users\Ravinder\Documents\Smith\CSC390\Final_Project\UCI_HAR_Dataset\train\X_train.txt") #is data
y_train = np.loadtxt(r"C:\Users\Ravinder\Documents\Smith\CSC390\Final_Project\UCI_HAR_Dataset\train\y_train.txt") #is label
x_test = np.loadtxt(r"C:\Users\Ravinder\Documents\Smith\CSC390\Final_Project\UCI_HAR_Dataset\test\X_test.txt") #is data
y_test = np.loadtxt(r"C:\Users\Ravinder\Documents\Smith\CSC390\Final_Project\UCI_HAR_Dataset\test\y_test.txt") #is label

#---------
# HELPERS
#---------
def filter(dataset,y_list):
    """Filter the data to keep only those examples with the desired labels
       Note that this modifies the dataset"""
    data_filter = []
    y_filter = []
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0

    for i in range(len(dataset)):
        if y_list[i] == 1 and count1 <200:
            data_filter.append(dataset[i])
            y_filter.append(y_list[i])
            count1 = count1+1
        elif y_list[i] == 2 and count2 <200:
            data_filter.append(dataset[i])
            y_filter.append(y_list[i])
            count2 = count2+1
        elif y_list[i] == 3 and count3 <200:
            data_filter.append(dataset[i])
            y_filter.append(y_list[i])
            count3 = count3+1
        elif y_list[i] == 4 and count4 <200:
            data_filter.append(dataset[i])
            y_filter.append(y_list[i])
            count4 = count4+1
        elif y_list[i] == 5 and count5 <200:
            data_filter.append(dataset[i])
            y_filter.append(y_list[i])
            count5 = count5+1
        elif y_list[i] == 6 and count6 <200:
            data_filter.append(dataset[i])
            y_filter.append(y_list[i])
            count6 = count6+1

    return np.array(data_filter), np.array(y_filter)


def clusterAccuracy(true_Y, pred_Y, relabel_dict):
    """First relabel the clusters, then compute the fraction of times we were correct"""
    m = len(true_Y)
    assert len(pred_Y) == m
    return [int(true_Y[i] == relabel_dict[pred_Y[i]]) for i in range(m)].count(1)/float(m)


# returns a dictionary that maps pred label to true label
def relabelClasses(true_Y, pred_Y, classes):
    """Relabel the k-means classes to reflect the true label of the majority of examples
       assigned to each cluster"""

    relabel_dict = {}
    remaining_classes = range(len(classes))

    # loop over the classes from k-means, reassigning each one
    for c in classes:
        # create a list of the TRUE labels for those examples assigned to class c
        assigned_class = []
        for i in range(len(pred_Y)):
            if pred_Y[i] == c:
                assigned_class.append(true_Y[i])

        # counts of each true label assigned class c
        label_counts = [assigned_class.count(x) for x in remaining_classes]
        new_label_index = label_counts.index(max(label_counts))
        new_label = remaining_classes[new_label_index]

        relabel_dict[c] = new_label
        remaining_classes.remove(new_label)

    return relabel_dict


def runKMeans(train, y_train, test, y_test, num_classes):
    """Run k-means and evaluate on both train and test data"""

    # separate into "X" and "Y"
    train_X = train
    train_Y = y_train
    test_X  = test
    test_Y  = y_test

    # train k-means
    kmeans = cluster.KMeans(n_clusters=num_classes)
    kmeans.fit(train_X)
    Y_pred_train = kmeans.predict(train_X) #use this for color
    Y_pred_test  = kmeans.predict(test_X)

    relabel_dict = relabelClasses(train_Y, Y_pred_train, range(num_classes))
    print('relabeling: ' + str(relabel_dict))
    print('train accuracy: ' + str(clusterAccuracy(train_Y, Y_pred_train, relabel_dict)))
    print('test accuracy: ' + str(clusterAccuracy(test_Y, Y_pred_test, relabel_dict)))
    return Y_pred_train

# compute the Euclidean distance squared between vectors x1 and x2
def distSquared(x1,x2):
    p = len(x1)
    assert len(x2) == p
    return sum([pow(x1[i]-x2[i],2) for i in range(p)])


# compute the within-cluster sum of squares
def withinClusterSS(means, data, clust_labels):
	k = len(means)
	wc_ss = 0 # initialize our sum of squares

	# loop through all our data points
	m = len(data)
	assert len(clust_labels) == m
	for j in range(m):
		i = clust_labels[j] # assigned the ith cluster
		wc_ss += distSquared(data[j],means[i]) # add to our sum of squares

	return wc_ss




#------
# MAIN
#------

#new_data=filter(x_train,y_train)[0]
#new_y=filter(x_train,y_train)[1]

#print(data_walking)
#pca = decomposition.PCA(n_components=2)
#pca.fit(data_walking)
#X_transform = pca.transform(data_walking)
#
## list comprehension for colors
#colors = [color_dict[y_value] for y_value in y_train]
##
## create a scatter plot
#plt.scatter(data_walking[:,0], data_walking[:,1], c=colors)


#k_means = runKMeans(new_data, new_y, x_test, y_test, 6)
k_means = runKMeans(x_train, y_train, x_test, y_test, 2)
#PCA
pca = decomposition.PCA(n_components=3)
pca.fit(x_train)
X_transform = pca.transform(x_train)

###PCA
#pca = decomposition.PCA(n_components=2)
#pca.fit(new_data)
#X_transform = pca.transform(new_data)

# list comprehension for colors
colors = [color_dict[y_value] for y_value in k_means]

# create a scatter plot
#plt.scatter(X_transform[:,0], X_transform[:,1], c=colors)
## create legend
#names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
## create legend
#leg_objects = []
#for i in range(6):
#    circle, = plt.plot([], color_dict[i] + 'o')
#    leg_objects.append(circle)
#plt.legend(leg_objects,names)

#labels_array = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
#labels_array = ['W', 'WU', 'WD', 'Sit', 'Sta', 'L']

#count=0
#for point in X_transform:
##    plt.annotate(labels_array[int(y_train[count]-1)], xy=(X_transform[count,0], X_transform[count,1]), xytext=(X_transform[count,0],X_transform[count,1]),)
#    count+=1

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=20)
ax.scatter(X_transform[:,0], X_transform[:,1], X_transform[:,2], c=colors, marker='+', alpha=.4)
 #store total sum of squares (k=1)
#total_ss = None
#K_MAX = 12
## keep track of fraction of variance explained
#frac_var_explained = []
#
#for k in range(1,K_MAX+1): # for k=1,2,...,12
#	kmeans = cluster.KMeans(n_clusters=k)
#	kmeans.fit(X_transform)
#	clust_labels = kmeans.predict(X_transform)
#	means = kmeans.cluster_centers_
#
#	# compute within-cluster sum of squares
#	wc_ss = withinClusterSS(means, X_transform, clust_labels)
#	#print(wc_ss)
#
#	if k==1:
#		total_ss = wc_ss
#
#	# fraction of variance explained
#	fv = (total_ss - wc_ss) / total_ss
#	frac_var_explained.append(fv)
#
## elbow plot (comment out plotting above to run this)
#plt.plot(range(1,K_MAX+1), frac_var_explained, 'bo-')



plt.show()

