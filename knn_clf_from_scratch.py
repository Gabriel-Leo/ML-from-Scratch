import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter

'-----------------------------------------------------------------------------------------------------------------------------------------'
'-----------------------------------------------------------------------------------------------------------------------------------------'
# Distance metrics that can be applied with the KNN Classification Algorithm
def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q))**2))

def manhattan_distance(p, q):
    return np.sum(abs((np.array(p) - np.array(q))))

def chebyshev_distance(p, q):
    return max(abs((np.array(p) - np.array(q))))

# values closer to 1 indicate more similarity, while values closer to 0 indicate less similarity. Negative values indicate opposite direction
def cosine_similarity(p, q):
    A = np.array(p)
    B = np.array(q)
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# between 0 and 2, lower values indicate closer distance (more similarity)
def cosine_distance(p, q):
    return 1 - (cosine_similarity(p, q))

distance_functions = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'chebyshev': chebyshev_distance,
    'cosine': cosine_distance
}


class KNearestNeighbors:
    
    # k = number ok neighbors to look at
    # weighted = True: allows for closer points to have more weight in the voting
    # options for distance: 'euclidean', 'manhattan', 'chebyshev', 'cosine'

    def __init__(self, k=3, weighted=False, distance='euclidean'):
        self.k = k
        self.point = None
        self.weighted = weighted
        self.distance = distance

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        
        # keep a record of the distances
        distances = []
        distance_func = distance_functions[self.distance]

        # for each class/category in the data...
        for category in self.points:
            # ...for each point in the class/category...
            for point in self.points[category]:
                # ...find the distance to each pont and its category
                distance = distance_func(point, new_point)
                distances.append([distance, category])

        if self.weighted == False:
            
            # get k categories for the shortest distances
            categories = [category[1] for category in sorted(distances)[:self.k]]
            
            # find the most repeated category
            result = Counter(categories).most_common(1)[0][0]
        
        if self.weighted == True:
            
            # Calculate inverse distances
            weights = [[1/distance[0], distance[1]] for distance in sorted(distances)[:self.k]]
            
            # Add the votes
            weighted_votes = dict()
            for weight in weights:
                # Add the category if not present
                if weight[1] not in weighted_votes:
                    weighted_votes[weight[1]] = 0
                # Add vote weight to the category
                weighted_votes[weight[1]] = weight[0]

            result = max(weighted_votes)

        return result

'-----------------------------------------------------------------------------------------------------------------------------------------'
'-----------------------------------------------------------------------------------------------------------------------------------------'

# Allows to visualize how the model classifies a single new data point 
def visualize_knn(clf, points, new_point):

    ax = plt.subplot()

    #set colors for the graph
    ax.grid(True, color='#323232')
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    #create a dictionary with classes associated to a color
    color_list = ['blue', 'red', 'green', 'pink', 'purple', 'cyan', 'orange', 'yellow']
    class_colors = {key: color_list[i] for i, key in enumerate(points)}

    # scatter data points
    for key in points.keys():
        [ax.scatter(point[0], point[1], color=class_colors[key], s=60) for point in points[key]]

    # scatter predicted point
    new_class = clf.predict(new_point)
    ax.scatter(new_point[0], new_point[1], color=class_colors[new_class], marker='*', s=200, zorder=100)

    # add lines
    for key in points.keys():
        [ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color=class_colors[key], linestyle='--', linewidth=1) for point in points[key]]

    # add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[key], markersize=10, label=key) for key in points.keys()]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.show()
    
'---------------------------------------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------------------------------------------------------------------------'

def main():

    # These points represent the training data, and the color is the 'class' of the data point. [x, y] represents the dimonsional values
    points = {'Group_A': [[2,4], [1,3], [2,3], [3,2], [2,1]],
            'Group_B': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

    # This point work as test data
    new_point = [3,3]

    clf = KNearestNeighbors(k=3, distance='euclidean', weighted=True)
    clf.fit(points)
    print(clf.predict(new_point))

    visualize_knn(clf, points, new_point)

main()