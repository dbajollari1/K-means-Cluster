import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class K_Means:
    def __init__(self,k,data):
        self.k = k
        self.data = data  
        
    def initialise_centroids(self,k,data):
        self.centroids = data[:k]
        return self.centroids    
 
    def fit(self,data):
        m = np.shape(data)[0]
        cluster_assignments = np.mat(np.zeros((m,2)))
        
        cents = self.initialise_centroids(self.k,data)
        
        # Preserve original centroids
        cents_orig = cents.copy()
        changed = True
        num_iter = 0
        
        while changed and num_iter<100:
            changed = False 
            # for each row in the dataset
            for i in range(m):
                # Track minimum distance and vector index of associated cluster
                min_dist = np.inf
                min_index = -1 
                #calculate distance 
                for j in range(self.k):
                    dist_ji = euclidean_dist(cents[j,:],data[i,:])
                    if(dist_ji < min_dist):
                        min_dist = dist_ji
                        min_index = j 
                    # Check if cluster assignment of instance has changed
                    if cluster_assignments[i, 0] != min_index: 
                        changed = True

                # Assign instance to appropriate cluster
                cluster_assignments[i, :] = min_index, min_dist**2

            # Update centroid location
            for cent in range(self.k):
                points = data[np.nonzero(cluster_assignments[:,0].A==cent)[0]]
                cents[cent,:] = np.mean(points, axis=0)
    
            # Count iterations
            num_iter += 1
            #print(num_iter)

         # Return important stuff when done
        return cents, cluster_assignments, num_iter, cents_orig

def plot(data,k,index,centroids,orig_centroids):
    print('ploting')
    input = []
    for i in range(len(index)):
        for j in index[i]:
            input.append(int(j))
            
    colors = 10*["g","r","c","b","k"]
    j=0
    for i in input:
        plt.scatter(data[j,0], data[j,1], marker="x", color=colors[i], s=150, linewidths=5)
        j+=1
    ## New centroids
    for centroid in range(len(centroids)):
        plt.scatter(centroids[centroid][0],centroids[centroid][1],marker="o", color="k", s=150, linewidths=5)
    # Original Clusters
    for centroid in range(len(orig_centroids)):
        plt.scatter(orig_centroids[centroid][0],orig_centroids[centroid][1],marker="D", color="DarkBlue", s=150, linewidths=5)
    plt.show()

def elbow(data):
    costs = []
    for i in range(10):
        kmeans = K_Means(k=i,data = data)
        centroids, cluster_assignments, iters, orig_centroids = kmeans.fit(data)
        distance = cluster_assignments[:,1]  ## This has the distance from their respective centroides for evaluation purposes 
        cost = sum(distance)/(2*len(data))
        cost = np.array(cost)
        cost =  cost.item()
        costs.append(cost)
        
    x = np.arange(10)
    plt.plot(x,costs)
    plt.title("Elbow curve")
    plt.xlabel("K -->")
    plt.ylabel("Dispersion")
    plt.show()

def preProcessData():
    df = pd.read_csv('income.csv')

    #replacing missing features
    missing_features = ['workclass', 'occupation','native-country']
    for column in missing_features: 
        mostUsed = df[column].value_counts().idxmax()
        df[column] = df[column].replace([' ?'],mostUsed)

    #replacing categorical features with numeric values
    categorical_features = ['workclass','marital-status','occupation','relationship','race','sex','native-country']
    for column in categorical_features: 
        df[column] = pd.factorize(df[column])[0]

    df = df.drop('education', 1) #drop education column as we have education-numbers
    
    #normalizing data 
    normalizing_features = ['age','workclass','fnlwgt','education-num','marital-status','occupation','relationship','race','capital-gain','capital-loss','hours-per-week','native-country']
    for column in normalizing_features: 
        df[column] = (df[column] - np.min(df[column])) / (np.max(df[column]) - np.min(df[column]))

    #print(df.head(10))
    return df

def main():
    
    df = preProcessData()
    #print(df.head(10))
    data = df.to_numpy() #df.values

    # kmeans = K_Means(k=3,data = data)
    # centroids, cluster_assignments, iters, orig_centroids = kmeans.fit(data)
    # index = cluster_assignments[:,0] ## This has the cluster assignment 0,1,.... 
    # distance = cluster_assignments[:,1]  ## This has the distance from their respective centroides for evaluation purposes 
    # k=3
    # plot(data,k,index,centroids,orig_centroids)
    elbow(data)


if __name__ == "__main__":
    main()

