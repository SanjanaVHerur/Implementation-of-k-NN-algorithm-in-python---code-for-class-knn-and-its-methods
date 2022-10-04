import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """ 
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        #r1,c1=self.data.shape
        #r2,c2=x.shape
        r1=len(self.data)
        c1=len(self.data[0])
        r2=len(x)
        if(type(x[0])==np.ndarray):
            c2=len(x[0])
        else:
            c2=r2
            r2=1
        dist=0
        k=0
        t = [[0]*r1 for i in range(r2)]
        if(type(x[0])!=np.ndarray):
            while(k!=r2):
                for i in range(r1):
                    for j in range(c1):
                        dist+=((abs(self.data[i][j]-x[j]))**self.p)
                    dist=(dist)**(1/self.p)
                    t[k][i]=dist
                    dist=0
                k+=1
        else:
            while(k!=r2):
                for i in range(r1):
                    for j in range(c1):
                        dist+=((abs(self.data[i][j]-x[k][j]))**self.p)
                    dist=(dist)**(1/self.p)
                    t[k][i]=dist
                    dist=0
                k+=1
        return t
        pass

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        t=self.find_distance(x)
        r1=len(t)
        if(type(t[0])==list):
            c1=len(t[0])
        else:
            c1=r1
            r1=1
        r2=len(self.data)
        c2=len(self.data[0])
        neigh_dists=[ [0]*self.k_neigh for i in range(r1)]
        idx_of_neigh=[ [0]*self.k_neigh for i in range(r1)]
        i=0
        d1={}
        while(i!=r1):
            a=t[i]
            for j in range(r2):
                d1[a[j]]=j
            l1=a
            l1.sort()
            for j in range(self.k_neigh):
                neigh_dists[i][j]=l1[j]
                idx_of_neigh[i][j]=d1[l1[j]]
            i+=1
            d1={}
        l2=[neigh_dists,idx_of_neigh]
        return l2
        pass

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        l1=self.k_neighbours(x)
        index=l1[1]
        r1=len(index)
        if(type(index[0])==list):
            c1=len(index[0])
        else:
            c1=r1
            r1=1
        k=0
        d1={}
        l=[0]*r1
        u=np.unique(self.target)
        if(self.weighted==True):
            dist=l1[0]
            wgt=dist
            for i in range(r1):
                for j in range(c1):
                    wgt[i][j]=(1/dist[i][j])
            for i in range(r1):
                for m in u:
                    d1[m]=0
                for j in range(c1):
                    k=index[i][j]
                    d1[self.target[k]]+=wgt[i][j]
                l[i]=max(d1,key=d1.get)
        else:
            for i in range(r1):
                for m in u:
                    d1[m]=0
                for j in range(c1):
                    k=index[i][j]
                    d1[self.target[k]]+=1
                l[i]=max(d1,key=d1.get)
        return l
        pass

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        l=self.predict(x)
        n=len(l)
        s=0
        for i in range(n):
            if(l[i]==y[i]):
                s+=1
        acc=(s/n)
        return acc
        pass
