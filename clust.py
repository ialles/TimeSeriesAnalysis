import numpy as np
import sim
import copy
from collections import defaultdict
import itertools

class Cluster:
    'Implements methods for HAC and k-means clustering'
    CLUSTERS = 0; MEANS = 1

    def __init__(self, sequences, num_cluster=2 ):
        self.sequences = sequences
        self.sim_matrix = None
        self.sim_matrix_eq_length = None
        self.name_dict = sequences.keys()
        self._clusters = defaultdict(list)
        self._num_clusters = num_cluster
        self._distance_data = []
        self._cluster_steps = [[],[]]
        self._verbose = False

    def create_sim_matrix(self):
        if self._verbose == True: print "Create_sim_matrix()"
        matrix = np.empty((len(self.name_dict), len(self.name_dict)))
        for i,item in enumerate (self.name_dict):
            for j, jtem in enumerate(self.name_dict[i:], i):
                b1 = np.asarray(
                                 self.sequences[item].biomass )
                b2 = np.asarray(
                                 self.sequences[jtem].biomass )
                res = sim.dtw_distance(b1.astype(np.float), b2.astype(np.float))
                #print "distance: ",res[len(res)-1,len(res[0])-1]
                matrix [i][j] = res[len(res)-1,len(res[0])-1]
                matrix [j] [i] = res[len(res)-1,len(res[0])-1]
        self.sim_matrix = matrix
        return matrix

    def create_sim_matrix_equal_length(self):
        if self._verbose == True: print "Create_sim_matrix()"
        matrix = np.empty((len(self.name_dict), len(self.name_dict)))
        for i,item in enumerate (self.name_dict):
            for j, jtem in enumerate(self.name_dict[i:], i):
                b1 = np.asarray(
                                 self.sequences[item].biomass, dtype=np.float)
                b2 = np.asarray(
                                 self.sequences[jtem].biomass, dtype=np.float)
                divlen = len(b1)
                if(len(b1)>len(b2)):
                    if np.shape(b2) == (2L,):
                        b2 = np.reshape(b2,(1,2))
                    #print "b2: ",b2, np.shape(b2)
                    minval = min(b1[:,0], key=lambda x:abs(x - b2[0,0]))
                    #print "minval: ",minval
                    minindex = np.where(b1[:,0] == minval)[0][0]
                    #print "minindex: ", minindex

                    maxval = min(b1[:,0], key=lambda x:abs(x - b2[-1,0]))
                    #print "maxval: ", maxval
                    maxindex = np.where(b1[:,0] == maxval)[0][0]
                    maxindex +=1
                    #print "min, max", minindex, maxindex, " vals: ", minval,maxval
                    #print "b1 : ",b1
                    b1 = b1[minindex:maxindex, :]
                    #print self.sequences[item].pot, b1
                elif(len(b1)<len(b2)):
                    divlen = len(b2)
                    if np.shape(b1) == (2L,):
                        b1 = np.reshape(b1,(1,2))
                    minval = min(b2[:,0], key=lambda x:abs(x- b1[0,0]))
                    #print "minval: ", minval
                    minindex = np.where(b2[:,0] == minval)[0][0]
                    #print "minindex: ",minindex

                    maxval = min(b2[:,0], key=lambda x:abs(x-b1[-1,0]))
                    #print "maxval: ",maxval
                    maxindex = np.where(b2[:,0] == maxval)[0][0]
                    maxindex +=1
                    #print "min, max", minindex, maxindex, " vals: ", minval,maxval
                    b2 = b2[minindex:maxindex, :]
                    #print self.sequences[jtem].pot, b2
                res = sim.dtw_distance(b1.astype(np.float), b2.astype(np.float))
                #print "distance: ",res[len(res)-1,len(res[0])-1]
                matrix.itemset((i,j),res[-1,-1])
                matrix.itemset((j,i),(res[-1,-1]))
            self.sim_matrix_eq_length = matrix
        return matrix

    def create_sim_matrix_weighted_equal_length(self, startweight = 0.6, endweight = 1.3):
        if self._verbose == True: print "Create_sim_matrix()"
        matrix = np.empty((len(self.name_dict), len(self.name_dict)))
        for i,item in enumerate (self.name_dict):
            for j, jtem in enumerate(self.name_dict[i:], i):
                b1 = np.asarray(
                                 self.sequences[item].biomass, dtype=np.float)
                b2 = np.asarray(
                                 self.sequences[jtem].biomass, dtype=np.float)
                if(len(b1)>len(b2)):
                    if np.shape(b2) == (2L,):
                        b2 = np.reshape(b2,(1,2))
                    minval = min(b1[:,0], key=lambda x:abs(x - b2[0,0]))
                    minindex = np.where(b1[:,0] == minval)[0][0]
                    maxval = min(b1[:,0], key=lambda x:abs(x - b2[-1,0]))
                    maxindex = np.where(b1[:,0] == maxval)[0][0]
                    maxindex +=1
                    b1 = b1[minindex:maxindex, :]
                elif(len(b1)<len(b2)):
                    if np.shape(b1) == (2L,):
                        b1 = np.reshape(b1,(1,2))
                    minval = min(b2[:,0], key=lambda x:abs(x- b1[0,0]))
                    minindex = np.where(b2[:,0] == minval)[0][0]
                    maxval = min(b2[:,0], key=lambda x:abs(x-b1[-1,0]))
                    maxindex = np.where(b2[:,0] == maxval)[0][0]
                    maxindex +=1
                    b2 = b2[minindex:maxindex, :]
                res = sim.dtw_weighted_distance(b1.astype(np.float), b2.astype(np.float), start_w=startweight, end_w=endweight)
                matrix.itemset((i,j),res.item(-1,-1))
                matrix.itemset((j,i), res.item(-1,-1))
            self.sim_matrix_eq_length = matrix
        return matrix

    

    def extract_WW_matrix(self, sim_matrix, sequences, namedict):
        ww_seq = []
        for k,v in sequences.iteritems():
            if v.scenario == 'WW':
                ww_seq.append(k)
        croppedmtr = np.zeros((len(ww_seq), len(ww_seq)))
        ww_index = np.zeros((len(ww_seq)), dtype=np.integer)
        #print len(ww_seq), len(ww_index)
        
        for i,v in enumerate (ww_seq):
            #print i,v
            ww_index[i]= np.where(np.asarray(namedict, dtype=np.integer) == int(v))[0][0]
        print ww_index
        for ci, k in enumerate(ww_index):
            #print i, np.shape(croppedmtr)
            #for cj,j in enumerate(ww_index):
               # croppedmtr[i,j] = sim_matrix[i,j]
                #print i, j, sim_matrix[i,j]
            croppedmtr[ci] = np.asarray([sim_matrix[k,j] for j in ww_index ])
        return croppedmtr, ww_seq

    def clear_object(self):
        self._clusters.clear()
        self._distance_data = []
        self._cluster_steps = [[],[]]

    def transform_to_dist_matrix(self):
        if self._verbose == True: print "transform_to_dist_matrix(self)"
        matrix = copy.copy(self.sim_matrix)
        matrix[np.diag_indices_from (matrix)] = float("inf")
        return matrix

    def get_smallest_dist(self, matrix):
        if self._verbose == True: print "get_smallest_dist()"
        i, j = np.unravel_index(np.nanargmin(matrix), matrix.shape)
        return matrix[i,j], i, j

    def hac(self, method):
        if self._verbose == True: print "hac("+str(method)+")"
        self.clear_object()
	# Initialize the clusters
	for i, pot in enumerate(self.name_dict):
			self._clusters[i].append(i)
	self._cluster_steps[Cluster.CLUSTERS].append(copy.deepcopy(self._clusters))
	self.calc_next_cluster_inertia_means(-1,-1)
        distance_mat = self.transform_to_dist_matrix()
        while len( self._clusters ) > max( self._num_clusters, 1 ):
            dist, i, j = self.get_smallest_dist(distance_mat)
            
            self._distance_data.append([len(self._clusters),
                                        self.calc_global_cluster_inertia()])
            distance_mat = self.update_matrix(distance_mat, i,j, method)
           
            for x in (self._clusters[j]):
                self._clusters[i].append(x)
            del self._clusters[j]
            self._cluster_steps[Cluster.CLUSTERS].append(copy.deepcopy(self._clusters))
            self.calc_next_cluster_inertia_means(i,j)
            distance_mat[:,j] = float("inf")
            distance_mat[j,:] = float("inf")
        print "hac done"

    def hac_ward(self):
        if self._verbose == True: print "hac_ward ..."
        self.clear_object()
	# Initialize the clusters
	for i, pot in enumerate(self.name_dict):
			self._clusters[i].append(i)
	self._cluster_steps[Cluster.CLUSTERS].append(copy.deepcopy(self._clusters))
	self.calc_next_cluster_inertia_means(-1,-1)
        distance_mat = self.transform_to_dist_matrix()
        while len( self._clusters ) > max( self._num_clusters, 1 ):
            dist, i, j = self.get_smallest_ess()
            
            self._distance_data.append([len(self._clusters), self.calc_global_cluster_inertia()])
           
            for x in (self._clusters[j]):
                self._clusters[i].append(x)
            del self._clusters[j]
            self._cluster_steps[Cluster.CLUSTERS].append(copy.deepcopy(self._clusters))
            self.calc_next_cluster_inertia_means(i,j)
        print "hac done"


    def calc_cluster_mean(self, cluster):
        if self._verbose == True: print "calc_cluster_mean()"
        if isinstance(cluster, list):
            temp = 0
            for i in range(len(cluster)):
                for j in range(i, len(cluster)):
                    if(cluster[i]!= cluster[j]):
                        temp += self.sim_matrix[cluster[i],cluster[j]]
            return temp/float(len(cluster))
        else:
            print "cluster has to be a list"
            

    def calc_next_cluster_inertia_means(self, i, j):
        if self._verbose == True: print "calc_next_cluster_inertia_means( i = "+str(i)+" j = "+str(j)
        if len(self._cluster_steps[Cluster.MEANS]) > 0:
            #print "\n means: ",self._cluster_steps[Cluster.MEANS][-1]
            means = self._cluster_steps[Cluster.MEANS][-1]
            if means[i] != 0 and means[j] != 0:
                new_i = (means[i] + means[j]) / float(2)
            else:
                new_i = self.calc_cluster_mean(self._cluster_steps[Cluster.CLUSTERS][-1][i])
            self._cluster_steps[Cluster.MEANS].append(copy.copy(self._cluster_steps[Cluster.MEANS][-1]))
            self._cluster_steps[Cluster.MEANS][-1][i]= new_i
            del self._cluster_steps[Cluster.MEANS][-1][j]
        else:
            # Initialize the means...
            self._cluster_steps[Cluster.MEANS].append({})
            #print "init \n self._cluster_steps[Cluster.MEANS]: ",self._cluster_steps[Cluster.MEANS]
            #print "type: ",type(self._cluster_steps[Cluster.MEANS][0])
            for n, el in self._clusters.items():
                self._cluster_steps[Cluster.MEANS][0][n] = 0

    def calc_global_cluster_inertia(self):
        if self._verbose == True: print "calc_global_cluster_inertia()"
        mean = 0
        temp = 0
        #print "Clusters.items: ",self._clusters.items()
        for n, cl in self._clusters.items():
            for i in range(len(cl)):
                for j in range(i, len(cl)):
                    if(cl[i]!=cl[j]):
                        #print "temp: += matrix ",cl[i],",",cl[j]," ",self.sim_matrix[cl[i],cl[j]]
                        temp += self.sim_matrix[cl[i],cl[j]]**2
            #print "temp: ",temp
            temp = temp/float(len(cl))
            #print "Divided by ",float(len(cl))," = ",temp
            mean += temp
            temp = 0
        #print "mean: ",mean," / ",float(len(self._clusters))
        mean = mean/float(len(self._clusters))
        if self._verbose == True: print "calc_global_cluster_inertia() done => "+str(mean)
        return mean

        
    def calc_minimum_dist(self, a, b):
        return min(a,b)

    def calc_mean_dist(self, a, b):
        return sum([a,b])/float(2)

    def calc_min(self, k, i, j, matrix, *clust):
        li = []
        li.append(matrix[i,k], matrix[j,k])
        for x in clust:
            li.append(matrix[i,x], matrix[j,x])
        return min(li)

    def update_matrix(self, matrix, i, j, method):
        if self._verbose == True: print "update_matrix()"
        for k in xrange(len(matrix)):
            if k!=j and k != i:
                #print "dist = ",matrix[i,k]," - ",matrix[j,k]," k = ",k
                if matrix[i,k] is float("inf") and matrix[j,k] is float("inf"):
                    dist = float("inf")
                else:
                    dist = method(matrix[i,k], matrix[j,k])
                matrix[i,k] = dist
                matrix[k,i] = dist
        return matrix

    def clustersize(self, clusters):
        if self._verbose == True: print "clustersize()"
	el = 0
	for x in clusters:
		if x is not None:
			el+=1
	return el

    def calc_sqrt_mean(self, cl):
        #print "list: ",cl
        temp = 0
        for i in range(len(cl)):
            # comparing the individual observations for each variable
            # against the cluster means for that variable
            for j in range(i, len(cl)):
                if(i != j):
                    #print "temp: += matrix ",cl[i],",",cl[j]," ",self.sim_matrix[cl[i],cl[j]]
                    temp += self.sim_matrix[cl[i],cl[j]]**2
        return temp

#TODO ... there's something going on ...
    def get_smallest_ess(self):
        clusters = self._cluster_steps[Cluster.CLUSTERS][-1]
        ess_dict = {}
        temp = 0
        # Step through all clusters
        cluster_list = clusters.items()
        for i, item in enumerate(cluster_list):
            for j in range(i, len(cluster_list)):
                if(i != j):
                    ki, vi = cluster_list[i]
                    kj, vj = cluster_list[j]
                    ess_dict[self.calc_sqrt_mean(np.asarray(vi+vj))] = [ki,kj]
        #print "keys",ess_dict.keys()

        min_dist = min(ess_dict.keys())
        #print ess_dict[min_dist]
        return min_dist, ess_dict[min_dist][0], ess_dict[min_dist][1]


    def dbscan(self, eps, min_pts):
        c = []
        seq_dict = defaultdict(int)
        # Use matrix here ...
        for key, val in enumerate(self.name_dict):
            if seq_dict[key] == 0:
                # this point has been visited
                seq_dict[key] +=1
                neighbours = self.region_query(key, eps, seq_dict)
                if (len(neighbours) < min_pts):
                    # this point is noise
                    seq_dict[key] = 13
                else:
                    self.expand_cluster(key, seq_dict, neighbours, c, eps, min_pts)
        #print "seq_dict: ",seq_dict
        #add noise cluster:
        noise = [x for x in seq_dict.keys() if seq_dict[x] == 13]
        #print "len seq dict", len(seq_dict)
        #print noise
        c.append(noise)
        return c

    def expand_cluster(self, key, seq_dict, neighbours, c, eps, min_pts):
        #print "expand cluster: key: ",key
        #print "neighbours: ",neighbours
        new_clust = [key]
        seq_dict[key] = 2
        numb = len(neighbours); i = 0
        while i < numb:
            #print "Check if ",neighbours[i], " is visited: ",seq_dict[neighbours[i]]
            # if i is not visited
            if seq_dict[neighbours[i]] == 0 :
                # i is visited now
                seq_dict[neighbours[i]] += 1
                #print neighbours[i]," is visited now: ", seq_dict[neighbours[i]]
                i_neighbours = self.region_query(neighbours[i], eps, seq_dict)
                if  len(i_neighbours) >= min_pts :
                    uniq = [x for x in i_neighbours if x not in neighbours]
                    #print "added neighbours: ",uniq
                    neighbours = neighbours + uniq
                    numb = len(neighbours)
            # check if i is already in a cluster
            if seq_dict[neighbours[i]] < 2 or seq_dict[neighbours[i]] == 13:
                #print neighbours[i]," is not in a cluster: ", seq_dict[neighbours[i]] 
                new_clust.append(neighbours[i])
                # Should be 2 no => part of cluster
                seq_dict[neighbours[i]] = 2
            i += 1
        c.append(copy.copy(new_clust))

#will be called for every point to check its neighbourhood -> distance_matrix necessary! Runtime complexity n^2
    def region_query(self, key, eps, seq_dict):
        neighbours = [i for i, item in enumerate(self.sim_matrix[key,:]) if item <= eps and key != i]
        #print "region_query: ",key, neighbours
        return neighbours
                    
            
           

                
                
                
            
        

    
