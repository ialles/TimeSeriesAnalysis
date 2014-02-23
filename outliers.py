import numpy as np
import sim
import manip
import clust
import copy
from collections import defaultdict
import itertools
import PlantDataAnalysis.manip as manip
import pylab
import random as rand
from collections import OrderedDict
import csv


relevant_names = ["Pot", "TT", "Biomass", "Bio2", "LeafArea",
                  "Genotype", "Scenario", "Genetic"]


def import_outlier_data(file_path, rows = None):
    data = manip.import_data_rows(file_path, ",", 2)
    data[0,:] = [x.strip('"') for x in data[0,:]]
    col_names = manip.extract_column_names(data, relevant_names)
    data = manip.import_data_rows(file_path, ",", rows_numb = rows, cols = col_names.values())
    data[0,:] = [x.strip('"') for x in data[0,:]]
    return data

# To be done ....change names ...
def extract_sequences_for_outliers(data_array, col_names):
    sequence_dict = {}
    for i,line in enumerate(data_array):
        try:
            el = sequence_dict[line[col_names['Pot']]]
            bm = el.biomass
            el.biomass = np.vstack([bm, [line[col_names['TT']],
                                                  line[col_names
                                                       ['Bio2']]]])
            bm_g = el.bio_gold
            el.bio_gold = np.vstack([bm_g, [line[col_names['TT']],
                                                  line[col_names
                                                       ['Biomass']]]])
        except:
            seq = manip.SequenceData(line[col_names['Pot']])
            seq.biomass = [line[col_names['TT']], line[col_names['Bio2']]]
            seq.bio_gold = [line[col_names['TT']], line[col_names['Biomass']]]
            seq.scenario = line[col_names['Scenario']]
            
            if 'Genetic' in col_names:
                    seq.genetic = line[col_names['Genetic']]
            else:
                seq.genetic = 'D'
                seq.genotype = line[col_names['Genotype']]
                sequence_dict.update ({line[col_names['Pot']]: seq})

    return sequence_dict

def calc_point_sim_matrix(tb_seq, costf=lambda x,y: np.linalg.norm(x - y) ):
    mtr = np.zeros((len(tb_seq),len(tb_seq)), dtype = float)
    for i,item in enumerate (tb_seq):
            for j, jtem in enumerate(tb_seq[i:], i):
                res = costf (jtem.astype(float), item.astype(float))
                mtr [i][j] = res
                mtr [j] [i] = res
    return mtr

def scan_sequences(seq, min_pts, epsf = lambda x: np.mean(x, dtype = float)*0.4):
    c = {}
    for k,v in seq.iteritems():
        eps = epsf(calc_point_sim_matrix(v.biomass))
        c.update({k:(dbscan(eps, 2, v.biomass, calc_point_sim_matrix(v.biomass)))})
    return c
    
def dbscan(eps, min_pts, tb_seq, sim_matrix):
    c = []
    seq_dict = defaultdict(int)
    # Use matrix here ...
    for i, val in enumerate(tb_seq):
        if seq_dict[i] == 0:
            # this point has been visited
            seq_dict[i] +=1
            neighbours = region_query(i, eps, sim_matrix)
            if (len(neighbours) < min_pts):
                # this point is noise
                seq_dict[i] = 13
            else:
                expand_cluster(i, seq_dict, neighbours, c, eps, min_pts, sim_matrix)
    #print "seq_dict: ",seq_dict
    #add noise cluster:
    noise = [x for x in seq_dict.keys() if seq_dict[x] == 13]
    print "len seq dict", len(seq_dict)
    print noise
    c.append(noise)
    return c

def expand_cluster(key, seq_dict, neighbours, c, eps, min_pts, sim_matrix):
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
            i_neighbours = region_query(neighbours[i], eps, sim_matrix)
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
def region_query(key, eps, sim_matrix):
    neighbours = [i for i, item in enumerate(sim_matrix[key,:]) if item <= eps and key != i]
    #print "region_query: ",key, neighbours
    return neighbours

def calc_prec_rec(seqs, cls):
    nas = 0; noise = 0; tp = 0
    for k,v in seqs.iteritems():
        v.calc_precision(cls[k])
        v.calc_recall(cls[k])
        nas += v.bio_nas
        tp += v.bio_tp
        noise += len(cls[k][-1])
    precision = (float(tp)/float(noise))
    recall = (float(tp)/float(nas))
    f1 = 2* (precision*recall)/(precision + recall)
    return precision, recall, f1
    

import matplotlib
import pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.mlab as mlab

def plot_outliers(seq, gold, cl, genotype, manip, seq_counter):
    # Create a figure with size 6 x 6 inches.
    fig = Figure(figsize=(9,6))

    # Create a canvas and add the figure to it.
    canvas = FigureCanvas(fig)

    # Create a subplot.
    ax = fig.add_subplot(111)

    # Set the title.
    ax.set_title(str(manip)+" "+str(genotype),fontsize=14)

    # Set the X Axis label.
    ax.set_xlabel('Time',fontsize=10)

    # Set the Y Axis label.
    ax.set_ylabel('Biomass',fontsize=10)

    # Display Grid.
    ax.grid(True, linestyle='-',color='0.75')
    
    ###############TODO print the clusters separately!##################
    cm = pylab.get_cmap('Dark2')
    NUM_COLORS = len(cl)
    colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
    res =[]
    for c in cl:
        r = [[] for i in range(2)]
        for el in c:
            if gold[el][1] == 'NA':
                r[1].append(seq[el])
            else:
                r[0].append(seq[el])
        res.append(r)

    for i,c in enumerate(res[:-1]):
        if len(c[0]) != 0:
            ax.scatter(np.asarray(c[0])[:,0].astype(float),np.asarray(c[0])[:,1].astype(float), s=15, color = "black")
        if len(c[1]) != 0:
            ax.scatter(np.asarray(c[1])[:,0].astype(float),np.asarray(c[1])[:,1].astype(float), s=15, color = "red")
    print len(res[-1])
    if len(res[-1][0]) > 0:
        ax.scatter(np.asarray(res[-1][0])[:,0].astype(float), np.asarray(res[-1][0])[:,1].astype(float), s=15, color='orange')
    if len(res[-1][1]) != 0:
            ax.scatter(np.asarray(res[-1][1])[:,0].astype(float),np.asarray(res[-1][1])[:,1].astype(float), s=15, color = "green")

    # Save the generated Scatter Plot to a PNG file.
    #canvas.draw()
    canvas.print_figure('D:\Master\Thesis\DEV\\'+"outliers_"+str(manip)+"_"+seq_counter+".png",dpi=500)
    
def plot_big_plant_outliers_scan(clusterobj, res, method_name, folder = ""):
    '''Plots the sequences of the given clusterobj into one plot, each cluster gets a different color/shape depending on the res parameter.
    The last cluster is considered as noise cluster and is printed in red'''
    if isinstance(clusterobj, clust.Cluster):
        cluster_nr = len(res)
        if len(res[-1]) == 0:
            cluster_nr -= 1
        output_file = "D:\Master\Thesis\DEV\plots\\"+folder+"Cluster_dbscan_big_"+method_name+"_seq"+str(len(clusterobj.sequences))#+"_clust"+str(cluster_nr)
        size = (len(clusterobj.sequences) - cluster_nr)
        NUM_COLORS = cluster_nr
	linestyles = ["-","--"]
        cm = pylab.get_cmap('Dark2')
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
        
        for nr, cl in enumerate(res[:-1]):
	    line = rand.randint(0,len(linestyles)-1)
            for el in cl:
                bio = np.asarray(clusterobj.sequences[clusterobj.name_dict[el]].biomass)
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1], linestyles[line], color = colors[nr],
                           label = str(nr))
        if len(res[-1]) > 0:
            line = rand.randint(0,len(linestyles)-1)
            for el in res[-1]:
                bio = np.asarray(clusterobj.sequences[clusterobj.name_dict[el]].biomass)
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1], linestyles[line], color = "red",
                         label = 'x'  )
            ax.patch.set_facecolor('red')
            ax.patch.set_alpha(0.2)
        handles, labels = ax.get_legend_handles_labels()
        #print clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario
        #ax.legend(handles,labels, loc=2)
        # Shink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
        fig.set_size_inches(20.5,12.5)
        pylab.xlabel('Time Steps')
        pylab.ylabel('BioVol')
        pylab.title('BioVol Clusters')
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusterobj.sequences))+" clusters: "+str(cluster_nr))
        pylab.savefig( output_file, dpi = 100)
        pylab.clf()
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusterobj)


def plot_sequences(clusterobj, method_name, folder = "", title = "BioVol Clusters", ylabel = 'BioVol', nolegend = True, color = "blue"):
    '''Plots the sequences of the given clusterobj into one plot'''
    if isinstance(clusterobj, clust.Cluster):
        output_file = "D:\Master\Thesis\DEV\plots\\"+folder+"Cluster_outlier_plant_"+method_name+"_seq"+str(len(clusterobj.sequences))#+"_clust"+str(cluster_nr)
        line = "-"
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for nr, cl in enumerate(clusterobj.sequences.itervalues()):
            bio = np.asarray(cl.biomass)
            if np.shape(bio) == (2,):
                bio = np.reshape(bio, (1,2))
            if color is None:
                pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr))
            else:
                pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr), color = color)
        if not nolegend:
            handles, labels = ax.get_legend_handles_labels()
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
            # Put a legend to the right of the current axis
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
        fig.set_size_inches(20.5,12.5)
        pylab.xlabel('Time Steps')
        pylab.ylabel(ylabel)
        pylab.title(title)
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusterobj.sequences)))
        pylab.savefig( output_file, dpi = 100)
        pylab.clf()
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusterobj)


def plot_sequences_bg(bg_sequences, clusterobj, method_name, folder = "", title = "BioVol Clusters", ylabel = 'BioVol', nolegend = True, color = "blue"):
    '''Plots the sequences of the given clusterobj into one plot,ads the bg_sequences as background'''
    if isinstance(clusterobj, clust.Cluster):
        print type(bg_sequences)
        output_file = "D:\Master\Thesis\DEV\plots\\"+folder+"Cluster_outlier_plant_"+method_name+"_seq"+str(len(clusterobj.sequences))#+"_clust"+str(cluster_nr)
    	#linestyles = ["-","--"]
        line = "-"
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        for nr, cl in enumerate(bg_sequences):
            bio = np.asarray(cl.biomass)
            if np.shape(bio) == (2,):
                bio = np.reshape(bio, (1,2))
            pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr), alpha = 0.1, color = "0.2")
        for nr, cl in enumerate(clusterobj.sequences.itervalues()):
            bio = np.asarray(cl.biomass)
            if np.shape(bio) == (2,):
                bio = np.reshape(bio, (1,2))
            if color is None:
                pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr))
            else:
                pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr), color = color)
        if not nolegend:
            handles, labels = ax.get_legend_handles_labels()
            # Shink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
            # Put a legend to the right of the current axis
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
        fig.set_size_inches(20.5,12.5)
        pylab.xlabel('Time Steps')
        pylab.ylabel(ylabel)
        pylab.title(title)
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusterobj.sequences)))
        pylab.savefig( output_file, dpi = 100)
        pylab.clf()
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusterobj)


def detect_plant_outlier(j, k, dist, gendict):
    outlierdict = {}
    for i,v in gendict.iteritems():
        firstgen = [x for x in v if x.scenario == 'WW']
        if len(firstgen) == 0: continue
        firstgenseq = {}
        firstgenavg = []
        for x in firstgen:
            firstgenseq.update({x.pot: x})
            firstgenavg.append((x.pot, (x.biomass[:,1][-1])))
            
        firstgenavg = np.array(firstgenavg, dtype= np.dtype([('pot', 'S10'),('avg', '>f4')]))
        firstgenavg.sort(order='avg')
        firstgenclust = clust.Cluster(firstgenseq)
        
        firstgenclust.sim_matrix = firstgenclust.create_sim_matrix_equal_length()
        dbres = [[],[]]
        #print np.asarray(firstgenclust.name_dict)
        #print firstgenavg['pot'][0]
        #print np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][0])
        #print np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][1])
        #print np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][0])[0], np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][1])[0]
        #print i," dist",firstgenclust.sim_matrix[np.where(np.asarray(
        #    firstgenclust.name_dict) == firstgenavg['pot'][0])[0],
        #                            np.where(np.asarray(
        #                                firstgenclust.name_dict) == firstgenavg['pot'][1])[0]]

        # firstgenavg is a sorted list, take the first one (worst performance) and check if its neighbour is higher than dist
        if firstgenclust.sim_matrix[np.where(np.asarray(
            firstgenclust.name_dict) == firstgenavg['pot'][0])[0],np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][1])[0]]> dist :
            dbres[-1].append(firstgenavg['pot'][0])
            #print "Adding first graph"
        # If the first one is not further away from dist, check for the second one
        elif firstgenclust.sim_matrix[np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][1])[0], np.where(np.asarray(firstgenclust.name_dict) == firstgenavg['pot'][2])[0]]>dist :
            dbres[-1].append(firstgenavg['pot'][0])
            dbres[-1].append(firstgenavg['pot'][1])
            #print "Adding both graphs"
        #else: print "\t adding nothing"
        dbscanres = firstgenclust.dbscan(j, k)
        dbscanres[-1] = [firstgenclust.name_dict[x] for x in dbscanres[-1]]
        distfirst = firstgenclust.sim_matrix[np.where(np.asarray(
            firstgenclust.name_dict) == firstgenavg['pot'][0])[0],
                                    np.where(np.asarray(
                                        firstgenclust.name_dict) == firstgenavg['pot'][1])[0]]
        distlast = firstgenclust.sim_matrix[np.where(np.asarray(
            firstgenclust.name_dict) == firstgenavg['pot'][-2])[0],
                                    np.where(np.asarray(
                                        firstgenclust.name_dict) == firstgenavg['pot'][-1])[0]]
        #print i," first: ",distfirst, " last", distlast
        #print "len dbscanres: ", len(dbscanres)
        if (len(dbres[-1])>0 and (len(dbscanres[-1])>0 and dbres[-1][0] in dbscanres[-1])) or len(dbscanres)> 2:
             outlierdict.update({i: [dbres[-1], dbscanres[-1]]})

        #if (len(dbscanres[-1])>0 and (np.absolute(distfirst-distlast)>dist)):
           
    return outlierdict




                
