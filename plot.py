import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim import *
import clust
import random as rand
from collections import OrderedDict
import csv

import numpy.linalg as la

def createPlotAndDist(numb_gen, numb_rep, init_data):
    for x in range(numb_gen):
        extraction = extractDiffReps(uniqNames[x], numb_rep, init_data)
        for i in range(numb_rep):
            pylab.plot(extraction[i][0][:,3], label='WW')
            pylab.plot(extraction[i][1][:,3], label ='WD')
            pylab.xlabel('Time steps')
            pylab.ylabel('BioVol')
            pylab.title((uniqNames[x],"rep "+str(i+1)))
            dist = dtw.dtw_distance(extraction[i][0][:,3:5].astype(np.float),extraction[i][1][:,3:5].astype(np.float))
            pylab.suptitle("DTW distance: "+str(dist))
            pylab.savefig("D:\Master\Thesis\DEV\plots\\"+uniqNames[x]+"_"+str(i+1))
            pylab.close()
            

            
def compareAndPlot(sequ1, sequ2, name):
    pylab.plot(sequ1[:,0],sequ1[:,1], label='sequ1')
    pylab.plot(sequ2[:,0],sequ2[:,1], label ='sequ2')
    pylab.xlabel('Time steps')
    pylab.title(name)
    #dist = dtw_distance(sequ1.astype(np.float),sequ2.astype(np.float))
    matrix = dtw_distance(sequ1.astype(np.float),sequ2.astype(np.float))
    dist = matrix[len(matrix)-1,len(matrix[0])-1]
    path = optWarpingPath (matrix)
    path = path[::1]
    pylab.plot (sequ1[:,0],sequ1[:,1], marker ='o', color="blue")
    pylab.plot (sequ2[:,0],sequ2[:,1], marker ='o', color="red")
    #alignment ??
    print "path ",path
    #path.pop()
    for tuple in path:
        print "tuple: ",tuple
        print "alignment print:",([sequ1[tuple[0]][0],sequ2[tuple[1]][0]],[sequ1[tuple[0]][1],sequ2[tuple[1]][1]])
        pylab.plot([sequ1[tuple[0]][0],sequ2[tuple[1]][0]],[sequ1[tuple[0]][1],sequ2[tuple[1]][1]],color="gray")
    pylab.suptitle("DTW distance: "+str(dist))
    pylab.savefig("D:\Master\Thesis\DEV\plots\\"+name)
    pylab.close()

def determine_plot_grid(size, cols):
    mod = size % cols
    rows = int(size / cols) + mod
    return rows, cols

def plot_cluster(clusters, method_name, cols = 2, cluster_nr = 3):
    if isinstance(clusters, clust.Cluster):
        print "\n steps to take: ",(len(clusters.sequences) - cluster_nr)
        size = (len(clusters.sequences) - cluster_nr)
        row, col = determine_plot_grid(cluster_nr, cols)
        fig = pylab.figure(1)
        pylab.title('BioVol Clusters')
        plot_nr = 0
        for nr, cl in (clusters._cluster_steps[0][size].items()):
            plot_nr += 1
            #print "row: ",row, "col: ", col, "nr: ",plot_nr
            ax = pylab.subplot(row,col,plot_nr)
            pylab.xlabel('Time Steps')
            pylab.ylabel('BioVol')
            for el in cl:
                bio = np.asarray(clusters.sequences[clusters.name_dict[el]].biomass )
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1] ,label = clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_ylim([0,700])
            ax.set_xlim([0,120])
            #ax.legend(handles,labels, loc=2)
        fig.set_size_inches(18.5,10.5)
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusters.sequences))+" clusters: "+str(cluster_nr))
        pylab.savefig( "D:\Master\Thesis\DEV\plots\Cluster_"+method_name+"_seq"+str(len(clusters.sequences))+"_clust"+str(cluster_nr), dpi =100)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)

def plot_big_cluster(clusters, method_name, cluster_nr = 3):
    if isinstance(clusters, clust.Cluster):
        size = (len(clusters.sequences) - cluster_nr)
        NUM_COLORS = cluster_nr
	linestyles = ["-","--","-.",":","-"]
        cm = pylab.get_cmap('Dark2')
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
        
        for nr, (_, cl) in enumerate(clusters._cluster_steps[0][size].items()):
	    line = rand.randint(0,len(linestyles)-1)
            for el in cl:
                bio = np.asarray(clusters.sequences[clusters.name_dict[el]].biomass)
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1], linestyles[line], color = colors[nr],label = clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario)
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles,labels, loc=2)
        fig.set_size_inches(20.5,12.5)
        pylab.xlabel('Time Steps')
        pylab.ylabel('BioVol')
        pylab.title('BioVol Clusters')
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusters.sequences))+" clusters: "+str(cluster_nr))
        pylab.savefig( "D:\Master\Thesis\DEV\plots\Cluster_big_"+method_name+"_seq"+str(len(clusters.sequences))+"_clust"+str(cluster_nr), dpi =200)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)

def plot_big_cluster_write_legend(clusters, method_name, cluster_nr = 3):
    if isinstance(clusters, clust.Cluster):
        output_file = "D:\Master\Thesis\DEV\plots\Cluster_big_"+method_name+"_seq"+str(len(clusters.sequences))+"_clust"+str(cluster_nr)
        legend_path = output_file+".csv"
        legend_file = open(legend_path, "wb")
        size = (len(clusters.sequences) - cluster_nr)
        csvwriter = csv.writer(legend_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        NUM_COLORS = cluster_nr
	linestyles = ["-","--","-.",":","-"]
        cm = pylab.get_cmap('Dark2')
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
        
        for nr, (_, cl) in enumerate(clusters._cluster_steps[0][size].items()):
	    line = rand.randint(0,len(linestyles)-1)
	    leg = [nr]
            for el in cl:
                bio = np.asarray(clusters.sequences[clusters.name_dict[el]].biomass)
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1], linestyles[line], color = colors[nr],
                           label = str(nr))
                leg.append(clusters.sequences[clusters.name_dict[el]].pot+", "+clusters.sequences[clusters.name_dict[el]].genotype+", "+clusters.sequences[clusters.name_dict[el]].scenario)
            csvwriter.writerow(leg)
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
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusters.sequences))+" clusters: "+str(cluster_nr))
        pylab.savefig( output_file, dpi =100)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)

def plot_inertia(clusters, method_name):
    if isinstance(clusters, clust.Cluster):
        pylab.plot(np.asarray(clusters._distance_data)[:,0], np.asarray(clusters._distance_data)[:,1], marker='o')
        pylab.xlabel('Number of Clusters')
        pylab.ylabel('Intra Cluster Inertia')
        pylab.title('Intra Cluster Inertia')
        pylab.savefig( "D:\Master\Thesis\DEV\plots\Cluster_inertia_"+method_name+"_seq"+str(len(clusters.sequences)), dpi =100)
        pylab.close()
        
def compareAndPlot_lax(sequ1, sequ2, name):
    pylab.plot(sequ1[:,0],sequ1[:,1], label='sequ1')
    pylab.plot(sequ2[:,0],sequ2[:,1], label ='sequ2')
    pylab.xlabel('Time steps')
    pylab.title(name)
    #dist = dtw_distance(sequ1.astype(np.float),sequ2.astype(np.float))
    matrix = dtw_step_size(sequ1, sequ2)
    dist = matrix[len(matrix)-1,len(matrix[0])-1]
    path, al = optWarpingPath_lax (matrix)
    path = path[::1]
    pylab.plot (sequ1[:,0],sequ1[:,1], marker ='o')
    pylab.plot (sequ2[:,0],sequ2[:,1], marker ='o')
    #alignment ??
    print "path ",path
    #path.pop()
    for tuple in path:
        print "tuple: ",tuple
        print "alignment print:",([sequ1[tuple[0]][0],sequ2[tuple[1]][0]],[sequ1[tuple[0]][1],sequ2[tuple[1]][1]])
        pylab.plot([sequ1[tuple[0]][0],sequ2[tuple[1]][0]],[sequ1[tuple[0]][1],sequ2[tuple[1]][1]])
    pylab.suptitle("DTW distance: "+str(dist)+" new: "+str(dist/al))
    pylab.savefig("D:\Master\Thesis\DEV\plots\\"+name)
    pylab.close()


def plot_scan(clusters, res , method_name, cols = 2):
    if isinstance(res, list):
        size = (len(res))
        row, col = determine_plot_grid(size, cols)
        fig = pylab.figure(1)
        pylab.title('BioVol Clusters')
        plot_nr = 0
        for nr, cl in enumerate(res):
            plot_nr += 1
            #print "row: ",row, "col: ", col, "nr: ",plot_nr
            ax = pylab.subplot(row,col,plot_nr)
            pylab.xlabel('Time Steps')
            pylab.ylabel('BioVol')
            for el in cl:
                bio = np.asarray(clusters.sequences[clusters.name_dict[el]].biomass )
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1] ,label = clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_ylim([0,700])
            ax.set_xlim([0,120])
            #ax.legend(handles,labels, loc=2)
        fig.set_size_inches(18.5,10.5)
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusters.sequences))+" clusters: "+str(size))
        pylab.savefig( "D:\Master\Thesis\DEV\plots\Cluster_"+method_name+"_seq"+str(len(clusters.sequences))+"_clust"+str(size), dpi =100)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)


def plot_scan_w_outliers(clusters, res , method_name, outlierlist, cols = 2):
    if isinstance(res, list):
        size = (len(res))
        row, col = determine_plot_grid(size, cols)
        fig = pylab.figure(1)
        pylab.title('BioVol Clusters')
        plot_nr = 0
        for nr, cl in enumerate(res):
            plot_nr += 1
            #print "row: ",row, "col: ", col, "nr: ",plot_nr
            ax = pylab.subplot(row,col,plot_nr)
            pylab.xlabel('Time Steps')
            pylab.ylabel('BioVol')
            for el in cl:
                bio = np.asarray(clusters.sequences[clusters.name_dict[el]].biomass )
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                if clusters.sequences[clusters.name_dict[el]].pot in outlierlist:
                    pylab.plot(bio[:,0], bio[:,1] ,label = clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario, color= "red")
                else:
                    pylab.plot(bio[:,0], bio[:,1] ,label = clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_ylim([0,700])
            ax.set_xlim([0,120])
            #ax.legend(handles,labels, loc=2)
        fig.set_size_inches(18.5,10.5)
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusters.sequences))+" clusters: "+str(size))
        pylab.savefig( "D:\Master\Thesis\DEV\plots\Cluster_"+method_name+"_seq"+str(len(clusters.sequences))+"_clust"+str(size), dpi =100)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)
    

def plot_sequence_cluster(sequences, res , name_dict , method_name,  cols = 2):
    if isinstance(res, list):
        size = (len(res))
        row, col = determine_plot_grid(size, cols)
        fig = pylab.figure(1)
        pylab.title('BioVol Clusters')
        plot_nr = 0
        for nr, cl in enumerate(res):
            plot_nr += 1
            #print "row: ",row, "col: ", col, "nr: ",plot_nr
            ax = pylab.subplot(row,col,plot_nr)
            pylab.xlabel('Time Steps')
            pylab.ylabel('BioVol')
            for el in cl:
                bio = np.asarray(sequences[name_dict[el]].biomass )
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1] ,label = sequences[name_dict[el]].pot+" "+sequences[name_dict[el]].genotype+" "+sequences[name_dict[el]].scenario)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_ylim([0,700])
            ax.set_xlim([0,120])
            #ax.legend(handles,labels, loc=2)
        fig.set_size_inches(18.5,10.5)
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(sequences))+" clusters: "+str(size))
        pylab.savefig( "D:\Master\Thesis\DEV\plots\Cluster_"+method_name+"_seq"+str(len(sequences))+"_clust"+str(size), dpi =100)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)

def plot_big_scan_write_legend(clusters, res, method_name, folder = ""):
    if isinstance(clusters, clust.Cluster):
        cluster_nr = len(res)
        if len(res[-1]) == 0:
            cluster_nr -= 1
        output_file = "D:\Master\Thesis\DEV\plots\\"+folder+"Cluster_dbscan_big_"+method_name+"_seq"+str(len(clusters.sequences))#+"_clust"+str(cluster_nr)
        legend_path = output_file+".csv"
        legend_file = open(legend_path, "wb")
        size = (len(clusters.sequences) - cluster_nr)
        csvwriter = csv.writer(legend_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        NUM_COLORS = cluster_nr
	linestyles = ["-","--","-.",":","-"]
        cm = pylab.get_cmap('Dark2')
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
        
        for nr, cl in enumerate(res):
	    line = rand.randint(0,len(linestyles)-1)
	    leg = [nr]
            for el in cl:
                bio = np.asarray(clusters.sequences[clusters.name_dict[el]].biomass)
                if np.shape(bio) == (2,):
                    bio = np.reshape(bio, (1,2))
                pylab.plot(bio[:,0], bio[:,1], linestyles[line], color = colors[nr],
                           label = str(nr))
                leg.append(clusters.sequences[clusters.name_dict[el]].pot+", "+clusters.sequences[clusters.name_dict[el]].genotype+", "+clusters.sequences[clusters.name_dict[el]].scenario)
            csvwriter.writerow(leg)
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
        pylab.suptitle("Method: "+method_name+" Sequences: "+str(len(clusters.sequences))+" clusters: "+str(cluster_nr))
        pylab.savefig( output_file, dpi =100)
        pylab.close()
    else: print "cluster is not of type clust.Cluster its: ", type(clusters)
    



def plot_bar_chart(res):

    N = len(res)
    menMeans = (20, 35, 30, 35, 27)
    menStd =   (2, 3, 4, 1, 2)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    NUM_COLORS = N
    cm = pylab.get_cmap('Dark2')
    colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = []
    y = []
    margin = 0
    for j,cl in enumerate(res):
        for i, val in enumerate(cl):
            if val != 0:
                x.append( val)
                y.append(i+margin)
        print "x:",x
        print "y:",y
        print j, colors
        ax.bar(y, x, width, color = colors[j])
        y = []
        x = []
        margin += 0.35
    #rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)
    

    womenMeans = (25, 32, 34, 20, 25)
    womenStd =   (3, 5, 2, 3, 3)
    #rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

    # add some
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    #ax.set_xticks(ind+width)
    #ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

    #ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )


    plt.show()



def plot_bi_columns(col_clust_distrib, method_name='biclustering'):
    NUM_COLORS = len(col_clust_distrib)
    linestyles = ["-","--","-.",":","-"]
    cm = plt.get_cmap('Dark2')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = [cm(i) for i in np.linspace(0,1, NUM_COLORS)]
    

    for i,x in enumerate(col_clust_distrib):
        xs = np.arange(len(x))
        line = rand.randint(0,len(linestyles)-1)
        ax.bar(xs, x, zs=i*10, zdir='y', color=colors[i], alpha=0.7)
        #pylab.plot(x, linestyles[line], color = colors[i],
                       #label = str(i))
        
    handles, labels = ax.get_legend_handles_labels()
    #print clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario
    #ax.legend(handles,labels, loc=2)
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

    # Put a legend to the right of the current axis
    by_label = OrderedDict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
    fig.set_size_inches(20.5,12.5)
    plt.xlabel('Values')
    plt.ylabel('Clusters')
    ax.set_zlabel('#Occurences')
    plt.title('Column Clusters Distribution')
    plt.suptitle("Method: "+method_name+" clusters: "+str(len(col_clust_distrib)))
    #pylab.savefig( output_file, dpi =100)
    plt.show()

def plot_sequences_bg_w_outliers(bg_sequences, clusterobj, distres, dbres, method_name, folder = "", title = "BioVol Clusters", ylabel = 'BioVol', nolegend = True, color = "blue", bgc=None):
    '''Plots the sequences of the given clusterobj into one plot,ads the bg_sequences as background'''
    if isinstance(clusterobj, clust.Cluster):
        #print type(bg_sequences)
        output_file = "D:\Master\Thesis\DEV\plots\\"+folder+"Cluster_outlier_plant_"+method_name+"_seq"+str(len(clusterobj.sequences))#+"_clust"+str(cluster_nr)
    	#linestyles = ["-","--"]
        line = "-"
        fig = pylab.figure()
        if bgc is None:
            ax = fig.add_subplot(111) 
        else:
            ax = fig.add_subplot(111, axisbg = bgc)
            
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
                
            if cl.pot in distres[-1] and np.where(np.array(clusterobj.name_dict) == cl.pot)[0] in dbres[-1]:
                color = 'red'
            elif cl.pot in distres[-1]:
                color = '#a901db' #lila
            elif np.where(np.array(clusterobj.name_dict) == cl.pot)[0] in dbres[-1]:
                color = '#ff8000' # orange
                
            if color is None:
                pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr))
            else:
                pylab.plot(bio[:,0], bio[:,1], line,
                       label = str(nr), color = color)
            color = "blue"
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



def plot_genotype(series_list, method_name, title = "sequences", folder = "", counter = "", ylim = 600):
      
        output_file = "D:\Master\Thesis\DEV\plots\\"+folder+method_name+counter#))#+"_clust"+str(cluster_nr)

        fig = pylab.figure()
        ax = fig.add_subplot(111)
        
        for nr,cl in enumerate(series_list):
            print "plot biomass of"+str(nr)
            bio = np.asarray(cl.biomass)
            if np.shape(bio) == (2,):
                bio = np.reshape(bio, (1,2))
            pylab.plot(bio[:,0], bio[:,1], "-",
                           label = str(nr))

        handles, labels = ax.get_legend_handles_labels()
        #print clusters.sequences[clusters.name_dict[el]].pot+" "+clusters.sequences[clusters.name_dict[el]].genotype+" "+clusters.sequences[clusters.name_dict[el]].scenario
        #ax.legend(handles,labels, loc=2)
        # Shink current axis by 20%
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # Put a legend to the right of the current axis
        #by_label = OrderedDict(zip(labels, handles))
        #ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
        #fig.set_size_inches(20.5,12.5)
        ax.set_ylim([0,ylim])
        pylab.xlabel('Time Steps')
        pylab.ylabel('BioVol')
        pylab.title(title)
        pylab.savefig( output_file)
        #dpi = 80
        pylab.close()
  

