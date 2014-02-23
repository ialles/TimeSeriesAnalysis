import numpy as np
import itertools
import pylab
import numpy.linalg as la
import scipy
import h5py
import PlantDataAnalysis.clust as clust
from collections import defaultdict
import copy

relevant_names = ["Pot", "TT", "Manip.Biomass", "Manip.LeafArea",
                  "Genotype", "Scenario", "Genetic"]

def uniq(alist):    # Fastest without order preserving
    set = {}
    map(set.__setitem__, alist, [])
    return set.keys()

def import_data(data_path, data_delimiter):
    return np.genfromtxt(data_path, delimiter=data_delimiter, dtype=None)

def import_data_rows(data_path, data_delimiter, rows_numb = 2, cols = None, data_type = None):
    with open (data_path) as f_in:
        if cols is None:
            return scipy.genfromtxt(itertools.islice(f_in, rows_numb),
                                 delimiter = data_delimiter, dtype = data_type, usemask = False, deletechars = '"')
        elif rows_numb is None:
           return scipy.genfromtxt(f_in,
                                 delimiter = data_delimiter, dtype = data_type, usecols = cols , deletechars = '"') 
        else:
            return np.genfromtxt(itertools.islice(f_in, rows_numb), delimiter = data_delimiter,
                                 usecols = cols, dtype = data_type , deletechars = '"')

def remove_NA_based_on_col(data, col_nr):
    nz = (np.asarray(data[:,col_nr:col_nr+1]) == 'NA').sum(1)
    data = data[nz == 0, :]
    return data

def save_matrix(file_path, file_name,  matrix):
    np.save(file_path+file_name+".npy", matrix)

def load_matrix(file_path, file_name):
    return np.load(file_path+file_name+".npy")

def clean_sequences(sequences):
	to_rem = []
	for k,v in (sequences.iteritems()):
		if len(v.biomass) < 2:
		    to_rem.append(k)
		    
	for k in to_rem:
		rem = sequences.pop(k)
		print "removed: ", rem.pot
	return sequences
            
        

def extract_column_names(data_array, colnames = None):
    name_dict = {}
    if colnames is None:
        colnames = relevant_names
    for i, item in enumerate(data_array[0]):
	if item in colnames:
		name_dict.update ({item : i})
    return name_dict

def crop_data(data_array, name_dict):
    cropped_data = [ data_array[:, int(name_dict[el])] for  el in name_dict.keys ()]
    return np.asarray(cropped_data).T

def extractDiffReps(gen, reps, data):
    resp =  [[] for _ in range(reps)]
    for i in range(reps):
        #extract the repitition
        temp = np.array([x for x in data if x[0] == gen and x[3] != 'NA'
                         and x[1].astype(np.int) == i+1]) 
        #split via the condition
        resp[i] = split(temp,temp[:,2]!='WD')
    return resp


def split(arr, cond):
  return [arr[cond], arr[~cond]]


def extract_sequences(data_array, transform_dict):
    sequence_dict = {}
    for i,line in enumerate(data_array):
	try:
		el = sequence_dict[line[transform_dict['Pot']]]
		bm = el.biomass
		el.biomass = np.vstack([bm, [line[transform_dict['TT']],
                                                  line[transform_dict
                                                       ['Manip.Biomass']]]])
	except:
		seq = SequenceData(line[transform_dict['Pot']])
		seq.biomass = [line[transform_dict['TT']],
                                                  line[transform_dict
                                                       ['Manip.Biomass']]]
		seq.scenario = line[transform_dict['Scenario']]
		if 'Genetic' in transform_dict:
                    seq.genetic = line[transform_dict['Genetic']]
                else:
                    seq.genetic = 'D'
		seq.genotype = line[transform_dict['Genotype']]
                sequence_dict.update ({line[transform_dict['Pot']]: seq})

    return sequence_dict

def save_data_to_hdf5(my_data, manip, seq_nr, matrix = None):
    f = h5py.File(str(manip)+"_"+str(seq_nr)+".hdf5", 'w')
    dset = f.create_dataset('data', data=my_data)
    dset.attrs['manip']= manip
    if matrix is not None:
        dset2 = f.create_dataset('matrix', data=matrix)
    f.close()

def save_bicluster_session_to_hdf5(jpdmatrix, genodata, combined_genmtr, rowclustlist, colclustlist, qXY, manip, seq_nr, info=""):
    f = h5py.File("Biclust_session_"+info+str(manip)+"_"+str(seq_nr)+".hdf5", 'w')
    dset = f.create_dataset('jpd', data=jpdmatrix)
    dset2 = f.create_dataset('genodata', data=genodata)
    dset3 = f.create_dataset('genmtr', data=combined_genmtr)
    dset4 = f.create_dataset('rowcluster', data=rowclustlist)
    dset5 = f.create_dataset('colcluster', data=colclustlist)
    dset6 = f.create_dataset('qxy', data=qXY)
    dset.attrs['manip']= manip
    f.close()

def load_bicluster_session_from_hdf5(manip, seq_nr, prefix = "", info=""):
    f = h5py.File(prefix+"Biclust_session_"+info+str(manip)+"_"+str(seq_nr)+".hdf5", 'r')
    jpd = np.asmatrix(f['jpd'], dtype=np.float64)
    genodata = np.asmatrix(f['genodata'])
    genmtr = np.asmatrix(f['genmtr'], dtype=np.integer)
    row_clust = np.asarray(f['rowcluster'])
    col_clust = np.asarray(f['colcluster'])
    qXY = np.asmatrix(f['qxy'], dtype=np.float64)
    f.close()
    return jpd, genodata, genmtr, row_clust, col_clust, qXY

def load_data_from_hdf5(manip, seq_nr, prefix = ""):
    f = h5py.File(prefix+str(manip)+"_"+str(seq_nr)+".hdf5", 'r')
    data = np.asarray(f['data'])
    res = []
    if 'matrix' in f:
        res.append(np.asarray(f['matrix']))
    f.close()
    return data, res
    
def restore_cluster_obj(data, res):
    col_names = extract_column_names(data)
    sequences = extract_sequences(data[1:], col_names)
    cluster_inst = clust.Cluster(sequences)
    try:
        cluster_inst.sim_matrix = res[0]
    except:
        print "Distance matrix could not be restored"
    return cluster_inst

def calc_k_distance(k, seq_nr, sim_matrix):
    sm_mtr = sim_matrix[:seq_nr, :seq_nr]
    k_dist =  []
    for i, m in enumerate(sm_mtr):
        li = copy.deepcopy(m)
        li.sort()
        avg = (li[:k+1])
        k_dist.append((avg[-1], str(i)))
    return k_dist

def create_gendict(sequences):
    gendict = defaultdict(list)
    for k,v in sequences.iteritems():
        gendict[v.genotype].append(v)
    return gendict

        

class SequenceData:
    'Represents a time series and its meta data'

    def __init__(self, pot, genotype = None, scenario = None, genetic = None,
                 biomass = None, leafarea = None, bio_gold = None):
        self.pot = pot
        self.genotype = genotype
        self.scenario = scenario
        self.genetic = genetic
        self.biomass = biomass
        self.leafarea = leafarea
        self.bio_gold = bio_gold
        self.bio_nas = 0
        self.bio_tp = 0
        self.bio_precision = 0.0
        self.bio_recall = 0.0
        self.marker = np.asarray([])

        if self.biomass is None:
            self.biomass = np.array([], dtype = float)
        if self.leafarea is None:
            self.leafarea = np.array([], dtype = float)
        if self.bio_gold is None:
            self.bio_gold = np.array([], dtype = float)

    def calc_precision(self, cl):
        self.bio_nas = (self.bio_gold[:,1]).tolist().count('NA')
        self.bio_tp = 0
        for x in cl[-1]:
            if self.bio_gold[x][1] == 'NA':
                self.bio_tp +=1
        if len(cl[-1]) > 0:
            self.bio_precision = float(self.bio_tp)/float(len(cl[-1]))
        elif len(cl[-1]) == 0 and self.bio_nas == 0:
            self.bio_precision = 1.0
        return self.bio_precision

    def calc_recall(self, cl):
        self.bio_nas = (self.bio_gold[:,1]).tolist().count('NA')
        if self.bio_tp == 0 and self.bio_nas == 0:
            self.bio_recall = 1.0
        else:
            self.bio_recall = float(self.bio_tp) / float(self.bio_nas)
        return self.bio_recall

    def calc_fn_tn(self, cl):
        for x in cl[:-1]:
            self.bio_fn += len([el for el in x if self.bio_gold[el][1] == 'NA'])
        
            

    def remove_duplicates(self, field):
        if field is self.bio_gold or field is self.biomass:
            to_del = np.zeros((len(field)), dtype = int)
            for i,el in enumerate(field[:-1]):
                if el[0] == field[i+1][0]:
                    if el[1] == 'NA':
                        to_del[i] += 1
                    else:
                       to_del[i+1] += 1
            if len(self.bio_gold)> 0:
                self.bio_gold = self.bio_gold[to_del == 0,:]
            if len(self.biomass)>0:
                self.biomass = self.biomass[to_del == 0,:]
        else:
            print "field is not of type SequenceData.bio_gold, treatment for other types is not implemented yet."
                    




        
        
