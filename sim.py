import numpy as np
import pylab
import numpy.linalg as la

def dtw_distance(list1, list2, costf=lambda x,y: la.norm(x - y) ):

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix(n+1,m+1)

    for (i,x) in enumerate(list1):
        i += 1
        for (j,y) in enumerate(list2):
            j += 1

            cost = costf(x,y)
            #print "cost: "+str(cost)+str(x)+str(y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])
    #for k in dtw:
     #   print str(k)+"\n"
    #print dtw[n,m]
    return dtw

def dtw_weighted_distance(list1, list2, costf=lambda x,y: la.norm(x - y), start_w = 0.6, end_w = 1.2):

    n = len(list1)
    m = len(list2)
    lstart = np.log(start_w)
    lend = np.log(end_w)
    wlist1 =[np.exp(x) for x in np.linspace(lstart,lend, len(list1)+1)]
    wlist2 = [np.exp(x) for x in np.linspace(lstart,lend, len(list2)+1)]
    dtw = initialize_dmatrix(n+1,m+1)

    for (i,x) in enumerate(list1):
        i += 1
        for (j,y) in enumerate(list2):
            j += 1

            cost = costf(x,y)
            if i == j: cost = cost*wlist1[i]
            if i>j: cost = cost*wlist1[i]
            if i>j: cost = cost*wlist2[j]
            #print "cost: "+str(cost)+str(x)+str(y)
            dtw[i,j] = cost + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1][j-1])
    #for k in dtw:
     #   print str(k)+"\n"
    #print dtw[n,m]
    return dtw

def initialize_dmatrix(rows,cols):
    d = np.zeros((rows,cols),dtype='float')

    d[:,0] = 1e6
    d[0,:] = 1e6
    d[0,0] = 0

    return d

def initialize_dmatrix_lax(rows,cols):
    d = np.zeros((rows,cols),dtype='float')

    d[:,0] = 1e6
    d[0,:] = 1e6
    d[:,1] = 1e6
    d[1,:] = 1e6
    d[0,0] = 0
    d[1,1] = 0

    return d


def warpingHelperSimpl(dmatrix, nm, path):
    print "warpingHelperSimpl called ...\n"
    if nm ==[0,0]:
        print "entered if 0,0 ..."
        print nm," value = ",dmatrix[nm[0], nm[1]]
        #path.append(nm)
        return path
    else:
        print "entered else ..."
        nmlist = [[nm[0]-1,nm[1]-1], [nm[0]-1,nm[1]],[nm[0],nm[1]-1]]
        print "nmlist: ", nmlist
        minval = min(dmatrix[nmlist[0][0],nmlist[0][1]], dmatrix[nmlist[1][0],nmlist[1][1]],dmatrix[nmlist[2][0],nmlist[2][1]])
        ind = [i for i, v in enumerate(nmlist) if dmatrix[v[0],v[1]] == minval]
        #print "min index: ",nmlist[ind[0]]
        print "min list: ",[dmatrix[nmlist[0][0],nmlist[0][1]], dmatrix[nmlist[1][0],nmlist[1][1]],dmatrix[nmlist[2][0],nmlist[2][1]]], "\n chosen ",ind        
        print nmlist[ind[0]]," value = ",dmatrix[nmlist[ind[0]][0], nmlist[ind[0]][1]]        
        path.append(nmlist[ind[0]])
        return warpingHelperSimpl(dmatrix, nmlist[ind[0]], path)
    

def optWarpingPath(dmatrix):
    path = []
    nm = [len(dmatrix)-1, len(dmatrix[0])-1]
    print "init nm: ",nm
    print "warping output", warpingHelperSimpl(dmatrix, nm, path)
    print "path ", path
    path = [[i-1, j-1] for i,j in path ]
    return path


def optWarpingPath_lax(dmatrix):
    path = []
    nm = [len(dmatrix)-2, len(dmatrix[0])-2]
    print "init nm: ",nm
    aligns = 0
    path,aligns= warping_helper_lax(dmatrix, nm, path, aligns)
    print "path ", path
    print "aligns: ", aligns
    path = [[i-1, j-1] for i,j in path ]
    return path, aligns

def warping_helper_lax(dmatrix, nm, path, aligns):
    print "warping_helper_lax called ...",aligns,"<\n"
    aligns = aligns+1
    print "aligns: ",aligns
    if nm ==[0,0]:
        print "entered if 0,0 ..."
        print nm," value = ",dmatrix[nm[0], nm[1]]
        #path.append(nm)
        return path,aligns
    else:
        print "entered else ..."
        if np.shape(dmatrix) < ((3,3)):
            nmlist = [[nm[0]-1,nm[1]-1], [nm[0]-1,nm[1]],[nm[0],nm[1]-1]]
        else:
            nmlist =  nmlist = [[nm[0]-1,nm[1]-1], [nm[0]-2,nm[1]-1],[nm[0]-1,nm[1]-2],[nm[0]-2,nm[1]],[nm[0],nm[1]-2]]
        print "nmlist: ", nmlist
        minlist = [dmatrix[x[0],x[1]] for x in nmlist]
        minval = min(minlist)
        #minval = min(dmatrix[nmlist[0][0],nmlist[0][1]], dmatrix[nmlist[1][0],nmlist[1][1]],dmatrix[nmlist[2][0],nmlist[2][1]])
        ind = [i for i, v in enumerate(nmlist) if dmatrix[v[0],v[1]] == minval]
        #print "min index: ",nmlist[ind[0]]
        print "min list: ",minlist, "\n chosen ",ind        
        print nmlist[ind[0]]," value = ",dmatrix[nmlist[ind[0]][0], nmlist[ind[0]][1]]        
        path.append(nmlist[ind[0]])
        return warping_helper_lax(dmatrix, nmlist[ind[0]], path, aligns)

def dtw_step_size(list1, list2, costf=lambda x,y: la.norm(x - y) ):

    n = len(list1)
    m = len(list2)
    dtw = initialize_dmatrix_lax(n+2,m+2)
    if np.shape(dtw) >= ((3,3)):
        for (i,x) in enumerate(list1):
            i += 2
            for (j,y) in enumerate(list2):
                j += 2
                cost = costf(x,y)
                #print "cost: "+str(cost)+str(x)+str(y)
                dtw[i,j] = cost + min(dtw[i-1,j-1],dtw[i-2,j-1],dtw[i-1][j-2], dtw[i-2][j], dtw[i][j-2])
    else:
        print "Matrix is to small switching back to classic dtw!"
        dtw = dtw_distance(list1, list2)
    #for k in dtw:
     #   print str(k)+"\n"
    #print dtw[n,m]
    return dtw



