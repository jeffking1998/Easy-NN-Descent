import numpy as np
from scipy.spatial import distance

np.random.seed(0)

def init_knn_graph(matrix, K):
    size_n = matrix.shape[0]

    knn_graph = []
    for idx in range(size_n):
        idx_choice  = list(np.random.choice(size_n, K, replace=False))
        while (idx in idx_choice): ## in my implement, KNN not include idx itself. 
            idx_choice  = list(np.random.choice(size_n, K, replace=False))
        knn_graph.append(idx_choice)    

    knn_graph_dist = [
        [] for _ in range(size_n)
    ]
    knn_graph_flag = [
        [ True for _ in range(K)] for _ in range(size_n)
        ## initally, all items are NEW neighbors
    ]

    for idx in range(size_n):
        idx_nbs = knn_graph[idx]
        idx_dist = [
            distance.euclidean(matrix[idx], matrix[i_nb]) for i_nb in idx_nbs
        ]
        iter = list(zip(idx_nbs, idx_dist))
        iter.sort(key=lambda x:x[1]) # sort by second item 
        idx_nbs = [x[0] for x in iter]
        idx_dist = [x[1] for x in iter]
        knn_graph[idx], knn_graph_dist[idx] = idx_nbs, idx_dist
    return np.array(knn_graph), np.array(knn_graph_dist), np.array(knn_graph_flag)



def get_reversed_neighbor(knn_graph):
    size_n = knn_graph.shape[0]
    reversed_neighbor = [
        [] for _ in range(size_n)
    ]
    for idx, nbs in enumerate(knn_graph):
        for nb in nbs:
            reversed_neighbor[nb].append(idx)
    return reversed_neighbor

def get_nb_rnb(knn_graph):
    reversed_neighbors = get_reversed_neighbor(knn_graph)
    concate_neighbor = []
    for nb, rnb in zip(knn_graph, reversed_neighbors):
        nb_rnb = list(set( list(nb) + list(rnb) ))
        concate_neighbor.append(nb_rnb)
    return concate_neighbor



def update_nerest_neighbor_with_flag(knn_graph, knn_graph_dist, knn_graph_flag , idx, candidate_idx, candidate_dist):
    idx_nbs  = list(knn_graph[idx])
    idx_dist = list(knn_graph_dist[idx])
    idx_flag = list(knn_graph_flag[idx])
    size_nb   = len(idx_nbs)
    if candidate_idx in idx_nbs:
        return 
    for i, dist in enumerate(idx_dist):
        if dist > candidate_dist:
            idx_dist.insert(i, candidate_dist)
            idx_nbs.insert(i, candidate_idx)
            idx_flag.insert(i, True)
            idx_dist.pop()
            idx_nbs.pop()
            idx_flag.pop()
            break 
    knn_graph[idx] = np.array(idx_nbs)
    knn_graph_dist[idx] = np.array(idx_dist)
    knn_graph_flag[idx] = np.array(idx_flag)

def cross_compare(knn_graph, knn_graph_dist, knn_graph_flag, reversed_neighbor, matrix, idx):
    """ 
    Suppose we have new & old neighbors,
    we need do cross matching in two ways:
    - inner of new items.
    - between new & old items. 
    """
    flags  = knn_graph_flag[idx]
    new_nb = knn_graph[idx][flags]
    old_nb = knn_graph[idx][~flags] 
    def cal_increment(nb_hood, reversed_neighbor):
        increment = []
        for nb in nb_hood:
            increment += list(reversed_neighbor[nb])
        return list(set(increment))
    new_nb = cal_increment(nb_hood=new_nb, reversed_neighbor=reversed_neighbor)
    old_nb = cal_increment(nb_hood=old_nb, reversed_neighbor=reversed_neighbor)
    ## inner of new 
    ## this is trival, can be improved further. 
    for onb in new_nb:       ## outer nb idx
        for inb in new_nb:   ## inner nb idx 
            if onb == inb:
                continue 
            dist = distance.euclidean(matrix[onb], matrix[inb])
            update_nerest_neighbor_with_flag(knn_graph, knn_graph_dist, knn_graph_flag, idx=onb, candidate_idx=inb, candidate_dist=dist)
            # no need this line above. 

    ## new and old 
    for onb in new_nb:
        for inb in old_nb:
            if onb == inb: ## though impossible, here for safety 
                continue
            dist = distance.euclidean(matrix[onb], matrix[inb])
            update_nerest_neighbor_with_flag(knn_graph, knn_graph_dist, knn_graph_flag, idx=onb, candidate_idx=inb, candidate_dist=dist)
            update_nerest_neighbor_with_flag(knn_graph, knn_graph_dist, knn_graph_flag, idx=inb, candidate_idx=onb, candidate_dist=dist)
    ture_idx = np.where(flags == True)[0]
    flags[ture_idx] = False
    knn_graph_flag[idx] = flags

def evaluate(GT, ANN):
    recall = 0
    size_n = GT.shape[0]
    K = GT.shape[1]
    for gt, ann in zip(GT, ANN):
        hit = 0
        for g in gt:
            if g in ann:
                hit += 1
        recall += hit / K 
    return recall / size_n

def GetGroundTrue(matrix, K):
    size_n = matrix.shape[0]
    knn = [
        # [] for _ in range(size_n)
    ]
    knn_dist = []
    all_item = list(range(size_n))
    for i in range(size_n):
        dist = [(j, distance.euclidean(matrix[i], matrix[j])) for j in range(size_n) if i!=j]
        dist.sort(key=lambda x:x[1])
        dist = dist[:K]
        iknn = [x[0] for x in dist]
        iknn_dist = [x[1] for x in dist]
        knn.append(iknn)
        knn_dist.append(iknn_dist)
    return np.array(knn), np.array(knn_dist)



def nn_descent(matrix, K):
    """
    - init 
    - sample   ( get concate nb is done.  ) sampl or define old/new nbs are to be done
    - cross match & update 
    """
    size_n = matrix.shape[0]
    knn_graph, knn_graph_dist, knn_graph_flag = init_knn_graph(matrix, K=K)
    order = np.arange(size_n)
    np.random.shuffle(order)
    max_iter = 10
    GT, GT_dist = GetGroundTrue(matrix, K=K)
    print('init, recall = {}'.format(evaluate(GT=GT, ANN=knn_graph)))
    for i_iter in range(max_iter):
        reversed_neighbor = get_reversed_neighbor(knn_graph)
        for idx in order:
            cross_compare(knn_graph, knn_graph_dist, knn_graph_flag, reversed_neighbor, matrix, idx)
        print('iter: {}, recall = {}'.format(i_iter+1, evaluate(GT=GT, ANN=knn_graph)))

    return knn_graph, knn_graph_dist

if __name__ == "__main__":
    matrix = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    nn_descent(matrix, K=2)
