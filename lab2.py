import random
import graphspace_utils
import json_utils
import matplotlib


def erdos_renyi(n,m,wantRank=False):
    """
    generates a graph (returned as a list of nodes and a list of edges) using the Erdos Renyi model.
    also returns a rank_ls that gives the order in which the edges were generated.
    """
    V = []
    E = []
    rank_ls = []
    ls = []
    E_set = set()
    for i in range(n):           #create nodes
        V.append(str(i))

    i = 0 
    while i < m:
        u = random.choice(V)
        v = random.choice(V)
        if u == v:
            pass
        elif frozenset([u,v]) in E_set:     #skips any already added edges or edges that go to self
            pass
        else:
            E_set.add(frozenset([u,v]))
            ls.append([frozenset([u,v]),i])
            i += 1

    for item in E_set:             #turns all the sets into lists and then appends them to E
        E.append(set_to_ls(item))

    for item in ls:                #lists each edge in the order that it was added
        rank_ls.append([set_to_ls(item[0]),item[1]])   #creates a rank_ls that has entries with format [edge,k]
                      
    if wantRank == True:
        return V, E, rank_ls
    else:
        return V, E


### abstraction barrier functions for entries in rank_ls
def get_edge(rank_tuple):
    return rank_tuple[0]

def get_k(rank_tuple):
    return rank_tuple[1]



def set_to_ls(s):
    """
    helper function for repackaging sets as lists
    """
    ls = []
    for item in s:
        ls.append(item)
    return ls



def barabasi_albert(t,n0,m0,wantRank=False):
    """
    generates a graph (reeturned as a list of nodes and edges) using the barabasi_albert model.
    t refers to the number of timesteps, n0 is the number of initial nodes, m0 is the number of edges added per timestep.
    """
    V = []
    E_set = set()
    p_deg = []      #keeps a list that gets updated such that: the number of copies of each node in p_deg is equal to its degree
    E = []
    ls = []
    rank_ls = []
    
    #initializes the starting nodes such that they form a big cycle
    for i in range(n0):
        V.append(str(i))

    for j in range(len(V)):
        if j < len(V)-1:
            E_set.add(frozenset([V[j],V[j+1]]))
        else:
            E_set.add(frozenset([V[j],V[0]]))

    #initializes p_deg
    for edge in E_set:
        for node in edge:
            p_deg.append(node)



    #this loop generates a new node with each timestep and attaches it to two nodes selected with probability proportional to degree
    i = 0
    k = 2
    i += n0
    t += n0       #still goes through t timesteps, but this makes naming the new nodes easier
    while i <= t:
        new_node = str(i)
        pick_set = BA_find(p_deg,m0)
        for item in pick_set:
            E_set.add(frozenset([new_node, item]))
            ls.append([frozenset([new_node,item]), k])
            p_deg += [new_node, item]
            
        V.append(new_node)
        i += 1
        k += 1

    for item in E_set:
        E.append(set_to_ls(item))

    for item in ls:
        rank_ls.append([set_to_ls(item[0]),item[1]])
            
    if wantRank == True:
        return V, E, rank_ls
    else:
        return V, E

        

def BA_find(p_deg,m0):
    """
    helper function that picks m0 unique nodes from p_deg and returns them as a set
    since we choose randomly from p_deg, the probability we pick a node is proportional to its degree.
    """
    pick_set = set()
    while len(pick_set) < m0:
        pick_set.add(random.choice(p_deg))
    return pick_set


        
def getNodeAttributes(nodes):
    attrs= {}
    for n in nodes:
        attrs[n] = {}
        attrs[n]['id'] = n
        attrs[n]['content'] = n
        attrs[n]['height'] = 60
        attrs[n]['width'] = 60
        attrs[n]['shape'] = 'star'
        attrs[n]['background_color'] = 'magenta'
        attrs[n]['k'] = 1
    return attrs

def getEdgeAttributes(edges,rank_ls=[]):
    attrs = {}
    for e in edges:
        source = e[0]
        target = e[1]
        if source not in attrs:
            attrs[source] = {}
        attrs[source][target] = {}
        attrs[source][target]['width'] = 2
        attrs[source][target]['k'] = 1
        
    if rank_ls != []:
        for item in rank_ls:
            edge = get_edge(item)
            source = edge[0]
            target = edge[1]
            k = get_k(item)
            attrs[source][target]['k'] = k
    
    return attrs





def main():
    for i in range(5):
        print(i, random.random())

    testList = ['A','B','C','D','E']
    for i in range(5):
        print(i,random.choice(testList))

    testV, testE, rank_ls = erdos_renyi(25,100,True)
    print(len(testV))
    print(len(testE))
    ER_node_attrs = getNodeAttributes(testV)
    ER_edge_attrs = getEdgeAttributes(testE,rank_ls)
    
    data = json_utils.make_json_data(testV,testE,ER_node_attrs,ER_edge_attrs,'Lab 2: Graph 2.2','Desc.',['Tag'])
    json_utils.write_json(data,'lab2.json')
    graphspace_utils.postGraph('lab2graph2-2','lab2.json','franzni@reed.edu','bio331')



    BA_V, BA_E, BA_rank_ls = barabasi_albert(30,5,2,True)

    BA_node_attrs = getNodeAttributes(BA_V)
    BA_edge_attrs = getEdgeAttributes(BA_E, BA_rank_ls)
    
    
    ba_data = json_utils.make_json_data(BA_V,BA_E,BA_node_attrs,BA_edge_attrs,'Lab2: BA Graph','Desc.',['Tag'])
    json_utils.write_json(ba_data,'lab2_ba.json')
    graphspace_utils.postGraph('lab2_ba','lab2_ba.json','franzni@reed.edu','bio331')
    

    graphspace_utils.shareGraph('lab2graph2-2','franzi@reed.edu','bio331','Lab2','aritz@reed.edu')
    graphspace_utils.shareGraph('lab2_ba', 'franzni@reed.edu','bio331','Lab2','aritz@reed.edu')
    
if __name__ == '__main__':
    main()
