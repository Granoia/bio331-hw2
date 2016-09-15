import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


import math
import pylab
import lab2


def readData(filename):
    """
    reads in a file, returns a list of unique nodes and a list of unique edges (under the assumption that the graph is undirected)
    """
    edge_set = set()
    node_set = set()
    with open(filename, 'r') as f:
        for line in f:
            current = line.strip('\n')
            curr_ls = current.split('\t')
            new_edge = frozenset(curr_ls)
            new_node1 = curr_ls[0]
            new_node2 = curr_ls[1]

            edge_set.add(new_edge)
            node_set.add(new_node1)
            node_set.add(new_node2)

    node_ls = set_to_list(node_set)

    edge_ls = []
    for edge in edge_set:
        edge_ls.append(set_to_list(edge))

    return node_ls, edge_ls


def set_to_list(s):
    """
    helper function that converts a set into a list containing all the elements that the set had.
    """
    ls = []
    for item in s:
        ls.append(item)
    return ls


ER_nodes, ER_edges = lab2.erdos_renyi(1500,3000)
BA_nodes, BA_edges = lab2.barabasi_albert(1500,4,2)



def make_adj_ls(nodes, edges):
    """
    makes an adjacency list from a list of nodes and a list of edges. The adj list is represented as a dictionary whose keys are nodes and whose values are ordered lists.
    the first entry of the list contains a list of neighboring nodes. the second entry of the list contains a boolean that indicates whether the node has been visited with BFS.
    this adj_ls will only be accessed using the following two accession functions.
    """
    adj_ls = {}
    for n in nodes:
        adj_ls[n] = [[],False]

    for e in edges:
        adj_ls[e[0]][0].append(e[1])
        adj_ls[e[1]][0].append(e[0])     #lists the nodes in each edge as adjacent to each other

    return adj_ls


def get_neighbors(adj_ls, node):
    #abstraction barrier function for accessing the adjacency list
    return adj_ls[node][0]

def get_visited(adj_ls, node):
    #abstraction barrier function for accessing the adjacency list
    return adj_ls[node][1]

def visit(adj_ls, node):
    adj_ls[node][1] = True

def devisit(adj_ls, node):
    adj_ls[node][1] = False

    

def BFS(adj_ls, start_node):
    """
    Breadth First Search
    """
    found_nodes = []
    Q = queue()
    visit(adj_ls, start_node)
    found_nodes.append(start_node)
    while Q.is_empty() == False:
        w = Q.dequeue()
        for n in get_neighbors(adj_ls, w):
            if get_visited(adj_ls, n) == False:
                visit(adj_ls, n)
                found_nodes.append(n)
                Q.enqueue(n)

    for node in adj_ls:         #resets all the visited statuses in the adj_ls to False after the search is done so that it doesn't interfere with the next BFS
        devisit(adj_ls, node)
    
    return found_nodes



def find_largest_cc(adj_ls):
    """
    Finds the largest connected component in the graph by running BFS starting from each node and finding whichever search returns the biggest list of nodes.
    """
    max_cc = []
    for node in adj_ls:
        current_cc = BFS(adj_ls, node)
        if len(current_cc) > len(max_cc):
            max_cc = current_cc
    return max_cc


class queue():                  #queue only for the purpose of implementing BFS
    def __init__(self):
        self.head = None
        self.tail = None
    
    def enqueue(self, item):
        new_item = LL_node(item)
        if self.head == None:
            self.head = new_item
            self.tail = new_item
        else:
            self.tail.attach(new_item)
            self.tail = new_item
    
    def dequeue(self):
        if self.head == None:
            return None
        else:
            ret = self.head.val
            new_head = self.head.give_next()
            self.head.detach()
            self.head = new_head
            return ret
    
    def is_empty(self):
        if self.head == None:
            return True
        else:
            return False

class LL_node():                 #node only for the purpose of implementing the queue class
    def __init__(self, val):
        self.val = val
        self.next = None
        
    def detach(self):
        self.next = None
    
    def give_next(self):
        return self.next
    
    def attach(self, next_node):
        self.next = next_node



def plot_deg_dist(prefix):
    fig = plt.figure(figsize=(6.5,4))
    x = list(range(1,10))
    y = [math.exp(-a) for a in x]
    logx = [math.log(a) for a in x]
    logy = [math.log(b) for b in y]
    
    plt.subplot(1,2,1)
    plt.plot(x,y,'o-r')
    plt.plot([0,8],[0,.3],'c--')
    plt.axis([0,10,0,.4])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(prefix)
    
    plt.subplot(1,2,2)
    plt.plot(logx,logy,'s-b')
    plt.axis([-.1,2.5,-10,1])
    plt.xlabel('log x')
    plt.ylabel('log y')
    plt.title(prefix+' (log)')
    
    plt.tight_layout()
    
    plt.savefig(prefix+'.png')
    
    print('wrote to '+prefix+'.png')
    return


def main():
    plot_deg_dist('test')


if __name__ == '__main__':
    main()
