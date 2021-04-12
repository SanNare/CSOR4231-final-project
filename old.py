from collections import defaultdict
from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    adj = defaultdict(lambda:[])
    def __init__(self, n = 0):
        self.n = n

    def add_edge(self, u,v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def construct(self, E):
        for e in E:
            self.adj[e[0]].append(e[1])
            self.adj[e[1]].append(e[0])
    
def readGraph(input_file):
    g = Graph()
    file = open(input_file, 'r')
    lines = file.readlines()
    next = 0
    for i in range(1,len(lines)):
        if lines[i] == 'PAIRS\n':
            next = i+1
            break
        l = lines[i].split(':')
        u = int(l[0])
        v = map(int,l[1][1:].split())
        for y in v:
            g.add_edge(u,y)
    pairs = []
    for i in range(next,len(lines)):
        p = tuple(map(int,lines[i].split()))
        pairs.append(p)
    return g,pairs


def getAllPathsUtil(g, u, d, visited, path, paths):
  
        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)
  
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            # print(path)
            paths.append(list(path))
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in g.adj[u]:
                if visited[i]== False:
                    getAllPathsUtil(g, i, d, visited, path, paths)
                      
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False
   
   
    # Prints all paths from 's' to 'd'
def getAllPaths(g, s, d):
  
    # Mark all the vertices as not visited
    visited =[False]*(g.n)
  
    # Create an array to store paths
    path = []
    paths = []
    # Call the recursive helper function to print all paths
    getAllPathsUtil(g, s, d, visited, path, paths)
    return paths

n = 100
p = 1
R = erdos_renyi_graph(n, p)
g = Graph(n)
g.construct(R.edges)
paths = getAllPaths(g, 0, 1)
# print(paths)
pos = nx.spring_layout(R)
# nx.draw(R, pos, with_labels = True)
# plt.savefig("Graph.png", format="PNG")