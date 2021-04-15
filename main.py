from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import queue
import networkx as nx
import matplotlib.pyplot as plt
import copy

class Graph:
    adj = defaultdict(lambda:set())
    w = defaultdict(lambda: float('inf'))
    n = 0
    def __init__(self, n = 0):
        self.n = n

    def add_edge(self, u,v):
        self.adj[u].add(v)

    def set_weight(self, u, v, wt):
        self.w[frozenset({u,v})] = wt

    def weight(self, u, v):
        return self.w[frozenset({u,v})]

    def construct(self, E):
        for e in E:
            self.adj[e[0]].add(e[1])
    
    def BFS(self, s):
        visited = defaultdict(lambda: False)
        prev = defaultdict(lambda: None)
        visited[s] = True
        q = queue.Queue()
        q.put(s)
        while not q.empty():
            u = q.get()
            for v in self.adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.put(v)
                    prev[v] = u
        return prev

    def shortest_path(self, s, t):
        prev = self.BFS(s)
        path = []
        curr = t
        while curr != s:
            path.append(curr)
            curr = prev[curr]
        path.append(s)
        path.reverse()
        return path


    def dijkstra(self, s):
        dist = defaultdict(lambda: float('inf'))
        prev = defaultdict(lambda: None)
        dist[s] = 0
        H = queue.PriorityQueue()
        H.put((0,s))
        while not H.empty():
            d,u = H.get()
            if d > dist[u]:
                continue
            for u in range(self.n):
                for v in self.adj[u]:
                    if dist[v] > dist[u] + self.weight(u,v):
                        dist[v] = dist[u] + self.weight(u,v)
                        prev[v] = u
                        H.put((dist[v],v))
        return prev

    def in_degree(self, v):
        deg = 0
        for u in range(self.n):
            if v in self.adj[u]:
                deg+=1
        return deg
    
    def min_cut(self, pairs):
        G = nx.DiGraph()
        for u in range(self.n):
            for v in self.adj[u]:
                G.add_edge(u, v, capacity=1)
        for p in pairs:
            u,v = p[0],p[1]
            G.add_edge('s',u,capacity = len(self.adj[u]))
            G.add_edge(v,'t',capacity = self.in_degree(v))
        cut_value, _ = nx.minimum_cut(G, 's', 't')
        nx.draw(G, with_labels = True, pos = nx.spring_layout(G))
        plt.savefig('sample.png')
        return cut_value



    
    
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

def nextPath(g, s, d, paths):
    for p in paths:
        n = len(p)
        for i in range(n-1):
            u,v = p[i],p[i+1]
            g.set_weight(u,v,g.weight(u,v)+1)
    prev =  g.dijkstra(s)
    curr = d
    new_path = [d]
    while curr != s:
        curr = prev[curr]
        new_path.append(curr)
    new_path.reverse()
    for p in paths:
        n = len(p)
        for i in range(n-1):
            u,v = p[i],p[i+1]
            g.set_weight(u,v,g.weight(u,v)-1)
    return new_path

def vertex_disjoint(p1,p2):
    ans = False
    s1,s2,t1,t2 = p1[0],p2[0],p1[-1],p2[-1]
    set1 = set(p1)
    set2 = set(p2)
    common = set1.intersection(set2)
    if s1 == s2 and t1 == t2:
        if common == set(s1,t1):
            ans = True
    else:
        if len(common) == 0:
            ans = True
    return ans

class Solution:
    priority = -float('inf')
    paths = defaultdict(lambda: None)

    def __init__(self, paths = defaultdict(lambda:None)):
        self.paths = paths

    def assign_new_path(self, g, p, tried_paths):
        new_path = nextPath(g, p[0], p[1], tried_paths)
        self.paths[p] = new_path
    
    def upper_bound(self, g, pairs):
        assigned_paths = list(self.paths.values())
        assigned_pairs = set(self.paths.keys())
        unassigned_pairs = set(pairs)-assigned_pairs
        part1 = len(assigned_pairs) #first part of lower bound
        rog = nx.DiGraph()
        path_edges = set()
        path_vertices = set()
        for p in assigned_paths:
            for i in range(len(p)-1):
                path_edges.add((p[i],p[i+1]))
                path_vertices.add(p[i])
                path_vertices.add(p[i+1])
        for u in range(g.n):
            for v in g.adj[u]:
                if u not in path_vertices and v not in path_vertices:
                    rog.add_edge(u, v, capacity=1)
        for p in unassigned_pairs:
            u,v = p[0],p[1]
            rog.add_edge('s',u,capacity = len(g.adj[u]-path_vertices))
            deg = 0
            for u in range(g.n):
                if v in g.adj[u] and (u,v) not in path_edges:
                    deg+=1
            rog.add_edge(v,'t',capacity = deg)
        cut_value, _ = nx.minimum_cut(rog, 's', 't') #second part of lower bound
        # nx.draw(rog, with_labels = True, pos = nx.spring_layout(rog))
        # plt.savefig('sample1.png')
        return part1 + cut_value

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    soln: Any=field(compare=False)

    def __init__(self,soln, g, pairs):
        self.priority = -soln.upper_bound(g, pairs)
        self.soln = soln

@dataclass(order=True)
class Item:
    priority: int
    pair: Any=field(compare=False)

    def __init__(self,p,deg_sum):
        self.priority = deg_sum
        self.pair = p

def vertex_disjoint_paths(g, pairs): #branch and bound algorithm for finding vertex disjoint paths
    paths = defaultdict(lambda: None)
    E = []
    for u in range(g.n):
        for v in g.adj[u]:
            E.append((u,v))
    G = nx.Graph()
    G.add_edges_from(E)
    #initialization of solution
    best_so_far = initialize(copy.deepcopy(G), pairs)
    max_so_far = 0
    for k in best_so_far.keys():
        if best_so_far[k] is not None:
            max_so_far+=1
    S = queue.PriorityQueue()
    it = PrioritizedItem(Solution(paths),g,pairs)
    S.put(it)
    while not S.empty():
        it = S.get()
        P = it.soln
        unassigned_pairs = set()
        for p in pairs:
            if P.paths[p] is None:
                unassigned_pairs.add(p)
        #we pick one of the pairs to assign a path to here, could be improved (for finding the solution quicker) using some heuristic
        #we pick the unassigned pair with a shortest path of minimum degree sum
        min_sum, min_pair = float('inf'), None
        shortest_paths = defaultdict(lambda: [])
        degree_sum = defaultdict(lambda: float('inf'))
        for p in unassigned_pairs:
            sp = nx.single_source_shortest_path(G, p[0])
            shortest_paths[p] = sp[p[1]] 
            degree_sum[p] = 0 
            for i in range(1,len(shortest_paths[p])-1):
                degree_sum[p]+= G.degree[shortest_paths[p][i]]
            if degree_sum[p] < min_sum:
                min_sum = degree_sum[p]
                min_pair = p
        p = min_pair

        found_paths = []
        
        while True: #reconsider the following: old_path != new_path (time vs correctness)
            new_path = nextPath(g,p[0],p[1],found_paths)
            if new_path in found_paths:
                break
            found_paths.append(new_path)
            is_vertex_disjoint = True
            assigned_paths = P.paths.values()
            for ap in assigned_paths:
                if not vertex_disjoint(new_path,ap):
                    is_vertex_disjoint = False
                    break
            if is_vertex_disjoint:
                cp = copy.deepcopy(P.paths)
                cp[p] = new_path
                Pi = Solution(cp)
                Pi.priority = Pi.upper_bound(g,pairs)
                if len(assigned_paths)+1 > max_so_far:
                    max_so_far = len(assigned_paths)+1
                    best_so_far = cp
                    best_so_far[p] = new_path
                if Pi.priority > best_so_far:
                    S.put(Pi)
                break
    return best_so_far
                

def shortest_path(g, s ,t):
    paths = nx.single_source_shortest_path(g, s)
    return paths[t]
                
def initialize(g, pairs):
    shortest_paths = defaultdict(lambda: [])
    assignment = defaultdict(lambda: None)
    degree_sum = defaultdict(lambda: float('inf')) #total degree for now, consider using only out degrees for performance
    while True:
        unassigned_pairs = [p for p in pairs if assignment[p] is None]
        q = queue.PriorityQueue()
        for p in unassigned_pairs:
            paths = nx.single_source_shortest_path(g, p[0])
            if p[1] not in paths.keys():
                continue
            else:
                shortest_paths[p] = paths[p[1]] 
            degree_sum[p] = 0 
            for i in range(1,len(shortest_paths[p])-1):
                degree_sum[p]+= g.degree[shortest_paths[p][i]]
                q.put(Item(p,degree_sum[p]))

        p = None
        while not q.empty():
            p = q.get().pair
            is_vd = True
            for assigned_path in assignment.values():
                if assigned_path is not None:
                    if not vertex_disjoint(shortest_paths[p],assigned_path):
                        is_vd = False
                        break
            if is_vd:
                break
        if q.empty(): 
            break
        assignment[p] = shortest_paths[p]
        # print(assignment)
        g.remove_nodes_from(assignment[p])
    return assignment



            

    


# def getAllPathsUtil(g, u, d, visited, path, paths):
  
#         # Mark the current node as visited and store in path
#         visited[u]= True
#         path.append(u)
  
#         # If current vertex is same as destination, then print
#         # current path[]
#         if u == d:
#             # print(path)
#             paths.append(list(path))
#         else:
#             # If current vertex is not destination
#             # Recur for all the vertices adjacent to this vertex
#             for i in g.adj[u]:
#                 if visited[i]== False:
#                     getAllPathsUtil(g, i, d, visited, path, paths)
                      
#         # Remove current vertex from path[] and mark it as unvisited
#         path.pop()
#         visited[u]= False
   
   
#     # Prints all paths from 's' to 'd'
# def getAllPaths(g, s, d):
  
#     # Mark all the vertices as not visited
#     visited =[False]*(g.n)
  
#     # Create an array to store paths
#     path = []
#     paths = []
#     # Call the recursive helper function to print all paths
#     getAllPathsUtil(g, s, d, visited, path, paths)
#     return paths

# n = 10
# p = 0.5
# R = erdos_renyi_graph(n, p)
# E = [(0,1),(0,4),(0,3),(1,2),(4,2),(3,2)]
E = [(0,5),(0,3),(1,3),(2,4),(3,5),(3,4),(4,6),(4,7),(5,6),(6,8),(6,9),(6,7),(7,10)]
pairs = [(0,8),(1,9),(2,10)]
g = Graph(11)
g.construct(E)
G = nx.Graph()
G.add_edges_from(E)
# g.set_weight(0,1,2)
# g.set_weight(0,3,2)
# g.set_weight(0,4,1)
# g.set_weight(1,2,2)
# g.set_weight(3,2,2)
# g.set_weight(4,2,1)
# paths = getAllPaths(g, 0, 1)
# print(paths)
# pos = nx.spring_layout(R)
# nx.draw(R, pos, with_labels = True)
# plt.savefig("Graph.png", format="PNG")
# prev = g.BFS(0)
# print(prev)
# prev = g.dijkstra(0)
# print(prev)
# new_path = nextPath(g, 0, 2, [[0,4,2]])
# print(new_path)
# min_cut_size = g.min_cut([(0,8),(1,9),(2,10)])
# print(min_cut_size)
# assigned = defaultdict(lambda: None)
# assigned[(0,8)] = [0,5,6,8]
# partial = Solution(assigned)
# print(partial.upper_bound(g,pairs))
# print(shortest_path(G,0,10))

print(initialize(G,pairs))

print(vertex_disjoint_paths(g,pairs))
