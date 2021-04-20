from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import queue
import networkx as nx
import matplotlib.pyplot as plt
import copy
import sys

class Graph:
    adj = defaultdict(lambda:set())
    vertices = set()
    w = defaultdict(lambda: 1)
    n = 0
    def __init__(self, n = 0):
        self.n = n

    def add_edge(self, u,v):
        if u not in self.vertices:
            self.vertices.add(u)
            self.n += 1
        if v not in self.vertices:
            self.vertices.add(v)
            self.n += 1
        self.adj[u].add(v)

    def set_weight(self, u, v, wt):
        self.w[frozenset({u,v})] = wt

    def weight(self, u, v):
        return self.w[frozenset({u,v})]

    def construct(self, E):
        for e in E:
            # self.adj[e[0]].add(e[1])
            self.add_edge(e[0],e[1])

    def edges(self):
        E = set()
        for u in self.vertices:
            for v in self.adj[u]:
                E.add((u,v))
        return E
    
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
        if prev[t] is None:
            return None
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
            for u in self.vertices:
                for v in self.adj[u]:
                    if dist[v] > dist[u] + self.weight(u,v):
                        dist[v] = dist[u] + self.weight(u,v)
                        prev[v] = u
                        H.put((dist[v],v))
        return prev

    def in_degree(self, v):
        deg = 0
        for u in self.vertices:
            if v in self.adj[u]:
                deg+=1
        return deg
    
    def min_cut(self, pairs):
        G = nx.DiGraph()
        for u in self.vertices:
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
        if len(lines[i]) > 1:
            l = lines[i].split(':')
            u = int(l[0])
            if len(l[1][1:]) != 0:
                v = map(int,l[1][1:].split())
                for y in v:
                    g.add_edge(u,y)
    pairs = []
    for i in range(next,len(lines)):
        if len(lines[i]) > 1:
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
        if common == set([s1,t1]):
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
        for u in g.vertices:
            for v in g.adj[u]:
                if u not in path_vertices and v not in path_vertices:
                    rog.add_edge(u, v, capacity=1)
        for p in unassigned_pairs:
            u,v = p[0],p[1]
            rog.add_edge('s',u,capacity = len(g.adj[u]-path_vertices))
            deg = 0
            for u in g.vertices:
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
    for u in g.vertices:
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
                obj = PrioritizedItem(Pi, g, pairs)
                if len(assigned_paths)+1 > max_so_far:
                    max_so_far = len(assigned_paths)+1
                    best_so_far = cp
                    best_so_far[p] = new_path
                if obj.priority > best_so_far:
                    S.put(obj)
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
        unassigned_pairs = set([p for p in pairs if assignment[p] is None])
        up = frozenset(unassigned_pairs)
        for p in up:
            # print(p)
            u,v = p[0],p[1]
            for path in assignment.values():
                if path is not None:
                    if u in path or v in path:
                        unassigned_pairs-={(u,v)}

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
        g.remove_nodes_from(assignment[p])
    return assignment

def write_paths(assignment, output_file):
    file = open(output_file, 'w')
    for p in assignment.values():
        if p is not None:
            for v in p:
                file.write('{} '.format(v))
            file.write('\n')
    file.close()

def improve(assignment, g, pairs):
    assigned_pairs = set([p for p in assignment.keys() if assignment[p] is not None])
    unassigned_pairs = set([p for p in pairs if p not in assigned_pairs])
    done = False
    for p in assigned_pairs:
        if done:
            break
        paths = [assignment[p]]
        next,prev = None, assignment[p]
        while next != prev:
            if done:
                break
            next = nextPath(g, p[0], p[1], paths)
            if next == prev:
                break
            prev = next
            paths.append(next)
            cont = False
            for a in assigned_pairs:
                if not vertex_disjoint(assignment[a],next):
                    cont = True
                    break
            if cont:
                continue
            for u in unassigned_pairs:
                test = g.shortest_path(u[0],u[1])
                if test is None:
                    continue
                if vertex_disjoint(next, test):
                    assignment[u] = test
                    assignment[p] = next 
                    done = True
                    break






# input_file, output_file = sys.argv[1], sys.argv[2]
input_file = 'tricky_examples_final/3.txt'
g, pairs = readGraph(input_file)
print(pairs)
E = g.edges()
G = nx.DiGraph()
G.add_edges_from(E)
assignment = initialize(G,pairs)
# p = pairs[0]
# assignment = defaultdict(lambda:None)
# assignment[p] = g.shortest_path(p[0],p[1])
# for t in pairs:
#     if t != p:
#         assignment[t] = None
# assignment[(91,92)] = g.shortest_path(91,92)
print('After initialization:')
print(assignment)
# write_paths(assignment, output_file)
# print(vertex_disjoint_paths(g,pairs))
print('After local search:')
for i in range(10):
    improve(assignment, g, pairs)
print(assignment)

output_file = input_file.split('.')[0]+'_out.txt'
write_paths(assignment,output_file)

# paths = []
# next = None 
# prev = 0
# while next != prev:
#     next = nextPath(g, 51, 52, paths)
#     print(next)
#     if next == prev:
#         break
#     prev = next
#     paths.append(next)
# print(paths)