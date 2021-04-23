from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import queue
import networkx as nx
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

@dataclass(order=True)
class Item:
    priority: int
    pair: Any=field(compare=False)

    def __init__(self,p,deg_sum):
        self.priority = deg_sum
        self.pair = p       
                
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
            for i in range(0,len(shortest_paths[p])):
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


input_file, output_file = sys.argv[1], sys.argv[2]

g, pairs = readGraph(input_file)
print(pairs)
E = g.edges()
G = nx.DiGraph()
G.add_edges_from(E)

assignment = initialize(G,pairs)
print('After initialization:')
print(assignment)
print('After local search:')
for i in range(10):
    improve(assignment, g, pairs)
print(assignment)
write_paths(assignment,output_file)