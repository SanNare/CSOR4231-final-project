from graph_tool.all import *
from collections import defaultdict


def readGraph(input_file):
    g = Graph(directed = False)
    v_prop = g.new_vertex_property('int')
    vertices = defaultdict(lambda: False)
    file = open(input_file, 'r')
    lines = file.readlines()
    next = 0
    for i in range(1,len(lines)):
        if lines[i] == 'PAIRS\n':
            next = i+1
            break
        l = lines[i].split(':')
        u = int(l[0])
        if not vertices[u]:
            x = g.add_vertex()
            v_prop[x] = u
            vertices[u] = x
        v = map(int,l[1][1:].split())
        for y in v:
            if not vertices[y]:
                z = g.add_vertex()
                v_prop[z] = y
                vertices[y] = z
            g.add_edge(vertices[u],vertices[y])
    pairs = []
    for i in range(next,len(lines)):
        p = tuple(map(int,lines[i].split()))
        pairs.append(p)
    return g,vertices,v_prop,pairs

class Visitor(BFSVisitor):

    def __init__(self, name, pred, dist):
        self.name = name
        self.pred = pred
        self.dist = dist

    def discover_vertex(self, u):
        pass
        # print("-->", self.name[u], "has been discovered!")

    def examine_vertex(self, u):
        pass
        # print(self.name[u], "has been examined...")

    def tree_edge(self, e):
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = self.dist[e.source()] + 1

def shortest_path(source, target, g):
    dist = g.new_vertex_property("int")
    pred = g.new_vertex_property("int64_t")
    bfs_search(g, source , Visitor(v_prop, pred, dist))
    path = []
    path.append(target)
    curr = pred[target]
    while g.vertex(curr) != source:
        path.append(curr)
        curr = pred[g.vertex(curr)]
    path.append(source)
    path.reverse()
    return path

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


g,vertices,v_prop,pairs = readGraph('input.txt')
# graph_draw(g, vertex_text=v_prop, output="output.png")
        
# dist = g.new_vertex_property("int")
# pred = g.new_vertex_property("int64_t")
# bfs_search(g, vertices[89], Visitor(v_prop, pred, dist))
paths = []
for pair in pairs:
    start,end = pair[0],pair[1]
    p = shortest_path(vertices[start],vertices[end],g)
    path = [v_prop[v] for v in p]
    paths.append(path)
n = len(paths)
for i in range(0,n-1):
    for j in range(i+1,n):
        if vertex_disjoint(paths[i],paths[j]):
            print('{} and {} are vertex disjoint'.format(paths[i],paths[j]))
        else:
            print('{} and {} are not vertex disjoint'.format(paths[i],paths[j]))