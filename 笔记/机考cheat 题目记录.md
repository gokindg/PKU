#简单题

```python
#括号
def check_brackets(s):
    stack = []
    nested = False
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs.keys():
            if not stack or stack.pop() != pairs[ch]:
                return "ERROR"
            if stack:
                nested = True
    if stack:
        return "ERROR"
    return "YES" if nested else "NO"

s = input()
print(check_brackets(s))
```



```python
#完成数算作业（并查集
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        parent[xroot] = yroot

n, m = map(int, input().split())
parent = list(range(n + 1))
for _ in range(m):
    i, j = map(int, input().split())
    union(parent, i, j)

count = sum(i == parent[i] for i in range(1, n + 1))
print(count)
```



```python
#排队 并查集
def getRoot(a):
    if parent[a] != a:
        parent[a] = getRoot(parent[a])
    return parent[a]


def merge(a, b):
    pa = getRoot(a)
    pb = getRoot(b)
    if pa != pb:
        parent[pa] = parent[pb]


t = int(input())
for i in range(t):
    n, m = map(int, input().split())
    parent = [i for i in range(n + 10)]
    for i in range(m):
        x, y = map(int, input().split())
        merge(x, y)
    for i in range(1, n + 1):
        print(getRoot(i), end=" ")
    # 注意，一定不能写成 print(parent[i],end= " ")
    # 因为只有执行路径压缩getRoot(i)以后，parent[i]才会是i的树根
    print()
```



```python
#深度遍历无向图
def dfs(graph, visited, node):
    visited[node] = True
    print(node, end=" ")

    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(graph, visited, neighbor)

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]
    visited = [False] * n

    for _ in range(m):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)

    for i in range(n):
        if not visited[i]:
            dfs(graph, visited, i)

if __name__ == "__main__":
    main()
```

```python
#奖金 出度入度
import collections
n,m = map(int,input().split())
G = [[] for i in range(n)]
award = [0 for i in range(n)]
inDegree = [0 for i in range(n)]

for i in range(m):
	a,b = map(int,input().split())
	G[b].append(a)
	inDegree[a] += 1
q = collections.deque()
for i in range(n):
	if inDegree[i] == 0:
		q.append(i)
		award[i] = 100
while len(q) > 0:
	u = q.popleft()
	for v in G[u]:
		inDegree[v] -= 1
		award[v] = max(award[v],award[u] + 1)
		if inDegree[v] == 0:
			q.append(v)
total = sum(award)
print(total)
```

```python
#Fraction类
def gcd(m,n):
    while m%n!=0:
        oldm=m
        oldn=n
        
        m=oldn
        n=oldm%oldn
    return n 
    
class Fraction:
    def __init__(self,top,bottom):
        self.num=top
        self.den=bottom
        
    def __str__(self):
        return str(self.num)+"/"+str(self.den)
    
    def __add__(self,otherfraction):
        newnum=self.num*otherfraction.den+self.den*otherfraction.num
        newden=self.den*otherfraction.den
        common=gcd(newnum, newden)
        return Fraction(newnum//common,newden//common)
    
a,b,c,d=map(int,input().split())
A=Fraction(a,b)
B=Fraction(c,d)
print(A+B) 
```

```python
#被5整除
def divisible_by_5(s):
    result = [] 
    num = 0 
    for bit in s: 
        num = (num << 1) + int(bit) 
        result.append(1 if num % 5 == 0 else 0) 
    return ''.join(map(str, result))
input_str = input().strip() 
output_str = divisible_by_5(input_str) 
print(output_str)
```

```python
#走山路
from heapq import heappop, heappush
def bfs(x1, y1): 
    q = [(0, x1, y1)] 
    visited = set() 
    while q:
        t, x, y = heappop(q) 
        if (x, y) in visited: 
            continue 
            visited.add((x, y)) 
            if x == x2 and y == y2: 
                return t for dx, dy in dir: 
                nx, ny = x+dx, y+dy 
                if 0 <= nx < m and 0 <= ny < n and \ ma[nx][ny] != '#' and (nx, ny) not in visited: 
                        nt = t+abs(int(ma[nx][ny])-int(ma[x][y])) 
                        heappush(q, (nt, nx, ny)) 
	return 'NO' 
m, n, p = map(int, input().split()) 
ma = [list(input().split()) for _ in range(m)] 
dir = [(1, 0), (-1, 0), (0, 1), (0, -1)] 
for _ in range(p): 
    x1, y1, x2, y2 = map(int, input().split()) 
    if ma[x1][y1] == '#' or ma[x2][y2] == '#': 
        print('NO') 
        continue 
print(bfs(x1, y1))
```

#树例题

```python
#二叉树深度
class TreeNode:
	def __init__(self):
		self.left = None
		self.right = None
def tree_depth(node):
	if node is None:
		return 0
	left_depth = tree_depth(node.left)
	right_depth = tree_depth(node.right)
	return max(left_depth, right_depth) + 1
n = int(input()) # 读取节点数量
nodes = [TreeNode() for _ in range(n)]
for i in range(n):
	left_index, right_index = map(int, input().split())
	if left_index != -1:
		nodes[i].left = nodes[left_index-1]
	if right_index != -1:
		nodes[i].right = nodes[right_index-1]
root = nodes[0]
depth = tree_depth(root)
print(depth)
```

```python
#二叉树高度和叶子数目
class TreeNode:
	def __init__(self):
		self.left = None
		self.right = None
def tree_height(node):
	if node is None:
		return -1 # 根据定义，空树⾼度为-1
	return max(tree_height(node.left), tree_height(node.right)) + 1
def count_leaves(node):
	if node is None:
		return 0
	if node.left is None and node.right is None:
		return 1
	return count_leaves(node.left) + count_leaves(node.right)
n = int(input()) # 读取节点数量
nodes = [TreeNode() for _ in range(n)]
has_parent = [False] * n # ⽤来标记节点是否有⽗节点
for i in range(n):
	left_index, right_index = map(int, input().split())
	if left_index != -1:
		nodes[i].left = nodes[left_index]
		has_parent[left_index] = True
	if right_index != -1:
		#print(right_index)
		nodes[i].right = nodes[right_index]
		has_parent[right_index] = True
# 寻找根节点，也就是没有⽗节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]
# 计算⾼度和叶⼦节点数
height = tree_height(root)
leaves = count_leaves(root)
print(f"{height} {leaves}")
```

```python
#括号嵌套树
class TreeNode:
	def __init__(self, value): #类似字典
		self.value = value
		self.children = []
def parse_tree(s):
	stack = []
	node = None
	for char in s:
    	if char.isalpha(): # 如果是字⺟，创建新节点
			node = TreeNode(char)
			if stack: # 如果栈不为空，把节点作为⼦节点加⼊到栈顶节点的⼦节点列表中
				stack[-1].children.append(node)
		elif char == '(': # 遇到左括号，当前节点可能会有⼦节点
			if node:
				stack.append(node) # 把当前节点推⼊栈中
				node = None
		elif char == ')': # 遇到右括号，⼦节点列表结束
			if stack:
				node = stack.pop() # 弹出当前节点
	return node # 根节点
def preorder(node):
	output = [node.value]
	for child in node.children:
		output.extend(preorder(child))
	return ''.join(output)
def postorder(node):
	output = []
	for child in node.children:
		output.extend(postorder(child))
	output.append(node.value)
	return ''.join(output)
# 主程序
def main():
	s = input().strip()
	s = ''.join(s.split()) # 去掉所有空⽩字符
	root = parse_tree(s) # 解析整棵树
	if root:
		print(preorder(root)) # 输出前序遍历序列
		print(postorder(root)) # 输出后序遍历序列
	else:
		print("input tree string error!")
if __name__ == "__main__":
	main()
```

```python
#后续表达式转队列表达式
class TreeNode:
	def __init__(self, value):
		self.value = value
		self.left = None
		self.right = None
def build_tree(postfix):
	stack = []
	for char in postfix:
		node = TreeNode(char)
		if char.isupper():
			node.right = stack.pop()
			node.left = stack.pop()
		stack.append(node)
	return stack[0]
def level_order_traversal(root):
	queue = [root]
	traversal = []
	while queue:
		node = queue.pop(0)
		traversal.append(node.value)
		if node.left:
			queue.append(node.left)
		if node.right:
    		queue.append(node.right)
	return traversal
n = int(input().strip())
for _ in range(n):
	postfix = input().strip()
	root = build_tree(postfix)
	queue_expression = level_order_traversal(root)[::-1]
	print(''.join(queue_expression))
```

```python
#给定中序和后序求前序
def build_tree(inorder,postorder):
    if not inorder or not postorder:
        return []
    
    root_val=postorder[-1]
    root_index=inorder.index(root_val)
    
    left_inorder=inorder[:root_index]
    right_inorder=inorder[root_index:]
    
    left_postorder=postorder[:len(left_inorder)]
    right_postorder=postorder[len(left_inorder):]
    
    root=[root_val]
    root.extend(build_tree(left_inorder, left_postorder))
    root.extend(build_tree(right_inorder, right_postorder))
    
    return root

def main():
    inorder=input().strip()
    postorder=input().strip()
    preorder=build_tree(inorder, postorder)
    print(''.join(preorder))
    
if __name__=='__main__':
    main()
```

```python
#给中序和后序求前序
from collections import deque
class Node:
	def __init__(self, data):
		self.data = data
		self.left = None
		self.right = None
def build_tree(inorder, postorder):
	if inorder:
		root = Node(postorder.pop())
		root_index = inorder.index(root.data)
		root.right = build_tree(inorder[root_index+1:], postorder)
		root.left = build_tree(inorder[:root_index], postorder)
		return root
def level_order_traversal(root):
	if root is None:
		return []
	result = []
	queue = deque([root])
	while queue:
		node = queue.popleft()
		result.append(node.data)
		if node.left:
			queue.append(node.left)
		if node.right:
			queue.append(node.right)
	return result
n = int(input())
for _ in range(n):
	inorder = list(input().strip())
	postorder = list(input().strip())
	root = build_tree(inorder, postorder)
	print(''.join(level_order_traversal(root)))
```

```python
#剪绳子
import sys
try: fin = open('test.in','r').readline
except: fin = sys.stdin.readline
n = int(fin())
import heapq
a = list(map(int, fin().split()))
heapq.heapify(a)
ans = 0
for i in range(n-1):
	x = heapq.heappop(a)
	y = heapq.heappop(a)
	z = x + y
	heapq.heappush(a, z)
	ans += z
print(ans)
```

```python
#给前序遍历求后序遍历
class Node():
	def __init__(self, val):
		self.val = val
		self.left = None
		self.right = None
def buildTree(preorder):
	if len(preorder) == 0:
		return None
	node = Node(preorder[0])
	idx = len(preorder)
	for i in range(1, len(preorder)):
        if preorder[i] > preorder[0]:
			idx = i
			break
	node.left = buildTree(preorder[1:idx])
	node.right = buildTree(preorder[idx:])
	return node
def postorder(node):
	if node is None:
		return []
	output = []
	output.extend(postorder(node.left))
	output.extend(postorder(node.right))
	output.append(str(node.val))
	return output
n = int(input())
preorder = list(map(int, input().split()))
print(' '.join(postorder(buildTree(preorder))))
```

```python
#二叉搜索树层次遍历
class TreeNode:
	def __init__(self, value):
		self.value = value
		self.left = None
		self.right = None
def insert(node, value):
	if node is None:
		return TreeNode(value)
	if value < node.value:
		node.left = insert(node.left, value)
	elif value > node.value:
		node.right = insert(node.right, value)
	return node
def level_order_traversal(root):
	queue = [root]
	traversal = []
    while queue:
		node = queue.pop(0)
		traversal.append(node.value)
		if node.left:
			queue.append(node.left)
		if node.right:
			queue.append(node.right)
	return traversal
numbers = list(map(int, input().strip().split()))
numbers = list(dict.fromkeys(numbers)) # remove duplicates
root = None
for number in numbers:
	root = insert(root, number)
traversal = level_order_traversal(root)
print(' '.join(map(str, traversal)))
```

图题目

```python
#无向图的连接矩阵
n, m = map(int, input().split())
adjacency_matrix = [[0]*n for _ in range(n)]
for _ in range(m):
	u, v = map(int, input().split())
	adjacency_matrix[u][v] = 1
	adjacency_matrix[v][u] = 1
for row in adjacency_matrix:
	print(' '.join(map(str, row)))
```

```python
#无向图是否连通
def dfs(node,visited,adjacency):
    visited[node]=True
    for nbr in adjacency[node]:
        if not visited[nbr]:
            dfs(nbr, visited, adjacency)

n,m=map(int,input().split())
adjacency=[[]for i in range(n)]
for i in range(m):
    u,v=map(int,input().split())
    adjacency[u].append(v)
    adjacency[v].append(u)
count=True

visited=[False]*n
dfs(0, visited, adjacency)
for i in range(n):
    
    if not visited[i]:
        dfs(i, visited, adjacency)
        count=False
if count:
    print('YES')
else:
    print('NO')
```

```python
#有向图判环
def has_cycle(n,edges):
    graph=[[]for i in range(n)]
    for u,v in edges:
        graph[u].append(v)
        
    color=[0]*n
    
    def dfs(node):
        if color[node]==1:
            return True
        if color[node]==2:
            return False
        
        color[node]=1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node]=2
        return False
    num=0
    for i in range(n):
        if dfs(i):
            num+=1
            return 'YES'+f'{num}'
    return'NO'

n,m=map(int,input().split())
edges=[]
adjacency=[[]for i in range(n)]
for i in range(m):
     u,v=map(int,input().split())
     edges.append((u,v))   
print(has_cycle(n, edges))
```

```python
#DFS 最大权值连通块
def max_weight(n,m,weights,edges):
    graph=[[]for i in range(n)]
    for u,v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited=[False]*n
    max_weight=0
    
    def dfs(node):
        visited[node]=True
        total_weight=weights[node]
        for nbr in graph[node]:
            if not visited[nbr]:
                total_weight+=dfs(nbr)
        return total_weight
    
    for i in range(n):
        if not visited[i]:
            max_weight=max(max_weight,dfs(i))
            
    return max_weight

n,m=map(int,input().split())
weights=list(map(int, input().split()))
edges=[]
adjacency=[[]for i in range(n)]
for i in range(m):
     u,v=map(int,input().split())
     edges.append((u,v))   
print(max_weight(n, m, weights, edges))
```

```python
#BFS 寻找顶点层号
from collections import deque

def bfs(n,m,s,edges):
    graph=[[]for i in range(n)]
    for u,v in edges:
        graph[u].append(v)
        graph[v].append(u)
        
    distance=[-1]*n
    distance[s]=0
    
    queue=deque([s])
    while queue:
        node=queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor]==-1:
                distance[neighbor]=distance[node]+1
                queue.append(neighbor)
    return distance

n,m,s=map(int,input().split())
edges=[]
for i in range(m):
    u,v=map(int,input().split())
    edges.append((u,v))
distance=bfs(n, m, s, edges)
print(' '.join(map(str, distance)))
```

```python
#Dijkstra 从s到t最短距离

import heapq

def dijkstra(n,edges,s,t):
    graph=[[]for i in range(n)]
    for u,v,w in edges:
        graph[u].append((v,w))
        graph[v].append((u,w))
        
    pq=[(0,s)]
    visited=set()
    distances=[float('inf')]*n
    distances[s]=0
    
    while pq:
        dist,node=heapq.heappop(pq)
        if node==t:
            return dist
        if node in visited:
            continue
        visited.add(node)
        for neighbor,weight in graph[node]:
            if neighbor not in visited:
                new_dist=dist+weight
                if new_dist<distances[neighbor]:
                    distances[neighbor]=new_dist
                    heapq.heappush(pq, (new_dist,neighbor))
        
    return -1

n,m,s,t=map(int,input().split())
edges=[list(map(int,input().split()))for i in range(m)]

result=dijkstra(n, edges, s, t)
print(result)
```

```python
#Prim 寻找最小生成树

import heapq

def prim(graph,n):
    visited=[False]*n
    min_heap=[(0,0)]
    min_spanning_tree_cost=0
    
    while min_heap:
        weight,vertex=heapq.heappop(min_heap)
        
        if visited[vertex]:
            continue
        
        visited[vertex]=True
        min_spanning_tree_cost+=weight
        
        for neighbor,neighbor_weight in graph[vertex]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (neighbor_weight,neighbor))
    
    return min_spanning_tree_cost if all(visited)else -1

def main():
    n,m=map(int, input().split())
    graph=[[]for i in range(n)]
    
    for i in range(m):
            u,v,w = map(int, input().split())
            graph[u].append((v,w))
            graph[v].append((u,w))
    min_spanning_tree_cost=prim(graph, n)
    print(min_spanning_tree_cost)

if __name__ =='__main__':
    main()
```

```python
#Kruskal 寻找最小生成树
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1
def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    res = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            res += w
    if len(set(uf.find(i) for i in range(n))) > 1:
        return -1
    return res
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))
print(kruskal(n, edges))
```

```python
#拓扑排序

from collections import defaultdict

def courseSchedule(n,edges):
    graph=defaultdict(list)
    indegree=[0]*n
    for u,v in edges:
        graph[u].append(v)
        indegree+=1
        
    queue=[i for i in range(n)if indegree[i]==0]
    queue.sort()
    result=[]
    
    while queue:
        u=queue.pop(0)
        result.append(u)
        for v in graph[u]:
            indegree[v]-=1
            if indegree[v]==0:
                queue.append(v)
        queue.sort()
        
    if len(result)==n:
        return'YES',result
    else:
        return 'NO',n-len(result)
    
n,m=map(int, input().split())
edges=[list(map(int,input().split()))for _ in range(m)]
res,courses=courseSchedule(n, edges)
print(res)
if res=='YES':
    print(*courses)
else:
    print(courses)
```

