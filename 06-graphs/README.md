# Chapter 6: 그래프 (Graphs)

## 6.1 그래프란?

### 정의
그래프는 정점(Vertex)과 간선(Edge)으로 구성된 비선형 자료구조입니다. 객체 간의 관계를 표현하는 데 사용되며, 네트워크, 경로 찾기, 소셜 네트워크 등 다양한 분야에서 활용됩니다.

### 그래프 용어
- **정점(Vertex/Node)**: 그래프의 기본 단위
- **간선(Edge)**: 정점을 연결하는 선
- **인접(Adjacent)**: 간선으로 직접 연결된 정점들
- **차수(Degree)**: 정점에 연결된 간선의 수
- **경로(Path)**: 정점들의 연속된 시퀀스
- **사이클(Cycle)**: 시작점과 끝점이 같은 경로
- **연결 그래프**: 모든 정점이 연결된 그래프
- **가중치(Weight)**: 간선에 할당된 값

### 그래프의 종류
1. **방향 그래프(Directed Graph)**: 간선에 방향이 있음
2. **무방향 그래프(Undirected Graph)**: 간선에 방향이 없음
3. **가중치 그래프(Weighted Graph)**: 간선에 가중치가 있음
4. **비가중치 그래프(Unweighted Graph)**: 간선에 가중치가 없음
5. **순환 그래프(Cyclic Graph)**: 사이클이 있음
6. **비순환 그래프(Acyclic Graph)**: 사이클이 없음
7. **완전 그래프(Complete Graph)**: 모든 정점이 서로 연결됨
8. **이분 그래프(Bipartite Graph)**: 정점을 두 그룹으로 나눌 수 있음

## 6.2 그래프의 표현

### 인접 행렬 (Adjacency Matrix)
2차원 배열로 그래프를 표현합니다. `matrix[i][j]`는 정점 i에서 j로의 간선 존재 여부를 나타냅니다.

```java
public class AdjacencyMatrixGraph {
    private int[][] adjacencyMatrix;
    private int numVertices;
    
    public AdjacencyMatrixGraph(int numVertices) {
        this.numVertices = numVertices;
        this.adjacencyMatrix = new int[numVertices][numVertices];
    }
    
    // 간선 추가 (무방향 그래프)
    public void addEdge(int source, int destination) {
        adjacencyMatrix[source][destination] = 1;
        adjacencyMatrix[destination][source] = 1;
    }
    
    // 간선 추가 (방향 그래프)
    public void addDirectedEdge(int source, int destination) {
        adjacencyMatrix[source][destination] = 1;
    }
    
    // 가중치 간선 추가
    public void addWeightedEdge(int source, int destination, int weight) {
        adjacencyMatrix[source][destination] = weight;
        adjacencyMatrix[destination][source] = weight; // 무방향
    }
    
    // 간선 제거
    public void removeEdge(int source, int destination) {
        adjacencyMatrix[source][destination] = 0;
        adjacencyMatrix[destination][source] = 0;
    }
    
    // 간선 존재 확인
    public boolean hasEdge(int source, int destination) {
        return adjacencyMatrix[source][destination] != 0;
    }
    
    // 정점의 인접 정점들 반환
    public List<Integer> getNeighbors(int vertex) {
        List<Integer> neighbors = new ArrayList<>();
        for (int i = 0; i < numVertices; i++) {
            if (adjacencyMatrix[vertex][i] != 0) {
                neighbors.add(i);
            }
        }
        return neighbors;
    }
    
    // 정점의 차수 계산
    public int getDegree(int vertex) {
        int degree = 0;
        for (int i = 0; i < numVertices; i++) {
            if (adjacencyMatrix[vertex][i] != 0) {
                degree++;
            }
        }
        return degree;
    }
    
    // 그래프 출력
    public void printGraph() {
        System.out.println("Adjacency Matrix:");
        for (int i = 0; i < numVertices; i++) {
            for (int j = 0; j < numVertices; j++) {
                System.out.print(adjacencyMatrix[i][j] + " ");
            }
            System.out.println();
        }
    }
}
```

### 인접 리스트 (Adjacency List)
각 정점마다 인접한 정점들의 리스트를 저장합니다.

```java
public class AdjacencyListGraph {
    private Map<Integer, List<Edge>> adjacencyList;
    private int numVertices;
    private boolean isDirected;
    
    // 간선 클래스
    static class Edge {
        int destination;
        int weight;
        
        Edge(int destination, int weight) {
            this.destination = destination;
            this.weight = weight;
        }
        
        Edge(int destination) {
            this(destination, 1);
        }
    }
    
    public AdjacencyListGraph(int numVertices, boolean isDirected) {
        this.numVertices = numVertices;
        this.isDirected = isDirected;
        this.adjacencyList = new HashMap<>();
        
        for (int i = 0; i < numVertices; i++) {
            adjacencyList.put(i, new ArrayList<>());
        }
    }
    
    // 간선 추가
    public void addEdge(int source, int destination) {
        adjacencyList.get(source).add(new Edge(destination));
        
        if (!isDirected) {
            adjacencyList.get(destination).add(new Edge(source));
        }
    }
    
    // 가중치 간선 추가
    public void addWeightedEdge(int source, int destination, int weight) {
        adjacencyList.get(source).add(new Edge(destination, weight));
        
        if (!isDirected) {
            adjacencyList.get(destination).add(new Edge(source, weight));
        }
    }
    
    // 간선 제거
    public void removeEdge(int source, int destination) {
        List<Edge> sourceEdges = adjacencyList.get(source);
        sourceEdges.removeIf(edge -> edge.destination == destination);
        
        if (!isDirected) {
            List<Edge> destEdges = adjacencyList.get(destination);
            destEdges.removeIf(edge -> edge.destination == source);
        }
    }
    
    // 간선 존재 확인
    public boolean hasEdge(int source, int destination) {
        List<Edge> edges = adjacencyList.get(source);
        return edges.stream().anyMatch(edge -> edge.destination == destination);
    }
    
    // 인접 정점들 반환
    public List<Integer> getNeighbors(int vertex) {
        return adjacencyList.get(vertex).stream()
                .map(edge -> edge.destination)
                .collect(Collectors.toList());
    }
    
    // 정점의 차수
    public int getDegree(int vertex) {
        if (isDirected) {
            return getOutDegree(vertex);
        }
        return adjacencyList.get(vertex).size();
    }
    
    // 진출 차수 (방향 그래프)
    public int getOutDegree(int vertex) {
        return adjacencyList.get(vertex).size();
    }
    
    // 진입 차수 (방향 그래프)
    public int getInDegree(int vertex) {
        int inDegree = 0;
        for (List<Edge> edges : adjacencyList.values()) {
            for (Edge edge : edges) {
                if (edge.destination == vertex) {
                    inDegree++;
                }
            }
        }
        return inDegree;
    }
}
```

## 6.3 그래프 순회

### 깊이 우선 탐색 (DFS - Depth First Search)
스택을 사용하여 깊이를 우선으로 탐색합니다.

```java
public class GraphTraversal {
    
    // 재귀적 DFS
    public void dfsRecursive(AdjacencyListGraph graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        dfsUtil(graph, start, visited);
    }
    
    private void dfsUtil(AdjacencyListGraph graph, int vertex, boolean[] visited) {
        visited[vertex] = true;
        System.out.print(vertex + " ");
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                dfsUtil(graph, neighbor, visited);
            }
        }
    }
    
    // 반복적 DFS
    public void dfsIterative(AdjacencyListGraph graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        
        while (!stack.isEmpty()) {
            int vertex = stack.pop();
            
            if (!visited[vertex]) {
                visited[vertex] = true;
                System.out.print(vertex + " ");
                
                for (Integer neighbor : graph.getNeighbors(vertex)) {
                    if (!visited[neighbor]) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }
    
    // 모든 컴포넌트 방문
    public void dfsAllComponents(AdjacencyListGraph graph) {
        boolean[] visited = new boolean[graph.numVertices];
        int components = 0;
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (!visited[i]) {
                System.out.print("Component " + (++components) + ": ");
                dfsUtil(graph, i, visited);
                System.out.println();
            }
        }
    }
}
```

### 너비 우선 탐색 (BFS - Breadth First Search)
큐를 사용하여 너비를 우선으로 탐색합니다.

```java
public class BreadthFirstSearch {
    
    // BFS 구현
    public void bfs(AdjacencyListGraph graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        Queue<Integer> queue = new LinkedList<>();
        
        visited[start] = true;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");
            
            for (Integer neighbor : graph.getNeighbors(vertex)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
    }
    
    // 최단 경로 찾기 (비가중치 그래프)
    public List<Integer> shortestPath(AdjacencyListGraph graph, int start, int end) {
        boolean[] visited = new boolean[graph.numVertices];
        int[] parent = new int[graph.numVertices];
        Arrays.fill(parent, -1);
        
        Queue<Integer> queue = new LinkedList<>();
        visited[start] = true;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            
            if (vertex == end) {
                return constructPath(parent, start, end);
            }
            
            for (Integer neighbor : graph.getNeighbors(vertex)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    parent[neighbor] = vertex;
                    queue.offer(neighbor);
                }
            }
        }
        
        return new ArrayList<>(); // 경로 없음
    }
    
    private List<Integer> constructPath(int[] parent, int start, int end) {
        List<Integer> path = new ArrayList<>();
        int current = end;
        
        while (current != -1) {
            path.add(0, current);
            current = parent[current];
        }
        
        return path;
    }
    
    // 레벨별 탐색
    public void levelOrderTraversal(AdjacencyListGraph graph, int start) {
        boolean[] visited = new boolean[graph.numVertices];
        Queue<Integer> queue = new LinkedList<>();
        
        visited[start] = true;
        queue.offer(start);
        
        int level = 0;
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            System.out.print("Level " + level + ": ");
            
            for (int i = 0; i < levelSize; i++) {
                int vertex = queue.poll();
                System.out.print(vertex + " ");
                
                for (Integer neighbor : graph.getNeighbors(vertex)) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.offer(neighbor);
                    }
                }
            }
            
            System.out.println();
            level++;
        }
    }
}
```

## 6.4 최단 경로 알고리즘

### 다익스트라 알고리즘 (Dijkstra's Algorithm)
단일 출발점에서 모든 정점까지의 최단 경로를 찾습니다. (음의 가중치 불가)

```java
public class DijkstraAlgorithm {
    
    static class Node implements Comparable<Node> {
        int vertex;
        int distance;
        
        Node(int vertex, int distance) {
            this.vertex = vertex;
            this.distance = distance;
        }
        
        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.distance, other.distance);
        }
    }
    
    public int[] dijkstra(AdjacencyListGraph graph, int start) {
        int[] distances = new int[graph.numVertices];
        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[start] = 0;
        
        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.offer(new Node(start, 0));
        
        boolean[] visited = new boolean[graph.numVertices];
        
        while (!pq.isEmpty()) {
            Node current = pq.poll();
            int u = current.vertex;
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (AdjacencyListGraph.Edge edge : graph.adjacencyList.get(u)) {
                int v = edge.destination;
                int weight = edge.weight;
                
                if (distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    pq.offer(new Node(v, distances[v]));
                }
            }
        }
        
        return distances;
    }
    
    // 경로 복원 포함
    public class DijkstraResult {
        int[] distances;
        int[] parent;
        
        DijkstraResult(int size) {
            distances = new int[size];
            parent = new int[size];
            Arrays.fill(distances, Integer.MAX_VALUE);
            Arrays.fill(parent, -1);
        }
    }
    
    public DijkstraResult dijkstraWithPath(AdjacencyListGraph graph, int start) {
        DijkstraResult result = new DijkstraResult(graph.numVertices);
        result.distances[start] = 0;
        
        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.offer(new Node(start, 0));
        
        boolean[] visited = new boolean[graph.numVertices];
        
        while (!pq.isEmpty()) {
            Node current = pq.poll();
            int u = current.vertex;
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (AdjacencyListGraph.Edge edge : graph.adjacencyList.get(u)) {
                int v = edge.destination;
                int weight = edge.weight;
                
                if (result.distances[u] + weight < result.distances[v]) {
                    result.distances[v] = result.distances[u] + weight;
                    result.parent[v] = u;
                    pq.offer(new Node(v, result.distances[v]));
                }
            }
        }
        
        return result;
    }
    
    // 최단 경로 출력
    public void printPath(DijkstraResult result, int start, int end) {
        if (result.distances[end] == Integer.MAX_VALUE) {
            System.out.println("No path exists");
            return;
        }
        
        Stack<Integer> path = new Stack<>();
        int current = end;
        
        while (current != -1) {
            path.push(current);
            current = result.parent[current];
        }
        
        System.out.print("Path: ");
        while (!path.isEmpty()) {
            System.out.print(path.pop());
            if (!path.isEmpty()) {
                System.out.print(" -> ");
            }
        }
        System.out.println("\nDistance: " + result.distances[end]);
    }
}
```

### 벨만-포드 알고리즘 (Bellman-Ford Algorithm)
음의 가중치를 허용하며, 음의 사이클을 감지할 수 있습니다.

```java
public class BellmanFordAlgorithm {
    
    static class Edge {
        int source;
        int destination;
        int weight;
        
        Edge(int source, int destination, int weight) {
            this.source = source;
            this.destination = destination;
            this.weight = weight;
        }
    }
    
    public class BellmanFordResult {
        int[] distances;
        int[] parent;
        boolean hasNegativeCycle;
        
        BellmanFordResult(int size) {
            distances = new int[size];
            parent = new int[size];
            Arrays.fill(distances, Integer.MAX_VALUE);
            Arrays.fill(parent, -1);
            hasNegativeCycle = false;
        }
    }
    
    public BellmanFordResult bellmanFord(List<Edge> edges, int numVertices, int start) {
        BellmanFordResult result = new BellmanFordResult(numVertices);
        result.distances[start] = 0;
        
        // V-1번 반복
        for (int i = 0; i < numVertices - 1; i++) {
            for (Edge edge : edges) {
                if (result.distances[edge.source] != Integer.MAX_VALUE &&
                    result.distances[edge.source] + edge.weight < result.distances[edge.destination]) {
                    result.distances[edge.destination] = result.distances[edge.source] + edge.weight;
                    result.parent[edge.destination] = edge.source;
                }
            }
        }
        
        // 음의 사이클 검사
        for (Edge edge : edges) {
            if (result.distances[edge.source] != Integer.MAX_VALUE &&
                result.distances[edge.source] + edge.weight < result.distances[edge.destination]) {
                result.hasNegativeCycle = true;
                break;
            }
        }
        
        return result;
    }
}
```

### 플로이드-워셜 알고리즘 (Floyd-Warshall Algorithm)
모든 정점 쌍 간의 최단 경로를 찾습니다.

```java
public class FloydWarshallAlgorithm {
    
    public int[][] floydWarshall(int[][] graph) {
        int n = graph.length;
        int[][] dist = new int[n][n];
        
        // 초기화
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    dist[i][j] = 0;
                } else if (graph[i][j] != 0) {
                    dist[i][j] = graph[i][j];
                } else {
                    dist[i][j] = Integer.MAX_VALUE / 2; // 오버플로우 방지
                }
            }
        }
        
        // 플로이드-워셜 알고리즘
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        
        return dist;
    }
    
    // 경로 복원 포함
    public class FloydWarshallResult {
        int[][] distances;
        int[][] next;
        
        FloydWarshallResult(int n) {
            distances = new int[n][n];
            next = new int[n][n];
            
            for (int i = 0; i < n; i++) {
                Arrays.fill(next[i], -1);
            }
        }
    }
    
    public FloydWarshallResult floydWarshallWithPath(int[][] graph) {
        int n = graph.length;
        FloydWarshallResult result = new FloydWarshallResult(n);
        
        // 초기화
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    result.distances[i][j] = 0;
                } else if (graph[i][j] != 0) {
                    result.distances[i][j] = graph[i][j];
                    result.next[i][j] = j;
                } else {
                    result.distances[i][j] = Integer.MAX_VALUE / 2;
                }
            }
        }
        
        // 플로이드-워셜
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (result.distances[i][k] + result.distances[k][j] < result.distances[i][j]) {
                        result.distances[i][j] = result.distances[i][k] + result.distances[k][j];
                        result.next[i][j] = result.next[i][k];
                    }
                }
            }
        }
        
        return result;
    }
    
    // 경로 출력
    public List<Integer> getPath(FloydWarshallResult result, int start, int end) {
        if (result.next[start][end] == -1) {
            return new ArrayList<>();
        }
        
        List<Integer> path = new ArrayList<>();
        path.add(start);
        
        while (start != end) {
            start = result.next[start][end];
            path.add(start);
        }
        
        return path;
    }
}
```

## 6.5 최소 신장 트리 (Minimum Spanning Tree)

### 크루스칼 알고리즘 (Kruskal's Algorithm)
간선을 가중치 순으로 정렬하여 사이클을 만들지 않는 간선을 선택합니다.

```java
public class KruskalAlgorithm {
    
    static class Edge implements Comparable<Edge> {
        int source;
        int destination;
        int weight;
        
        Edge(int source, int destination, int weight) {
            this.source = source;
            this.destination = destination;
            this.weight = weight;
        }
        
        @Override
        public int compareTo(Edge other) {
            return Integer.compare(this.weight, other.weight);
        }
    }
    
    // Union-Find 자료구조
    static class UnionFind {
        int[] parent;
        int[] rank;
        
        UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
        
        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // 경로 압축
            }
            return parent[x];
        }
        
        boolean union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            
            if (rootX == rootY) {
                return false;
            }
            
            // 랭크에 의한 합병
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            
            return true;
        }
    }
    
    public List<Edge> kruskal(List<Edge> edges, int numVertices) {
        List<Edge> mst = new ArrayList<>();
        Collections.sort(edges);
        
        UnionFind uf = new UnionFind(numVertices);
        
        for (Edge edge : edges) {
            if (uf.union(edge.source, edge.destination)) {
                mst.add(edge);
                if (mst.size() == numVertices - 1) {
                    break;
                }
            }
        }
        
        return mst;
    }
    
    // MST의 총 가중치 계산
    public int getMSTWeight(List<Edge> mst) {
        return mst.stream().mapToInt(e -> e.weight).sum();
    }
}
```

### 프림 알고리즘 (Prim's Algorithm)
하나의 정점에서 시작하여 최소 가중치 간선을 선택하며 확장합니다.

```java
public class PrimAlgorithm {
    
    static class Edge implements Comparable<Edge> {
        int vertex;
        int weight;
        
        Edge(int vertex, int weight) {
            this.vertex = vertex;
            this.weight = weight;
        }
        
        @Override
        public int compareTo(Edge other) {
            return Integer.compare(this.weight, other.weight);
        }
    }
    
    public List<Edge> prim(AdjacencyListGraph graph, int start) {
        List<Edge> mst = new ArrayList<>();
        boolean[] inMST = new boolean[graph.numVertices];
        PriorityQueue<Edge> pq = new PriorityQueue<>();
        
        // 시작 정점 처리
        inMST[start] = true;
        for (AdjacencyListGraph.Edge edge : graph.adjacencyList.get(start)) {
            pq.offer(new Edge(edge.destination, edge.weight));
        }
        
        while (!pq.isEmpty() && mst.size() < graph.numVertices - 1) {
            Edge current = pq.poll();
            
            if (inMST[current.vertex]) {
                continue;
            }
            
            mst.add(current);
            inMST[current.vertex] = true;
            
            // 새로 추가된 정점의 인접 간선들 추가
            for (AdjacencyListGraph.Edge edge : graph.adjacencyList.get(current.vertex)) {
                if (!inMST[edge.destination]) {
                    pq.offer(new Edge(edge.destination, edge.weight));
                }
            }
        }
        
        return mst;
    }
}
```

## 6.6 위상 정렬 (Topological Sort)

### 위상 정렬의 정의
방향 비순환 그래프(DAG)의 정점들을 간선의 방향을 거스르지 않도록 나열하는 것입니다.

```java
public class TopologicalSort {
    
    // DFS 기반 위상 정렬
    public List<Integer> topologicalSortDFS(AdjacencyListGraph graph) {
        boolean[] visited = new boolean[graph.numVertices];
        Stack<Integer> stack = new Stack<>();
        
        // 모든 정점에 대해 DFS 수행
        for (int i = 0; i < graph.numVertices; i++) {
            if (!visited[i]) {
                topologicalSortUtil(graph, i, visited, stack);
            }
        }
        
        // 스택에서 결과 추출
        List<Integer> result = new ArrayList<>();
        while (!stack.isEmpty()) {
            result.add(stack.pop());
        }
        
        return result;
    }
    
    private void topologicalSortUtil(AdjacencyListGraph graph, int vertex,
                                   boolean[] visited, Stack<Integer> stack) {
        visited[vertex] = true;
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                topologicalSortUtil(graph, neighbor, visited, stack);
            }
        }
        
        stack.push(vertex);
    }
    
    // Kahn's Algorithm (BFS 기반)
    public List<Integer> topologicalSortBFS(AdjacencyListGraph graph) {
        int[] inDegree = new int[graph.numVertices];
        
        // 진입 차수 계산
        for (int i = 0; i < graph.numVertices; i++) {
            for (Integer neighbor : graph.getNeighbors(i)) {
                inDegree[neighbor]++;
            }
        }
        
        // 진입 차수가 0인 정점들을 큐에 추가
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < graph.numVertices; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            result.add(vertex);
            
            // 인접 정점들의 진입 차수 감소
            for (Integer neighbor : graph.getNeighbors(vertex)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        
        // 사이클 검사
        if (result.size() != graph.numVertices) {
            throw new RuntimeException("Graph has a cycle!");
        }
        
        return result;
    }
    
    // 모든 위상 정렬 찾기
    public void allTopologicalSorts(AdjacencyListGraph graph) {
        boolean[] visited = new boolean[graph.numVertices];
        int[] inDegree = new int[graph.numVertices];
        List<Integer> result = new ArrayList<>();
        
        // 진입 차수 계산
        for (int i = 0; i < graph.numVertices; i++) {
            for (Integer neighbor : graph.getNeighbors(i)) {
                inDegree[neighbor]++;
            }
        }
        
        allTopologicalSortsUtil(graph, visited, inDegree, result);
    }
    
    private void allTopologicalSortsUtil(AdjacencyListGraph graph, boolean[] visited,
                                       int[] inDegree, List<Integer> result) {
        boolean flag = false;
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (!visited[i] && inDegree[i] == 0) {
                visited[i] = true;
                result.add(i);
                
                for (Integer neighbor : graph.getNeighbors(i)) {
                    inDegree[neighbor]--;
                }
                
                allTopologicalSortsUtil(graph, visited, inDegree, result);
                
                // 백트래킹
                visited[i] = false;
                result.remove(result.size() - 1);
                for (Integer neighbor : graph.getNeighbors(i)) {
                    inDegree[neighbor]++;
                }
                
                flag = true;
            }
        }
        
        if (!flag) {
            System.out.println(result);
        }
    }
}
```

## 6.7 그래프 응용 문제

### 사이클 검출

```java
public class CycleDetection {
    
    // 무방향 그래프의 사이클 검출 (DFS)
    public boolean hasCycleUndirected(AdjacencyListGraph graph) {
        boolean[] visited = new boolean[graph.numVertices];
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (!visited[i]) {
                if (hasCycleUndirectedUtil(graph, i, -1, visited)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private boolean hasCycleUndirectedUtil(AdjacencyListGraph graph, int vertex,
                                         int parent, boolean[] visited) {
        visited[vertex] = true;
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                if (hasCycleUndirectedUtil(graph, neighbor, vertex, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        
        return false;
    }
    
    // 방향 그래프의 사이클 검출 (DFS)
    public boolean hasCycleDirected(AdjacencyListGraph graph) {
        int[] color = new int[graph.numVertices]; // 0: white, 1: gray, 2: black
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (color[i] == 0) {
                if (hasCycleDirectedUtil(graph, i, color)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private boolean hasCycleDirectedUtil(AdjacencyListGraph graph, int vertex, int[] color) {
        color[vertex] = 1; // Gray
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (color[neighbor] == 1) { // Back edge
                return true;
            }
            if (color[neighbor] == 0) {
                if (hasCycleDirectedUtil(graph, neighbor, color)) {
                    return true;
                }
            }
        }
        
        color[vertex] = 2; // Black
        return false;
    }
}
```

### 이분 그래프 판별

```java
public class BipartiteGraph {
    
    public boolean isBipartite(AdjacencyListGraph graph) {
        int[] color = new int[graph.numVertices]; // 0: 미방문, 1: 색1, -1: 색2
        
        for (int i = 0; i < graph.numVertices; i++) {
            if (color[i] == 0) {
                if (!isBipartiteUtil(graph, i, 1, color)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    private boolean isBipartiteUtil(AdjacencyListGraph graph, int vertex,
                                  int c, int[] color) {
        color[vertex] = c;
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (color[neighbor] == 0) {
                if (!isBipartiteUtil(graph, neighbor, -c, color)) {
                    return false;
                }
            } else if (color[neighbor] == c) {
                return false;
            }
        }
        
        return true;
    }
}
```

### 강연결 요소 (Strongly Connected Components)

```java
public class StronglyConnectedComponents {
    
    // Kosaraju's Algorithm
    public List<List<Integer>> findSCCs(AdjacencyListGraph graph) {
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[graph.numVertices];
        
        // 1단계: 종료 시간 순으로 정점들을 스택에 저장
        for (int i = 0; i < graph.numVertices; i++) {
            if (!visited[i]) {
                fillOrder(graph, i, visited, stack);
            }
        }
        
        // 2단계: 그래프 전치
        AdjacencyListGraph transposed = getTranspose(graph);
        
        // 3단계: 전치 그래프에서 DFS 수행
        Arrays.fill(visited, false);
        List<List<Integer>> sccs = new ArrayList<>();
        
        while (!stack.isEmpty()) {
            int vertex = stack.pop();
            if (!visited[vertex]) {
                List<Integer> component = new ArrayList<>();
                dfsUtil(transposed, vertex, visited, component);
                sccs.add(component);
            }
        }
        
        return sccs;
    }
    
    private void fillOrder(AdjacencyListGraph graph, int vertex,
                         boolean[] visited, Stack<Integer> stack) {
        visited[vertex] = true;
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                fillOrder(graph, neighbor, visited, stack);
            }
        }
        
        stack.push(vertex);
    }
    
    private AdjacencyListGraph getTranspose(AdjacencyListGraph graph) {
        AdjacencyListGraph transposed = new AdjacencyListGraph(graph.numVertices, true);
        
        for (int v = 0; v < graph.numVertices; v++) {
            for (Integer neighbor : graph.getNeighbors(v)) {
                transposed.addEdge(neighbor, v);
            }
        }
        
        return transposed;
    }
    
    private void dfsUtil(AdjacencyListGraph graph, int vertex,
                       boolean[] visited, List<Integer> component) {
        visited[vertex] = true;
        component.add(vertex);
        
        for (Integer neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                dfsUtil(graph, neighbor, visited, component);
            }
        }
    }
}
```

## 6.8 실습 문제

### 문제 1: 섬의 개수
2D 그리드에서 연결된 1들의 그룹(섬) 개수를 세는 프로그램을 작성하세요.

### 문제 2: 네트워크 지연 시간
네트워크의 모든 노드에 신호가 도달하는 최소 시간을 계산하세요.

### 문제 3: 과목 이수 순서
선수 과목이 있는 과목들의 이수 순서를 결정하세요.

### 문제 4: 최소 비용으로 모든 도시 연결
모든 도시를 최소 비용으로 연결하는 도로망을 구성하세요.

## 6.9 요약

이 장에서는 그래프의 개념과 다양한 알고리즘에 대해 학습했습니다:

1. **그래프 표현**: 인접 행렬과 인접 리스트
2. **그래프 순회**: DFS와 BFS
3. **최단 경로**: 다익스트라, 벨만-포드, 플로이드-워셜
4. **최소 신장 트리**: 크루스칼, 프림
5. **위상 정렬**: DAG에서의 순서 결정
6. **그래프 응용**: 사이클 검출, 이분 그래프, 강연결 요소

그래프는 실세계의 복잡한 관계를 모델링하는 강력한 도구입니다. 다음 장에서는 효율적인 검색을 위한 해싱에 대해 알아보겠습니다.