# Chapter 10: 고급 자료구조 (Advanced Data Structures)

## 10.1 고급 자료구조 개요

### 고급 자료구조의 필요성
기본적인 자료구조로는 해결하기 어려운 복잡한 문제들을 효율적으로 처리하기 위해 특수한 목적의 고급 자료구조가 필요합니다.

### 고급 자료구조의 특징
1. **특수 목적**: 특정 문제를 해결하기 위해 최적화됨
2. **복잡한 구조**: 여러 기본 자료구조를 조합
3. **효율적인 연산**: 특정 연산에 대해 최적화된 성능
4. **공간과 시간의 트레이드오프**: 메모리를 더 사용하여 속도 향상

## 10.2 트라이 (Trie)

### 트라이의 정의
문자열을 저장하고 효율적으로 탐색하기 위한 트리 기반 자료구조입니다.

```java
public class Trie {
    
    // 트라이 노드
    private static class TrieNode {
        Map<Character, TrieNode> children;
        boolean isEndOfWord;
        int count; // 이 노드를 지나가는 단어의 수
        
        TrieNode() {
            children = new HashMap<>();
            isEndOfWord = false;
            count = 0;
        }
    }
    
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    // 단어 삽입
    public void insert(String word) {
        TrieNode current = root;
        
        for (char ch : word.toCharArray()) {
            current.children.putIfAbsent(ch, new TrieNode());
            current = current.children.get(ch);
            current.count++;
        }
        
        current.isEndOfWord = true;
    }
    
    // 단어 검색
    public boolean search(String word) {
        TrieNode node = searchNode(word);
        return node != null && node.isEndOfWord;
    }
    
    // 접두사 검색
    public boolean startsWith(String prefix) {
        return searchNode(prefix) != null;
    }
    
    private TrieNode searchNode(String str) {
        TrieNode current = root;
        
        for (char ch : str.toCharArray()) {
            if (!current.children.containsKey(ch)) {
                return null;
            }
            current = current.children.get(ch);
        }
        
        return current;
    }
    
    // 단어 삭제
    public boolean delete(String word) {
        return deleteHelper(root, word, 0);
    }
    
    private boolean deleteHelper(TrieNode node, String word, int index) {
        if (index == word.length()) {
            if (!node.isEndOfWord) {
                return false;
            }
            node.isEndOfWord = false;
            node.count--;
            return node.children.isEmpty();
        }
        
        char ch = word.charAt(index);
        TrieNode child = node.children.get(ch);
        
        if (child == null) {
            return false;
        }
        
        boolean shouldDeleteChild = deleteHelper(child, word, index + 1);
        
        if (shouldDeleteChild) {
            node.children.remove(ch);
            return node.children.isEmpty() && !node.isEndOfWord;
        }
        
        child.count--;
        return false;
    }
    
    // 주어진 접두사로 시작하는 모든 단어 찾기
    public List<String> findAllWordsWithPrefix(String prefix) {
        List<String> result = new ArrayList<>();
        TrieNode prefixNode = searchNode(prefix);
        
        if (prefixNode != null) {
            findAllWords(prefixNode, prefix, result);
        }
        
        return result;
    }
    
    private void findAllWords(TrieNode node, String prefix, List<String> result) {
        if (node.isEndOfWord) {
            result.add(prefix);
        }
        
        for (Map.Entry<Character, TrieNode> entry : node.children.entrySet()) {
            findAllWords(entry.getValue(), prefix + entry.getKey(), result);
        }
    }
    
    // 자동 완성 기능
    public List<String> autoComplete(String prefix, int maxSuggestions) {
        List<String> suggestions = new ArrayList<>();
        TrieNode prefixNode = searchNode(prefix);
        
        if (prefixNode != null) {
            autoCompleteHelper(prefixNode, prefix, suggestions, maxSuggestions);
        }
        
        return suggestions;
    }
    
    private void autoCompleteHelper(TrieNode node, String word, 
                                   List<String> suggestions, int maxSuggestions) {
        if (suggestions.size() >= maxSuggestions) {
            return;
        }
        
        if (node.isEndOfWord) {
            suggestions.add(word);
        }
        
        // 알파벳 순서로 탐색
        for (char ch = 'a'; ch <= 'z'; ch++) {
            if (node.children.containsKey(ch)) {
                autoCompleteHelper(node.children.get(ch), word + ch, 
                                 suggestions, maxSuggestions);
            }
        }
    }
    
    // 가장 긴 공통 접두사
    public String longestCommonPrefix() {
        StringBuilder prefix = new StringBuilder();
        TrieNode current = root;
        
        while (current.children.size() == 1 && !current.isEndOfWord) {
            char ch = current.children.keySet().iterator().next();
            prefix.append(ch);
            current = current.children.get(ch);
        }
        
        return prefix.toString();
    }
}

// 압축 트라이 (Radix Tree / Patricia Trie)
public class RadixTree {
    
    private static class RadixNode {
        String key;
        Map<Character, RadixNode> children;
        boolean isEndOfWord;
        
        RadixNode(String key) {
            this.key = key;
            this.children = new HashMap<>();
            this.isEndOfWord = false;
        }
    }
    
    private RadixNode root;
    
    public RadixTree() {
        root = new RadixNode("");
    }
    
    // 삽입
    public void insert(String word) {
        insertHelper(root, word);
    }
    
    private void insertHelper(RadixNode node, String word) {
        if (word.isEmpty()) {
            node.isEndOfWord = true;
            return;
        }
        
        char firstChar = word.charAt(0);
        RadixNode child = node.children.get(firstChar);
        
        if (child == null) {
            // 새 노드 생성
            node.children.put(firstChar, new RadixNode(word));
            node.children.get(firstChar).isEndOfWord = true;
        } else {
            // 공통 접두사 찾기
            String childKey = child.key;
            int commonLength = getCommonPrefixLength(word, childKey);
            
            if (commonLength == childKey.length()) {
                // child의 키가 word의 접두사
                insertHelper(child, word.substring(commonLength));
            } else if (commonLength == word.length()) {
                // word가 child의 키의 접두사
                splitNode(node, firstChar, word, childKey, commonLength);
                node.children.get(firstChar).isEndOfWord = true;
            } else {
                // 부분적으로 일치
                splitNode(node, firstChar, word, childKey, commonLength);
                insertHelper(node.children.get(firstChar), word.substring(commonLength));
            }
        }
    }
    
    private void splitNode(RadixNode parent, char firstChar, 
                          String word, String childKey, int splitIndex) {
        RadixNode child = parent.children.get(firstChar);
        
        // 새로운 부모 노드 생성
        RadixNode newParent = new RadixNode(childKey.substring(0, splitIndex));
        parent.children.put(firstChar, newParent);
        
        // 기존 자식 노드 업데이트
        child.key = childKey.substring(splitIndex);
        newParent.children.put(child.key.charAt(0), child);
    }
    
    private int getCommonPrefixLength(String s1, String s2) {
        int minLength = Math.min(s1.length(), s2.length());
        for (int i = 0; i < minLength; i++) {
            if (s1.charAt(i) != s2.charAt(i)) {
                return i;
            }
        }
        return minLength;
    }
}
```

## 10.3 세그먼트 트리 (Segment Tree)

### 세그먼트 트리의 정의
배열의 구간 질의(range query)를 효율적으로 처리하기 위한 트리 자료구조입니다.

```java
public class SegmentTree {
    private int[] tree;
    private int n;
    
    // 구간 합 세그먼트 트리
    public SegmentTree(int[] arr) {
        n = arr.length;
        tree = new int[4 * n];
        build(arr, 0, 0, n - 1);
    }
    
    // 트리 구성
    private void build(int[] arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node + 1, start, mid);
            build(arr, 2 * node + 2, mid + 1, end);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }
    
    // 구간 질의
    public int query(int l, int r) {
        return query(0, 0, n - 1, l, r);
    }
    
    private int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return 0;
        }
        
        if (l <= start && end <= r) {
            return tree[node];
        }
        
        int mid = (start + end) / 2;
        int leftSum = query(2 * node + 1, start, mid, l, r);
        int rightSum = query(2 * node + 2, mid + 1, end, l, r);
        
        return leftSum + rightSum;
    }
    
    // 값 업데이트
    public void update(int idx, int val) {
        update(0, 0, n - 1, idx, val);
    }
    
    private void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            
            if (idx <= mid) {
                update(2 * node + 1, start, mid, idx, val);
            } else {
                update(2 * node + 2, mid + 1, end, idx, val);
            }
            
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }
}

// Lazy Propagation을 사용한 세그먼트 트리
public class LazySegmentTree {
    private long[] tree;
    private long[] lazy;
    private int n;
    
    public LazySegmentTree(int[] arr) {
        n = arr.length;
        tree = new long[4 * n];
        lazy = new long[4 * n];
        build(arr, 0, 0, n - 1);
    }
    
    private void build(int[] arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node + 1, start, mid);
            build(arr, 2 * node + 2, mid + 1, end);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }
    
    // 지연 전파
    private void pushDown(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node];
            
            if (start != end) {
                lazy[2 * node + 1] += lazy[node];
                lazy[2 * node + 2] += lazy[node];
            }
            
            lazy[node] = 0;
        }
    }
    
    // 구간 업데이트
    public void updateRange(int l, int r, long val) {
        updateRange(0, 0, n - 1, l, r, val);
    }
    
    private void updateRange(int node, int start, int end, int l, int r, long val) {
        pushDown(node, start, end);
        
        if (start > r || end < l) {
            return;
        }
        
        if (l <= start && end <= r) {
            tree[node] += (end - start + 1) * val;
            
            if (start != end) {
                lazy[2 * node + 1] += val;
                lazy[2 * node + 2] += val;
            }
            
            return;
        }
        
        int mid = (start + end) / 2;
        updateRange(2 * node + 1, start, mid, l, r, val);
        updateRange(2 * node + 2, mid + 1, end, l, r, val);
        
        tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
    }
    
    // 구간 질의
    public long queryRange(int l, int r) {
        return queryRange(0, 0, n - 1, l, r);
    }
    
    private long queryRange(int node, int start, int end, int l, int r) {
        pushDown(node, start, end);
        
        if (start > r || end < l) {
            return 0;
        }
        
        if (l <= start && end <= r) {
            return tree[node];
        }
        
        int mid = (start + end) / 2;
        long leftSum = queryRange(2 * node + 1, start, mid, l, r);
        long rightSum = queryRange(2 * node + 2, mid + 1, end, l, r);
        
        return leftSum + rightSum;
    }
}

// 다양한 연산을 지원하는 제네릭 세그먼트 트리
public class GenericSegmentTree<T> {
    private T[] tree;
    private T[] arr;
    private BiFunction<T, T, T> combiner;
    private T identity;
    private int n;
    
    @SuppressWarnings("unchecked")
    public GenericSegmentTree(T[] arr, BiFunction<T, T, T> combiner, T identity) {
        this.arr = arr;
        this.n = arr.length;
        this.combiner = combiner;
        this.identity = identity;
        this.tree = (T[]) new Object[4 * n];
        build(0, 0, n - 1);
    }
    
    private void build(int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(2 * node + 1, start, mid);
            build(2 * node + 2, mid + 1, end);
            tree[node] = combiner.apply(tree[2 * node + 1], tree[2 * node + 2]);
        }
    }
    
    public T query(int l, int r) {
        return query(0, 0, n - 1, l, r);
    }
    
    private T query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return identity;
        }
        
        if (l <= start && end <= r) {
            return tree[node];
        }
        
        int mid = (start + end) / 2;
        T left = query(2 * node + 1, start, mid, l, r);
        T right = query(2 * node + 2, mid + 1, end, l, r);
        
        return combiner.apply(left, right);
    }
}
```

## 10.4 펜윅 트리 (Fenwick Tree / Binary Indexed Tree)

### 펜윅 트리의 정의
누적 합과 구간 합을 효율적으로 계산하기 위한 자료구조입니다.

```java
public class FenwickTree {
    private int[] tree;
    private int n;
    
    public FenwickTree(int n) {
        this.n = n;
        this.tree = new int[n + 1];
    }
    
    // 배열로부터 펜윅 트리 생성
    public FenwickTree(int[] arr) {
        this.n = arr.length;
        this.tree = new int[n + 1];
        
        for (int i = 0; i < n; i++) {
            update(i, arr[i]);
        }
    }
    
    // 값 업데이트
    public void update(int idx, int delta) {
        idx++; // 1-indexed
        
        while (idx <= n) {
            tree[idx] += delta;
            idx += idx & (-idx); // 다음 인덱스로 이동
        }
    }
    
    // 구간 [0, idx]의 합
    public int query(int idx) {
        idx++; // 1-indexed
        int sum = 0;
        
        while (idx > 0) {
            sum += tree[idx];
            idx -= idx & (-idx); // 이전 인덱스로 이동
        }
        
        return sum;
    }
    
    // 구간 [left, right]의 합
    public int rangeQuery(int left, int right) {
        if (left == 0) {
            return query(right);
        }
        return query(right) - query(left - 1);
    }
    
    // 특정 인덱스의 값
    public int get(int idx) {
        return rangeQuery(idx, idx);
    }
    
    // 값 설정 (업데이트가 아닌 설정)
    public void set(int idx, int val) {
        int current = get(idx);
        update(idx, val - current);
    }
}

// 2D 펜윅 트리
public class FenwickTree2D {
    private int[][] tree;
    private int rows, cols;
    
    public FenwickTree2D(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.tree = new int[rows + 1][cols + 1];
    }
    
    public void update(int row, int col, int delta) {
        row++; col++; // 1-indexed
        
        for (int i = row; i <= rows; i += i & (-i)) {
            for (int j = col; j <= cols; j += j & (-j)) {
                tree[i][j] += delta;
            }
        }
    }
    
    public int query(int row, int col) {
        row++; col++; // 1-indexed
        int sum = 0;
        
        for (int i = row; i > 0; i -= i & (-i)) {
            for (int j = col; j > 0; j -= j & (-j)) {
                sum += tree[i][j];
            }
        }
        
        return sum;
    }
    
    public int rangeQuery(int row1, int col1, int row2, int col2) {
        return query(row2, col2) - query(row1 - 1, col2) 
             - query(row2, col1 - 1) + query(row1 - 1, col1 - 1);
    }
}

// 구간 업데이트를 지원하는 펜윅 트리
public class RangeFenwickTree {
    private FenwickTree tree1;
    private FenwickTree tree2;
    private int n;
    
    public RangeFenwickTree(int n) {
        this.n = n;
        this.tree1 = new FenwickTree(n);
        this.tree2 = new FenwickTree(n);
    }
    
    // 구간 [left, right]에 val 추가
    public void rangeUpdate(int left, int right, int val) {
        tree1.update(left, val);
        tree1.update(right + 1, -val);
        tree2.update(left, val * (left - 1));
        tree2.update(right + 1, -val * right);
    }
    
    // 구간 [0, idx]의 합
    public long prefixSum(int idx) {
        return (long)tree1.query(idx) * idx - tree2.query(idx);
    }
    
    // 구간 [left, right]의 합
    public long rangeSum(int left, int right) {
        return prefixSum(right) - (left > 0 ? prefixSum(left - 1) : 0);
    }
}
```

## 10.5 Union-Find (Disjoint Set Union)

### Union-Find의 정의
서로소 집합들을 효율적으로 표현하고 관리하는 자료구조입니다.

```java
public class UnionFind {
    private int[] parent;
    private int[] rank;
    private int count;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        count = n;
        
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }
    
    // Find with path compression
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    // Union by rank
    public boolean union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) {
            return false;
        }
        
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
        
        count--;
        return true;
    }
    
    // 같은 집합에 속하는지 확인
    public boolean connected(int x, int y) {
        return find(x) == find(y);
    }
    
    // 연결 요소의 개수
    public int getCount() {
        return count;
    }
}

// 가중치가 있는 Union-Find
public class WeightedUnionFind {
    private int[] parent;
    private double[] weight;
    
    public WeightedUnionFind(int n) {
        parent = new int[n];
        weight = new double[n];
        
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            weight[i] = 1.0;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            int originalParent = parent[x];
            parent[x] = find(parent[x]);
            weight[x] *= weight[originalParent];
        }
        return parent[x];
    }
    
    public void union(int x, int y, double value) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) {
            return;
        }
        
        parent[rootX] = rootY;
        weight[rootX] = weight[y] * value / weight[x];
    }
    
    public double query(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX != rootY) {
            return -1.0; // 연결되지 않음
        }
        
        return weight[x] / weight[y];
    }
}

// 크기 기반 Union-Find
public class UnionFindBySize {
    private int[] parent;
    private int[] size;
    
    public UnionFindBySize(int n) {
        parent = new int[n];
        size = new int[n];
        
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    public boolean union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY) {
            return false;
        }
        
        if (size[rootX] < size[rootY]) {
            parent[rootX] = rootY;
            size[rootY] += size[rootX];
        } else {
            parent[rootY] = rootX;
            size[rootX] += size[rootY];
        }
        
        return true;
    }
    
    public int getSize(int x) {
        return size[find(x)];
    }
}
```

## 10.6 스킵 리스트 (Skip List)

### 스킵 리스트의 정의
확률적으로 균형을 유지하는 계층적 연결 리스트입니다.

```java
public class SkipList<T extends Comparable<T>> {
    private static final double PROBABILITY = 0.5;
    private static final int MAX_LEVEL = 16;
    
    private class Node {
        T value;
        Node[] forward;
        
        @SuppressWarnings("unchecked")
        Node(T value, int level) {
            this.value = value;
            this.forward = new Node[level + 1];
        }
    }
    
    private Node head;
    private int level;
    private int size;
    private Random random;
    
    public SkipList() {
        this.head = new Node(null, MAX_LEVEL);
        this.level = 0;
        this.size = 0;
        this.random = new Random();
    }
    
    // 랜덤 레벨 생성
    private int randomLevel() {
        int level = 0;
        while (random.nextDouble() < PROBABILITY && level < MAX_LEVEL) {
            level++;
        }
        return level;
    }
    
    // 삽입
    public void insert(T value) {
        @SuppressWarnings("unchecked")
        Node[] update = new Node[MAX_LEVEL + 1];
        Node current = head;
        
        // 삽입 위치 찾기
        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && 
                   current.forward[i].value.compareTo(value) < 0) {
                current = current.forward[i];
            }
            update[i] = current;
        }
        
        current = current.forward[0];
        
        // 중복 확인
        if (current == null || !current.value.equals(value)) {
            int newLevel = randomLevel();
            
            if (newLevel > level) {
                for (int i = level + 1; i <= newLevel; i++) {
                    update[i] = head;
                }
                level = newLevel;
            }
            
            Node newNode = new Node(value, newLevel);
            for (int i = 0; i <= newLevel; i++) {
                newNode.forward[i] = update[i].forward[i];
                update[i].forward[i] = newNode;
            }
            
            size++;
        }
    }
    
    // 검색
    public boolean search(T value) {
        Node current = head;
        
        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && 
                   current.forward[i].value.compareTo(value) < 0) {
                current = current.forward[i];
            }
        }
        
        current = current.forward[0];
        return current != null && current.value.equals(value);
    }
    
    // 삭제
    public boolean delete(T value) {
        @SuppressWarnings("unchecked")
        Node[] update = new Node[MAX_LEVEL + 1];
        Node current = head;
        
        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && 
                   current.forward[i].value.compareTo(value) < 0) {
                current = current.forward[i];
            }
            update[i] = current;
        }
        
        current = current.forward[0];
        
        if (current != null && current.value.equals(value)) {
            for (int i = 0; i <= level; i++) {
                if (update[i].forward[i] != current) {
                    break;
                }
                update[i].forward[i] = current.forward[i];
            }
            
            while (level > 0 && head.forward[level] == null) {
                level--;
            }
            
            size--;
            return true;
        }
        
        return false;
    }
    
    // 범위 검색
    public List<T> range(T min, T max) {
        List<T> result = new ArrayList<>();
        Node current = head;
        
        // min 위치 찾기
        for (int i = level; i >= 0; i--) {
            while (current.forward[i] != null && 
                   current.forward[i].value.compareTo(min) < 0) {
                current = current.forward[i];
            }
        }
        
        current = current.forward[0];
        
        // 범위 내의 모든 값 수집
        while (current != null && current.value.compareTo(max) <= 0) {
            if (current.value.compareTo(min) >= 0) {
                result.add(current.value);
            }
            current = current.forward[0];
        }
        
        return result;
    }
}
```

## 10.7 B-트리 (B-Tree)

### B-트리의 정의
디스크 기반 저장 시스템에 최적화된 자가 균형 트리입니다.

```java
public class BTree<T extends Comparable<T>> {
    private static final int MIN_DEGREE = 3;
    
    private class Node {
        int n; // 현재 키의 개수
        boolean leaf;
        List<T> keys;
        List<Node> children;
        
        Node(boolean leaf) {
            this.leaf = leaf;
            this.keys = new ArrayList<>();
            this.children = new ArrayList<>();
            this.n = 0;
        }
        
        // 노드가 가득 찼는지 확인
        boolean isFull() {
            return n == 2 * MIN_DEGREE - 1;
        }
        
        // 노드가 최소 키 개수를 가지고 있는지 확인
        boolean isMinimal() {
            return n == MIN_DEGREE - 1;
        }
    }
    
    private Node root;
    
    public BTree() {
        root = new Node(true);
    }
    
    // 검색
    public boolean search(T key) {
        return search(root, key);
    }
    
    private boolean search(Node node, T key) {
        int i = 0;
        
        // 키보다 크거나 같은 첫 번째 키 찾기
        while (i < node.n && key.compareTo(node.keys.get(i)) > 0) {
            i++;
        }
        
        // 키를 찾음
        if (i < node.n && key.equals(node.keys.get(i))) {
            return true;
        }
        
        // 리프 노드면 키가 없음
        if (node.leaf) {
            return false;
        }
        
        // 적절한 자식에서 재귀적으로 검색
        return search(node.children.get(i), key);
    }
    
    // 삽입
    public void insert(T key) {
        Node r = root;
        
        if (r.isFull()) {
            Node s = new Node(false);
            root = s;
            s.children.add(r);
            splitChild(s, 0);
            insertNonFull(s, key);
        } else {
            insertNonFull(r, key);
        }
    }
    
    private void insertNonFull(Node node, T key) {
        int i = node.n - 1;
        
        if (node.leaf) {
            // 리프 노드에 삽입
            node.keys.add(null);
            while (i >= 0 && key.compareTo(node.keys.get(i)) < 0) {
                node.keys.set(i + 1, node.keys.get(i));
                i--;
            }
            node.keys.set(i + 1, key);
            node.n++;
        } else {
            // 내부 노드
            while (i >= 0 && key.compareTo(node.keys.get(i)) < 0) {
                i--;
            }
            i++;
            
            if (node.children.get(i).isFull()) {
                splitChild(node, i);
                if (key.compareTo(node.keys.get(i)) > 0) {
                    i++;
                }
            }
            
            insertNonFull(node.children.get(i), key);
        }
    }
    
    // 자식 분할
    private void splitChild(Node parent, int i) {
        Node fullChild = parent.children.get(i);
        Node newChild = new Node(fullChild.leaf);
        
        // 키 분할
        T midKey = fullChild.keys.get(MIN_DEGREE - 1);
        
        // 새 노드로 키 이동
        for (int j = 0; j < MIN_DEGREE - 1; j++) {
            newChild.keys.add(fullChild.keys.get(j + MIN_DEGREE));
        }
        
        // 자식 포인터 이동 (리프가 아닌 경우)
        if (!fullChild.leaf) {
            for (int j = 0; j < MIN_DEGREE; j++) {
                newChild.children.add(fullChild.children.get(j + MIN_DEGREE));
            }
        }
        
        // 기존 노드 조정
        fullChild.n = MIN_DEGREE - 1;
        newChild.n = MIN_DEGREE - 1;
        
        // 부모에 중간 키 삽입
        parent.keys.add(i, midKey);
        parent.children.add(i + 1, newChild);
        parent.n++;
        
        // 기존 노드에서 이동한 키와 자식 제거
        for (int j = fullChild.keys.size() - 1; j >= MIN_DEGREE - 1; j--) {
            fullChild.keys.remove(j);
        }
        
        if (!fullChild.leaf) {
            for (int j = fullChild.children.size() - 1; j >= MIN_DEGREE; j--) {
                fullChild.children.remove(j);
            }
        }
    }
}
```

## 10.8 Suffix Array와 Suffix Tree

### Suffix Array
문자열의 모든 접미사를 정렬한 배열입니다.

```java
public class SuffixArray {
    private String text;
    private int[] suffixArray;
    private int[] lcp; // Longest Common Prefix
    
    public SuffixArray(String text) {
        this.text = text;
        buildSuffixArray();
        buildLCPArray();
    }
    
    // O(n log n) Suffix Array 구축
    private void buildSuffixArray() {
        int n = text.length();
        Integer[] suffixes = new Integer[n];
        
        for (int i = 0; i < n; i++) {
            suffixes[i] = i;
        }
        
        // 접미사 정렬
        Arrays.sort(suffixes, (a, b) -> text.substring(a).compareTo(text.substring(b)));
        
        suffixArray = new int[n];
        for (int i = 0; i < n; i++) {
            suffixArray[i] = suffixes[i];
        }
    }
    
    // LCP 배열 구축
    private void buildLCPArray() {
        int n = text.length();
        lcp = new int[n];
        int[] rank = new int[n];
        
        // rank 배열 구축
        for (int i = 0; i < n; i++) {
            rank[suffixArray[i]] = i;
        }
        
        int h = 0;
        for (int i = 0; i < n; i++) {
            if (rank[i] > 0) {
                int j = suffixArray[rank[i] - 1];
                
                while (i + h < n && j + h < n && text.charAt(i + h) == text.charAt(j + h)) {
                    h++;
                }
                
                lcp[rank[i]] = h;
                
                if (h > 0) {
                    h--;
                }
            }
        }
    }
    
    // 패턴 검색
    public List<Integer> search(String pattern) {
        List<Integer> result = new ArrayList<>();
        int n = text.length();
        int m = pattern.length();
        
        // 이진 탐색으로 범위 찾기
        int left = 0, right = n - 1;
        
        // 하한 찾기
        while (left < right) {
            int mid = left + (right - left) / 2;
            String suffix = text.substring(suffixArray[mid]);
            
            if (suffix.compareTo(pattern) < 0) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        // 패턴과 일치하는 모든 위치 찾기
        while (left < n) {
            int pos = suffixArray[left];
            if (pos + m > n) break;
            
            String suffix = text.substring(pos, pos + m);
            if (!suffix.equals(pattern)) break;
            
            result.add(pos);
            left++;
        }
        
        return result;
    }
}

// Suffix Tree (Ukkonen's Algorithm)
public class SuffixTree {
    private class Node {
        Map<Character, Node> children;
        Node suffixLink;
        int start;
        int end;
        
        Node(int start, int end) {
            this.children = new HashMap<>();
            this.start = start;
            this.end = end;
        }
    }
    
    private String text;
    private Node root;
    
    public SuffixTree(String text) {
        this.text = text + "$"; // 종료 문자 추가
        buildSuffixTree();
    }
    
    private void buildSuffixTree() {
        root = new Node(-1, -1);
        Node activeNode = root;
        int activeEdge = 0;
        int activeLength = 0;
        int remainingSuffixCount = 0;
        int leafEnd = -1;
        
        for (int i = 0; i < text.length(); i++) {
            leafEnd++;
            remainingSuffixCount++;
            Node lastNewNode = null;
            
            while (remainingSuffixCount > 0) {
                if (activeLength == 0) {
                    activeEdge = i;
                }
                
                char ch = text.charAt(activeEdge);
                Node next = activeNode.children.get(ch);
                
                if (next == null) {
                    // 새 리프 노드 생성
                    activeNode.children.put(ch, new Node(i, leafEnd));
                    
                    if (lastNewNode != null) {
                        lastNewNode.suffixLink = activeNode;
                        lastNewNode = null;
                    }
                } else {
                    // 기존 경로를 따라감
                    int edgeLength = next.end - next.start + 1;
                    
                    if (activeLength >= edgeLength) {
                        activeEdge += edgeLength;
                        activeLength -= edgeLength;
                        activeNode = next;
                        continue;
                    }
                    
                    if (text.charAt(next.start + activeLength) == text.charAt(i)) {
                        activeLength++;
                        break;
                    }
                    
                    // 분할이 필요한 경우
                    Node split = new Node(next.start, next.start + activeLength - 1);
                    activeNode.children.put(ch, split);
                    
                    split.children.put(text.charAt(i), new Node(i, leafEnd));
                    next.start += activeLength;
                    split.children.put(text.charAt(next.start), next);
                    
                    if (lastNewNode != null) {
                        lastNewNode.suffixLink = split;
                    }
                    lastNewNode = split;
                }
                
                remainingSuffixCount--;
                
                if (activeNode == root && activeLength > 0) {
                    activeLength--;
                    activeEdge = i - remainingSuffixCount + 1;
                } else if (activeNode != root) {
                    activeNode = activeNode.suffixLink != null ? 
                                activeNode.suffixLink : root;
                }
            }
        }
    }
}
```

## 10.9 고급 자료구조 응용

### 영구적 자료구조 (Persistent Data Structure)

```java
public class PersistentSegmentTree {
    private class Node {
        int value;
        Node left, right;
        
        Node(int value) {
            this.value = value;
        }
        
        Node(Node left, Node right) {
            this.left = left;
            this.right = right;
            this.value = left.value + right.value;
        }
    }
    
    private Node[] roots;
    private int n;
    
    public PersistentSegmentTree(int[] arr) {
        n = arr.length;
        roots = new Node[n + 1];
        roots[0] = build(arr, 0, n - 1);
    }
    
    private Node build(int[] arr, int l, int r) {
        if (l == r) {
            return new Node(arr[l]);
        }
        
        int mid = (l + r) / 2;
        return new Node(build(arr, l, mid), build(arr, mid + 1, r));
    }
    
    // 새 버전 생성하며 업데이트
    public void update(int version, int idx, int value) {
        roots[version] = update(roots[version - 1], 0, n - 1, idx, value);
    }
    
    private Node update(Node node, int l, int r, int idx, int value) {
        if (l == r) {
            return new Node(value);
        }
        
        int mid = (l + r) / 2;
        if (idx <= mid) {
            return new Node(update(node.left, l, mid, idx, value), node.right);
        } else {
            return new Node(node.left, update(node.right, mid + 1, r, idx, value));
        }
    }
    
    // 특정 버전에서 쿼리
    public int query(int version, int ql, int qr) {
        return query(roots[version], 0, n - 1, ql, qr);
    }
    
    private int query(Node node, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) {
            return node.value;
        }
        
        if (qr < l || r < ql) {
            return 0;
        }
        
        int mid = (l + r) / 2;
        return query(node.left, l, mid, ql, qr) + 
               query(node.right, mid + 1, r, ql, qr);
    }
}
```

### K-D Tree

```java
public class KDTree {
    private class Node {
        double[] point;
        Node left, right;
        
        Node(double[] point) {
            this.point = point;
        }
    }
    
    private Node root;
    private int k; // 차원
    
    public KDTree(int k) {
        this.k = k;
    }
    
    public void insert(double[] point) {
        root = insert(root, point, 0);
    }
    
    private Node insert(Node node, double[] point, int depth) {
        if (node == null) {
            return new Node(point);
        }
        
        int cd = depth % k;
        
        if (point[cd] < node.point[cd]) {
            node.left = insert(node.left, point, depth + 1);
        } else {
            node.right = insert(node.right, point, depth + 1);
        }
        
        return node;
    }
    
    // 가장 가까운 이웃 찾기
    public double[] nearestNeighbor(double[] target) {
        return nearestNeighbor(root, target, 0, null);
    }
    
    private double[] nearestNeighbor(Node node, double[] target, int depth, double[] best) {
        if (node == null) {
            return best;
        }
        
        double distance = euclideanDistance(node.point, target);
        
        if (best == null || distance < euclideanDistance(best, target)) {
            best = node.point;
        }
        
        int cd = depth % k;
        Node goodSide = target[cd] < node.point[cd] ? node.left : node.right;
        Node badSide = target[cd] < node.point[cd] ? node.right : node.left;
        
        best = nearestNeighbor(goodSide, target, depth + 1, best);
        
        if (Math.abs(target[cd] - node.point[cd]) < euclideanDistance(best, target)) {
            best = nearestNeighbor(badSide, target, depth + 1, best);
        }
        
        return best;
    }
    
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < k; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
```

## 10.10 실습 문제

### 문제 1: 자동 완성 시스템
트라이를 사용하여 효율적인 자동 완성 시스템을 구현하세요.

### 문제 2: 구간 최댓값 업데이트
세그먼트 트리를 사용하여 구간 업데이트와 구간 최댓값 쿼리를 지원하는 자료구조를 구현하세요.

### 문제 3: 동적 연결성
Union-Find를 사용하여 동적으로 변하는 그래프의 연결성을 효율적으로 관리하세요.

### 문제 4: 2D 범위 합
2D 펜윅 트리를 사용하여 2차원 배열의 부분 직사각형 합을 구하세요.

## 10.11 요약

이 장에서는 특수한 목적을 위한 고급 자료구조들을 학습했습니다:

1. **트라이**: 문자열 처리에 최적화된 트리
2. **세그먼트 트리**: 구간 쿼리와 업데이트에 효율적
3. **펜윅 트리**: 누적 합과 구간 합 계산에 특화
4. **Union-Find**: 집합 연산과 연결성 확인
5. **스킵 리스트**: 확률적 균형 유지 리스트
6. **B-트리**: 디스크 기반 시스템에 최적화
7. **Suffix 구조**: 문자열 패턴 매칭에 특화

이러한 고급 자료구조들은 특정 문제를 해결하는 데 매우 효율적이며, 실제 시스템에서 널리 사용됩니다. 적절한 자료구조의 선택과 구현은 프로그램의 성능을 크게 향상시킬 수 있습니다.