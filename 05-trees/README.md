# Chapter 5: 트리 (Trees)

## 5.1 트리란?

### 정의
트리는 계층적 구조를 가진 비선형 자료구조입니다. 노드들이 부모-자식 관계로 연결되어 있으며, 사이클이 없는 연결 그래프입니다.

### 트리 용어
- **루트(Root)**: 부모가 없는 최상위 노드
- **노드(Node)**: 트리의 구성 요소
- **간선(Edge)**: 노드를 연결하는 선
- **부모(Parent)**: 바로 위 노드
- **자식(Child)**: 바로 아래 노드
- **형제(Sibling)**: 같은 부모를 가진 노드
- **리프(Leaf)**: 자식이 없는 노드
- **내부 노드**: 자식이 있는 노드
- **서브트리(Subtree)**: 노드와 그 자손들로 이루어진 트리
- **높이(Height)**: 루트에서 가장 깊은 리프까지의 거리
- **깊이(Depth)**: 루트에서 특정 노드까지의 거리
- **레벨(Level)**: 깊이 + 1
- **차수(Degree)**: 노드가 가진 자식의 수

### 트리의 특징
1. 하나의 루트 노드 존재
2. 모든 노드는 0개 이상의 자식 노드를 가짐
3. 사이클이 없음
4. 노드가 N개면 간선은 N-1개
5. 임의의 두 노드 간 경로는 유일

## 5.2 이진 트리 (Binary Tree)

### 정의
각 노드가 최대 2개의 자식을 가지는 트리입니다.

### 이진 트리의 종류
1. **포화 이진 트리(Full Binary Tree)**: 모든 레벨이 꽉 찬 트리
2. **완전 이진 트리(Complete Binary Tree)**: 마지막 레벨을 제외하고 모든 레벨이 완전히 채워지고, 마지막 레벨은 왼쪽부터 채워진 트리
3. **편향 트리(Skewed Tree)**: 모든 노드가 한쪽으로만 자식을 가진 트리
4. **균형 트리(Balanced Tree)**: 모든 리프의 깊이 차이가 1 이하인 트리

### 이진 트리 노드 구조

```java
public class TreeNode<T> {
    T data;
    TreeNode<T> left;
    TreeNode<T> right;
    
    public TreeNode(T data) {
        this.data = data;
        this.left = null;
        this.right = null;
    }
}
```

### 이진 트리 구현

```java
public class BinaryTree<T> {
    private TreeNode<T> root;
    
    private static class TreeNode<T> {
        T data;
        TreeNode<T> left;
        TreeNode<T> right;
        
        TreeNode(T data) {
            this.data = data;
            this.left = null;
            this.right = null;
        }
    }
    
    public BinaryTree() {
        this.root = null;
    }
    
    // 루트 설정
    public void setRoot(T data) {
        root = new TreeNode<>(data);
    }
    
    // 전위 순회 (Preorder: Root -> Left -> Right)
    public void preorderTraversal() {
        preorderHelper(root);
        System.out.println();
    }
    
    private void preorderHelper(TreeNode<T> node) {
        if (node != null) {
            System.out.print(node.data + " ");
            preorderHelper(node.left);
            preorderHelper(node.right);
        }
    }
    
    // 중위 순회 (Inorder: Left -> Root -> Right)
    public void inorderTraversal() {
        inorderHelper(root);
        System.out.println();
    }
    
    private void inorderHelper(TreeNode<T> node) {
        if (node != null) {
            inorderHelper(node.left);
            System.out.print(node.data + " ");
            inorderHelper(node.right);
        }
    }
    
    // 후위 순회 (Postorder: Left -> Right -> Root)
    public void postorderTraversal() {
        postorderHelper(root);
        System.out.println();
    }
    
    private void postorderHelper(TreeNode<T> node) {
        if (node != null) {
            postorderHelper(node.left);
            postorderHelper(node.right);
            System.out.print(node.data + " ");
        }
    }
    
    // 레벨 순회 (Level Order: BFS)
    public void levelOrderTraversal() {
        if (root == null) return;
        
        Queue<TreeNode<T>> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            TreeNode<T> node = queue.poll();
            System.out.print(node.data + " ");
            
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        System.out.println();
    }
    
    // 반복적 전위 순회
    public void iterativePreorder() {
        if (root == null) return;
        
        Stack<TreeNode<T>> stack = new Stack<>();
        stack.push(root);
        
        while (!stack.isEmpty()) {
            TreeNode<T> node = stack.pop();
            System.out.print(node.data + " ");
            
            // 오른쪽을 먼저 넣어야 왼쪽이 먼저 처리됨
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        System.out.println();
    }
    
    // 반복적 중위 순회
    public void iterativeInorder() {
        if (root == null) return;
        
        Stack<TreeNode<T>> stack = new Stack<>();
        TreeNode<T> current = root;
        
        while (current != null || !stack.isEmpty()) {
            // 왼쪽 끝까지 이동
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            
            current = stack.pop();
            System.out.print(current.data + " ");
            current = current.right;
        }
        System.out.println();
    }
    
    // 트리 높이
    public int height() {
        return heightHelper(root);
    }
    
    private int heightHelper(TreeNode<T> node) {
        if (node == null) {
            return -1;
        }
        
        int leftHeight = heightHelper(node.left);
        int rightHeight = heightHelper(node.right);
        
        return Math.max(leftHeight, rightHeight) + 1;
    }
    
    // 노드 개수
    public int size() {
        return sizeHelper(root);
    }
    
    private int sizeHelper(TreeNode<T> node) {
        if (node == null) {
            return 0;
        }
        
        return 1 + sizeHelper(node.left) + sizeHelper(node.right);
    }
    
    // 리프 노드 개수
    public int countLeaves() {
        return countLeavesHelper(root);
    }
    
    private int countLeavesHelper(TreeNode<T> node) {
        if (node == null) {
            return 0;
        }
        
        if (node.left == null && node.right == null) {
            return 1;
        }
        
        return countLeavesHelper(node.left) + countLeavesHelper(node.right);
    }
    
    // 트리가 균형잡혀 있는지 확인
    public boolean isBalanced() {
        return checkBalance(root) != -1;
    }
    
    private int checkBalance(TreeNode<T> node) {
        if (node == null) {
            return 0;
        }
        
        int leftHeight = checkBalance(node.left);
        if (leftHeight == -1) return -1;
        
        int rightHeight = checkBalance(node.right);
        if (rightHeight == -1) return -1;
        
        if (Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        }
        
        return Math.max(leftHeight, rightHeight) + 1;
    }
    
    // 동일한 트리인지 확인
    public boolean isSameTree(BinaryTree<T> other) {
        return isSameTreeHelper(this.root, other.root);
    }
    
    private boolean isSameTreeHelper(TreeNode<T> p, TreeNode<T> q) {
        if (p == null && q == null) {
            return true;
        }
        
        if (p == null || q == null) {
            return false;
        }
        
        return p.data.equals(q.data) &&
               isSameTreeHelper(p.left, q.left) &&
               isSameTreeHelper(p.right, q.right);
    }
    
    // 대칭 트리인지 확인
    public boolean isSymmetric() {
        return isSymmetricHelper(root, root);
    }
    
    private boolean isSymmetricHelper(TreeNode<T> left, TreeNode<T> right) {
        if (left == null && right == null) {
            return true;
        }
        
        if (left == null || right == null) {
            return false;
        }
        
        return left.data.equals(right.data) &&
               isSymmetricHelper(left.left, right.right) &&
               isSymmetricHelper(left.right, right.left);
    }
}
```

## 5.3 이진 탐색 트리 (Binary Search Tree)

### 정의
왼쪽 서브트리의 모든 노드가 부모보다 작고, 오른쪽 서브트리의 모든 노드가 부모보다 큰 이진 트리입니다.

### BST의 특징
- 중위 순회 시 정렬된 순서로 방문
- 검색, 삽입, 삭제가 평균 O(log n)
- 최악의 경우 O(n) (편향 트리)

### 이진 탐색 트리 구현

```java
public class BinarySearchTree<T extends Comparable<T>> {
    private TreeNode<T> root;
    
    private static class TreeNode<T> {
        T data;
        TreeNode<T> left;
        TreeNode<T> right;
        
        TreeNode(T data) {
            this.data = data;
            this.left = null;
            this.right = null;
        }
    }
    
    public BinarySearchTree() {
        this.root = null;
    }
    
    // 삽입
    public void insert(T data) {
        root = insertHelper(root, data);
    }
    
    private TreeNode<T> insertHelper(TreeNode<T> node, T data) {
        if (node == null) {
            return new TreeNode<>(data);
        }
        
        if (data.compareTo(node.data) < 0) {
            node.left = insertHelper(node.left, data);
        } else if (data.compareTo(node.data) > 0) {
            node.right = insertHelper(node.right, data);
        }
        
        return node;
    }
    
    // 검색
    public boolean search(T data) {
        return searchHelper(root, data);
    }
    
    private boolean searchHelper(TreeNode<T> node, T data) {
        if (node == null) {
            return false;
        }
        
        if (data.equals(node.data)) {
            return true;
        } else if (data.compareTo(node.data) < 0) {
            return searchHelper(node.left, data);
        } else {
            return searchHelper(node.right, data);
        }
    }
    
    // 반복적 검색
    public boolean iterativeSearch(T data) {
        TreeNode<T> current = root;
        
        while (current != null) {
            if (data.equals(current.data)) {
                return true;
            } else if (data.compareTo(current.data) < 0) {
                current = current.left;
            } else {
                current = current.right;
            }
        }
        
        return false;
    }
    
    // 삭제
    public void delete(T data) {
        root = deleteHelper(root, data);
    }
    
    private TreeNode<T> deleteHelper(TreeNode<T> node, T data) {
        if (node == null) {
            return null;
        }
        
        if (data.compareTo(node.data) < 0) {
            node.left = deleteHelper(node.left, data);
        } else if (data.compareTo(node.data) > 0) {
            node.right = deleteHelper(node.right, data);
        } else {
            // 삭제할 노드를 찾음
            
            // 경우 1: 리프 노드
            if (node.left == null && node.right == null) {
                return null;
            }
            
            // 경우 2: 자식이 하나인 노드
            if (node.left == null) {
                return node.right;
            }
            if (node.right == null) {
                return node.left;
            }
            
            // 경우 3: 자식이 둘인 노드
            // 오른쪽 서브트리의 최솟값을 찾아 교체
            TreeNode<T> minNode = findMin(node.right);
            node.data = minNode.data;
            node.right = deleteHelper(node.right, minNode.data);
        }
        
        return node;
    }
    
    // 최솟값 찾기
    public T findMin() {
        if (root == null) {
            throw new NoSuchElementException();
        }
        return findMin(root).data;
    }
    
    private TreeNode<T> findMin(TreeNode<T> node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }
    
    // 최댓값 찾기
    public T findMax() {
        if (root == null) {
            throw new NoSuchElementException();
        }
        return findMax(root).data;
    }
    
    private TreeNode<T> findMax(TreeNode<T> node) {
        while (node.right != null) {
            node = node.right;
        }
        return node;
    }
    
    // 유효한 BST인지 확인
    public boolean isValidBST() {
        return isValidBSTHelper(root, null, null);
    }
    
    private boolean isValidBSTHelper(TreeNode<T> node, T min, T max) {
        if (node == null) {
            return true;
        }
        
        if ((min != null && node.data.compareTo(min) <= 0) ||
            (max != null && node.data.compareTo(max) >= 0)) {
            return false;
        }
        
        return isValidBSTHelper(node.left, min, node.data) &&
               isValidBSTHelper(node.right, node.data, max);
    }
    
    // k번째 작은 요소 찾기
    public T kthSmallest(int k) {
        List<T> inorder = new ArrayList<>();
        inorderTraversal(root, inorder);
        
        if (k <= 0 || k > inorder.size()) {
            throw new IndexOutOfBoundsException();
        }
        
        return inorder.get(k - 1);
    }
    
    private void inorderTraversal(TreeNode<T> node, List<T> list) {
        if (node != null) {
            inorderTraversal(node.left, list);
            list.add(node.data);
            inorderTraversal(node.right, list);
        }
    }
    
    // 두 노드의 최소 공통 조상 찾기
    public T lowestCommonAncestor(T p, T q) {
        return lcaHelper(root, p, q).data;
    }
    
    private TreeNode<T> lcaHelper(TreeNode<T> node, T p, T q) {
        if (node == null) {
            return null;
        }
        
        // p와 q가 모두 왼쪽에 있는 경우
        if (p.compareTo(node.data) < 0 && q.compareTo(node.data) < 0) {
            return lcaHelper(node.left, p, q);
        }
        
        // p와 q가 모두 오른쪽에 있는 경우
        if (p.compareTo(node.data) > 0 && q.compareTo(node.data) > 0) {
            return lcaHelper(node.right, p, q);
        }
        
        // p와 q가 현재 노드의 양쪽에 있는 경우
        return node;
    }
}
```

## 5.4 AVL 트리

### 정의
AVL 트리는 자가 균형 이진 탐색 트리로, 모든 노드에서 왼쪽과 오른쪽 서브트리의 높이 차이가 1 이하입니다.

### AVL 트리의 회전
1. **LL 회전**: 왼쪽-왼쪽 경우
2. **RR 회전**: 오른쪽-오른쪽 경우
3. **LR 회전**: 왼쪽-오른쪽 경우
4. **RL 회전**: 오른쪽-왼쪽 경우

### AVL 트리 구현

```java
public class AVLTree<T extends Comparable<T>> {
    private AVLNode<T> root;
    
    private static class AVLNode<T> {
        T data;
        AVLNode<T> left;
        AVLNode<T> right;
        int height;
        
        AVLNode(T data) {
            this.data = data;
            this.left = null;
            this.right = null;
            this.height = 0;
        }
    }
    
    public AVLTree() {
        this.root = null;
    }
    
    // 높이 얻기
    private int height(AVLNode<T> node) {
        return node == null ? -1 : node.height;
    }
    
    // 균형 인수 계산
    private int getBalance(AVLNode<T> node) {
        return node == null ? 0 : height(node.left) - height(node.right);
    }
    
    // 높이 업데이트
    private void updateHeight(AVLNode<T> node) {
        node.height = Math.max(height(node.left), height(node.right)) + 1;
    }
    
    // 오른쪽 회전 (LL)
    private AVLNode<T> rotateRight(AVLNode<T> y) {
        AVLNode<T> x = y.left;
        AVLNode<T> T2 = x.right;
        
        x.right = y;
        y.left = T2;
        
        updateHeight(y);
        updateHeight(x);
        
        return x;
    }
    
    // 왼쪽 회전 (RR)
    private AVLNode<T> rotateLeft(AVLNode<T> x) {
        AVLNode<T> y = x.right;
        AVLNode<T> T2 = y.left;
        
        y.left = x;
        x.right = T2;
        
        updateHeight(x);
        updateHeight(y);
        
        return y;
    }
    
    // 삽입
    public void insert(T data) {
        root = insertHelper(root, data);
    }
    
    private AVLNode<T> insertHelper(AVLNode<T> node, T data) {
        // 1. 일반 BST 삽입
        if (node == null) {
            return new AVLNode<>(data);
        }
        
        if (data.compareTo(node.data) < 0) {
            node.left = insertHelper(node.left, data);
        } else if (data.compareTo(node.data) > 0) {
            node.right = insertHelper(node.right, data);
        } else {
            return node; // 중복 허용하지 않음
        }
        
        // 2. 높이 업데이트
        updateHeight(node);
        
        // 3. 균형 인수 확인
        int balance = getBalance(node);
        
        // 4. 불균형 해결
        // LL 케이스
        if (balance > 1 && data.compareTo(node.left.data) < 0) {
            return rotateRight(node);
        }
        
        // RR 케이스
        if (balance < -1 && data.compareTo(node.right.data) > 0) {
            return rotateLeft(node);
        }
        
        // LR 케이스
        if (balance > 1 && data.compareTo(node.left.data) > 0) {
            node.left = rotateLeft(node.left);
            return rotateRight(node);
        }
        
        // RL 케이스
        if (balance < -1 && data.compareTo(node.right.data) < 0) {
            node.right = rotateRight(node.right);
            return rotateLeft(node);
        }
        
        return node;
    }
    
    // 삭제
    public void delete(T data) {
        root = deleteHelper(root, data);
    }
    
    private AVLNode<T> deleteHelper(AVLNode<T> node, T data) {
        // 1. 일반 BST 삭제
        if (node == null) {
            return null;
        }
        
        if (data.compareTo(node.data) < 0) {
            node.left = deleteHelper(node.left, data);
        } else if (data.compareTo(node.data) > 0) {
            node.right = deleteHelper(node.right, data);
        } else {
            // 삭제할 노드 찾음
            if (node.left == null) {
                return node.right;
            } else if (node.right == null) {
                return node.left;
            }
            
            // 두 자식이 있는 경우
            AVLNode<T> minNode = findMin(node.right);
            node.data = minNode.data;
            node.right = deleteHelper(node.right, minNode.data);
        }
        
        // 2. 높이 업데이트
        updateHeight(node);
        
        // 3. 균형 확인 및 회전
        int balance = getBalance(node);
        
        // LL 케이스
        if (balance > 1 && getBalance(node.left) >= 0) {
            return rotateRight(node);
        }
        
        // LR 케이스
        if (balance > 1 && getBalance(node.left) < 0) {
            node.left = rotateLeft(node.left);
            return rotateRight(node);
        }
        
        // RR 케이스
        if (balance < -1 && getBalance(node.right) <= 0) {
            return rotateLeft(node);
        }
        
        // RL 케이스
        if (balance < -1 && getBalance(node.right) > 0) {
            node.right = rotateRight(node.right);
            return rotateLeft(node);
        }
        
        return node;
    }
    
    private AVLNode<T> findMin(AVLNode<T> node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }
}
```

## 5.5 힙 (Heap)

### 정의
힙은 완전 이진 트리로, 부모 노드가 자식 노드보다 항상 크거나(최대 힙) 작은(최소 힙) 특성을 가집니다.

### 힙의 특징
- 완전 이진 트리
- 배열로 효율적 구현 가능
- 삽입/삭제: O(log n)
- 최댓값/최솟값 접근: O(1)

### 최대 힙 구현

```java
public class MaxHeap<T extends Comparable<T>> {
    private List<T> heap;
    
    public MaxHeap() {
        heap = new ArrayList<>();
    }
    
    // 부모 인덱스
    private int parent(int i) {
        return (i - 1) / 2;
    }
    
    // 왼쪽 자식 인덱스
    private int leftChild(int i) {
        return 2 * i + 1;
    }
    
    // 오른쪽 자식 인덱스
    private int rightChild(int i) {
        return 2 * i + 2;
    }
    
    // 두 요소 교환
    private void swap(int i, int j) {
        T temp = heap.get(i);
        heap.set(i, heap.get(j));
        heap.set(j, temp);
    }
    
    // 삽입
    public void insert(T data) {
        heap.add(data);
        heapifyUp(heap.size() - 1);
    }
    
    // 위로 재정렬
    private void heapifyUp(int i) {
        while (i > 0 && heap.get(parent(i)).compareTo(heap.get(i)) < 0) {
            swap(i, parent(i));
            i = parent(i);
        }
    }
    
    // 최댓값 추출
    public T extractMax() {
        if (heap.isEmpty()) {
            throw new NoSuchElementException();
        }
        
        T max = heap.get(0);
        T last = heap.remove(heap.size() - 1);
        
        if (!heap.isEmpty()) {
            heap.set(0, last);
            heapifyDown(0);
        }
        
        return max;
    }
    
    // 아래로 재정렬
    private void heapifyDown(int i) {
        int maxIndex = i;
        int left = leftChild(i);
        int right = rightChild(i);
        
        if (left < heap.size() && 
            heap.get(left).compareTo(heap.get(maxIndex)) > 0) {
            maxIndex = left;
        }
        
        if (right < heap.size() && 
            heap.get(right).compareTo(heap.get(maxIndex)) > 0) {
            maxIndex = right;
        }
        
        if (i != maxIndex) {
            swap(i, maxIndex);
            heapifyDown(maxIndex);
        }
    }
    
    // 최댓값 확인
    public T peek() {
        if (heap.isEmpty()) {
            throw new NoSuchElementException();
        }
        return heap.get(0);
    }
    
    // 크기
    public int size() {
        return heap.size();
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return heap.isEmpty();
    }
    
    // 힙 정렬
    public static <T extends Comparable<T>> void heapSort(T[] array) {
        int n = array.length;
        
        // 힙 구성 (Build Heap)
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(array, n, i);
        }
        
        // 하나씩 추출
        for (int i = n - 1; i > 0; i--) {
            // 루트(최댓값)를 끝으로 이동
            T temp = array[0];
            array[0] = array[i];
            array[i] = temp;
            
            // 힙 재정렬
            heapify(array, i, 0);
        }
    }
    
    private static <T extends Comparable<T>> void heapify(T[] array, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && array[left].compareTo(array[largest]) > 0) {
            largest = left;
        }
        
        if (right < n && array[right].compareTo(array[largest]) > 0) {
            largest = right;
        }
        
        if (largest != i) {
            T swap = array[i];
            array[i] = array[largest];
            array[largest] = swap;
            
            heapify(array, n, largest);
        }
    }
}
```

## 5.6 트리 문제 풀이

### 트리 관련 알고리즘

```java
public class TreeProblems {
    
    // 경로 합계
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        
        if (root.left == null && root.right == null) {
            return root.val == targetSum;
        }
        
        return hasPathSum(root.left, targetSum - root.val) ||
               hasPathSum(root.right, targetSum - root.val);
    }
    
    // 모든 경로 찾기
    public List<List<Integer>> allPaths(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> currentPath = new ArrayList<>();
        allPathsHelper(root, currentPath, result);
        return result;
    }
    
    private void allPathsHelper(TreeNode node, List<Integer> path, 
                               List<List<Integer>> result) {
        if (node == null) {
            return;
        }
        
        path.add(node.val);
        
        if (node.left == null && node.right == null) {
            result.add(new ArrayList<>(path));
        } else {
            allPathsHelper(node.left, path, result);
            allPathsHelper(node.right, path, result);
        }
        
        path.remove(path.size() - 1);
    }
    
    // 트리 직렬화
    public String serialize(TreeNode root) {
        if (root == null) {
            return "null";
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        StringBuilder sb = new StringBuilder();
        
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            
            if (node == null) {
                sb.append("null,");
            } else {
                sb.append(node.val).append(",");
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }
        
        return sb.toString();
    }
    
    // 트리 역직렬화
    public TreeNode deserialize(String data) {
        if (data.equals("null")) {
            return null;
        }
        
        String[] values = data.split(",");
        TreeNode root = new TreeNode(Integer.parseInt(values[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        int i = 1;
        while (!queue.isEmpty() && i < values.length) {
            TreeNode node = queue.poll();
            
            if (!values[i].equals("null")) {
                node.left = new TreeNode(Integer.parseInt(values[i]));
                queue.offer(node.left);
            }
            i++;
            
            if (i < values.length && !values[i].equals("null")) {
                node.right = new TreeNode(Integer.parseInt(values[i]));
                queue.offer(node.right);
            }
            i++;
        }
        
        return root;
    }
    
    // 지그재그 레벨 순회
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean leftToRight = true;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> level = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (leftToRight) {
                    level.add(node.val);
                } else {
                    level.add(0, node.val);
                }
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            result.add(level);
            leftToRight = !leftToRight;
        }
        
        return result;
    }
}
```

## 5.7 실습 문제

### 문제 1: 트리의 지름
트리의 지름(두 노드 간 가장 긴 경로)을 구하세요.

### 문제 2: 트리 뒤집기
이진 트리를 좌우 대칭으로 뒤집으세요.

### 문제 3: 서브트리 확인
한 트리가 다른 트리의 서브트리인지 확인하세요.

### 문제 4: 트라이 구현
문자열 검색을 위한 트라이 자료구조를 구현하세요.

## 5.8 요약

이 장에서는 트리 자료구조의 다양한 형태와 활용에 대해 학습했습니다:

1. **이진 트리**: 각 노드가 최대 2개의 자식을 가지는 기본 트리
2. **이진 탐색 트리**: 정렬된 데이터를 효율적으로 관리
3. **AVL 트리**: 자가 균형 트리로 O(log n) 보장
4. **힙**: 우선순위 큐 구현에 적합한 완전 이진 트리

트리는 계층적 데이터를 표현하고 효율적인 검색을 제공하는 중요한 자료구조입니다. 다음 장에서는 더 복잡한 비선형 구조인 그래프에 대해 알아보겠습니다.