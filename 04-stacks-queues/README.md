# Chapter 4: 스택과 큐 (Stacks and Queues)

## 4.1 스택 (Stack)

### 스택의 정의
스택은 LIFO(Last In First Out) 원칙을 따르는 선형 자료구조입니다. 가장 마지막에 삽입된 요소가 가장 먼저 제거됩니다.

### 스택의 주요 연산
- **push(item)**: 스택의 맨 위에 요소 추가 - O(1)
- **pop()**: 스택의 맨 위 요소 제거 및 반환 - O(1)
- **peek()/top()**: 스택의 맨 위 요소 확인 - O(1)
- **isEmpty()**: 스택이 비어있는지 확인 - O(1)
- **size()**: 스택의 크기 반환 - O(1)

### 스택의 활용
1. 함수 호출 스택
2. 괄호 매칭
3. 수식 계산 (후위 표기법)
4. 브라우저 뒤로가기
5. 실행 취소(Undo) 기능
6. DFS(깊이 우선 탐색)

### 배열 기반 스택 구현

```java
public class ArrayStack<T> {
    private T[] array;
    private int top;
    private int capacity;
    
    @SuppressWarnings("unchecked")
    public ArrayStack(int capacity) {
        this.capacity = capacity;
        this.array = (T[]) new Object[capacity];
        this.top = -1;
    }
    
    // 요소 추가
    public void push(T item) {
        if (isFull()) {
            throw new StackOverflowError("Stack is full");
        }
        array[++top] = item;
    }
    
    // 요소 제거 및 반환
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        T item = array[top];
        array[top--] = null; // 메모리 누수 방지
        return item;
    }
    
    // 맨 위 요소 확인
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return array[top];
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return top == -1;
    }
    
    // 가득 찼는지 확인
    public boolean isFull() {
        return top == capacity - 1;
    }
    
    // 크기 반환
    public int size() {
        return top + 1;
    }
    
    // 스택 비우기
    public void clear() {
        while (!isEmpty()) {
            pop();
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i <= top; i++) {
            sb.append(array[i]);
            if (i < top) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
```

### 연결 리스트 기반 스택 구현

```java
public class LinkedStack<T> {
    private Node<T> top;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public LinkedStack() {
        this.top = null;
        this.size = 0;
    }
    
    // 요소 추가
    public void push(T item) {
        Node<T> newNode = new Node<>(item);
        newNode.next = top;
        top = newNode;
        size++;
    }
    
    // 요소 제거 및 반환
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        T item = top.data;
        top = top.next;
        size--;
        return item;
    }
    
    // 맨 위 요소 확인
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return top.data;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return top == null;
    }
    
    // 크기 반환
    public int size() {
        return size;
    }
}
```

### 스택 활용 예제

```java
public class StackApplications {
    
    // 괄호 매칭
    public boolean isValidParentheses(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else if (c == ')' || c == '}' || c == ']') {
                if (stack.isEmpty()) {
                    return false;
                }
                
                char top = stack.pop();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) {
                    return false;
                }
            }
        }
        
        return stack.isEmpty();
    }
    
    // 후위 표기법 계산
    public int evaluatePostfix(String expression) {
        Stack<Integer> stack = new Stack<>();
        String[] tokens = expression.split(" ");
        
        for (String token : tokens) {
            if (isOperator(token)) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(calculate(a, b, token));
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        
        return stack.pop();
    }
    
    private boolean isOperator(String token) {
        return token.equals("+") || token.equals("-") || 
               token.equals("*") || token.equals("/");
    }
    
    private int calculate(int a, int b, String operator) {
        switch (operator) {
            case "+": return a + b;
            case "-": return a - b;
            case "*": return a * b;
            case "/": return a / b;
            default: throw new IllegalArgumentException("Invalid operator");
        }
    }
    
    // 중위 표기법을 후위 표기법으로 변환
    public String infixToPostfix(String expression) {
        Stack<Character> stack = new Stack<>();
        StringBuilder postfix = new StringBuilder();
        
        for (char c : expression.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                postfix.append(c);
            } else if (c == '(') {
                stack.push(c);
            } else if (c == ')') {
                while (!stack.isEmpty() && stack.peek() != '(') {
                    postfix.append(stack.pop());
                }
                stack.pop(); // '(' 제거
            } else if (isOperator(c)) {
                while (!stack.isEmpty() && 
                       precedence(c) <= precedence(stack.peek())) {
                    postfix.append(stack.pop());
                }
                stack.push(c);
            }
        }
        
        while (!stack.isEmpty()) {
            postfix.append(stack.pop());
        }
        
        return postfix.toString();
    }
    
    private boolean isOperator(char c) {
        return c == '+' || c == '-' || c == '*' || c == '/';
    }
    
    private int precedence(char operator) {
        switch (operator) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
                return 2;
            default:
                return -1;
        }
    }
    
    // 최소 스택 (O(1)로 최솟값 반환)
    public class MinStack {
        private Stack<Integer> stack;
        private Stack<Integer> minStack;
        
        public MinStack() {
            stack = new Stack<>();
            minStack = new Stack<>();
        }
        
        public void push(int val) {
            stack.push(val);
            if (minStack.isEmpty() || val <= minStack.peek()) {
                minStack.push(val);
            }
        }
        
        public void pop() {
            if (stack.pop().equals(minStack.peek())) {
                minStack.pop();
            }
        }
        
        public int top() {
            return stack.peek();
        }
        
        public int getMin() {
            return minStack.peek();
        }
    }
}
```

## 4.2 큐 (Queue)

### 큐의 정의
큐는 FIFO(First In First Out) 원칙을 따르는 선형 자료구조입니다. 가장 먼저 삽입된 요소가 가장 먼저 제거됩니다.

### 큐의 주요 연산
- **enqueue(item)**: 큐의 뒤쪽에 요소 추가 - O(1)
- **dequeue()**: 큐의 앞쪽 요소 제거 및 반환 - O(1)
- **front()/peek()**: 큐의 앞쪽 요소 확인 - O(1)
- **isEmpty()**: 큐가 비어있는지 확인 - O(1)
- **size()**: 큐의 크기 반환 - O(1)

### 큐의 활용
1. 프로세스 스케줄링
2. BFS(너비 우선 탐색)
3. 프린터 대기열
4. 캐시 구현
5. 메시지 큐

### 배열 기반 큐 구현 (원형 큐)

```java
public class CircularQueue<T> {
    private T[] array;
    private int front;
    private int rear;
    private int size;
    private int capacity;
    
    @SuppressWarnings("unchecked")
    public CircularQueue(int capacity) {
        this.capacity = capacity;
        this.array = (T[]) new Object[capacity];
        this.front = 0;
        this.rear = -1;
        this.size = 0;
    }
    
    // 요소 추가
    public void enqueue(T item) {
        if (isFull()) {
            throw new IllegalStateException("Queue is full");
        }
        rear = (rear + 1) % capacity;
        array[rear] = item;
        size++;
    }
    
    // 요소 제거 및 반환
    public T dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue is empty");
        }
        T item = array[front];
        array[front] = null; // 메모리 누수 방지
        front = (front + 1) % capacity;
        size--;
        return item;
    }
    
    // 앞쪽 요소 확인
    public T peek() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue is empty");
        }
        return array[front];
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return size == 0;
    }
    
    // 가득 찼는지 확인
    public boolean isFull() {
        return size == capacity;
    }
    
    // 크기 반환
    public int size() {
        return size;
    }
    
    @Override
    public String toString() {
        if (isEmpty()) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder("[");
        int index = front;
        for (int i = 0; i < size; i++) {
            sb.append(array[index]);
            if (i < size - 1) {
                sb.append(", ");
            }
            index = (index + 1) % capacity;
        }
        sb.append("]");
        return sb.toString();
    }
}
```

### 연결 리스트 기반 큐 구현

```java
public class LinkedQueue<T> {
    private Node<T> front;
    private Node<T> rear;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public LinkedQueue() {
        this.front = null;
        this.rear = null;
        this.size = 0;
    }
    
    // 요소 추가
    public void enqueue(T item) {
        Node<T> newNode = new Node<>(item);
        
        if (rear == null) {
            front = rear = newNode;
        } else {
            rear.next = newNode;
            rear = newNode;
        }
        size++;
    }
    
    // 요소 제거 및 반환
    public T dequeue() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue is empty");
        }
        
        T item = front.data;
        front = front.next;
        
        if (front == null) {
            rear = null;
        }
        size--;
        return item;
    }
    
    // 앞쪽 요소 확인
    public T peek() {
        if (isEmpty()) {
            throw new NoSuchElementException("Queue is empty");
        }
        return front.data;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return front == null;
    }
    
    // 크기 반환
    public int size() {
        return size;
    }
}
```

## 4.3 덱 (Deque - Double Ended Queue)

### 덱의 정의
덱은 양쪽 끝에서 삽입과 삭제가 모두 가능한 자료구조입니다.

### 덱의 주요 연산
- **addFirst(item)**: 앞쪽에 요소 추가
- **addLast(item)**: 뒤쪽에 요소 추가
- **removeFirst()**: 앞쪽 요소 제거
- **removeLast()**: 뒤쪽 요소 제거
- **peekFirst()**: 앞쪽 요소 확인
- **peekLast()**: 뒤쪽 요소 확인

```java
public class Deque<T> {
    private Node<T> head;
    private Node<T> tail;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> prev;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
        }
    }
    
    public Deque() {
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    
    // 앞쪽에 추가
    public void addFirst(T item) {
        Node<T> newNode = new Node<>(item);
        
        if (isEmpty()) {
            head = tail = newNode;
        } else {
            newNode.next = head;
            head.prev = newNode;
            head = newNode;
        }
        size++;
    }
    
    // 뒤쪽에 추가
    public void addLast(T item) {
        Node<T> newNode = new Node<>(item);
        
        if (isEmpty()) {
            head = tail = newNode;
        } else {
            tail.next = newNode;
            newNode.prev = tail;
            tail = newNode;
        }
        size++;
    }
    
    // 앞쪽에서 제거
    public T removeFirst() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        
        T item = head.data;
        head = head.next;
        
        if (head == null) {
            tail = null;
        } else {
            head.prev = null;
        }
        size--;
        return item;
    }
    
    // 뒤쪽에서 제거
    public T removeLast() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        
        T item = tail.data;
        tail = tail.prev;
        
        if (tail == null) {
            head = null;
        } else {
            tail.next = null;
        }
        size--;
        return item;
    }
    
    // 앞쪽 요소 확인
    public T peekFirst() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        return head.data;
    }
    
    // 뒤쪽 요소 확인
    public T peekLast() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        return tail.data;
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    public int size() {
        return size;
    }
}
```

## 4.4 우선순위 큐 (Priority Queue)

### 우선순위 큐의 정의
우선순위 큐는 각 요소가 우선순위를 가지며, 우선순위가 높은 요소가 먼저 제거되는 자료구조입니다.

### 힙 기반 우선순위 큐 구현

```java
public class PriorityQueue<T extends Comparable<T>> {
    private List<T> heap;
    
    public PriorityQueue() {
        heap = new ArrayList<>();
    }
    
    // 요소 추가
    public void offer(T item) {
        heap.add(item);
        heapifyUp(heap.size() - 1);
    }
    
    // 최우선 요소 제거 및 반환
    public T poll() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        
        T item = heap.get(0);
        T lastItem = heap.remove(heap.size() - 1);
        
        if (!isEmpty()) {
            heap.set(0, lastItem);
            heapifyDown(0);
        }
        
        return item;
    }
    
    // 최우선 요소 확인
    public T peek() {
        if (isEmpty()) {
            throw new NoSuchElementException();
        }
        return heap.get(0);
    }
    
    // 위로 재정렬
    private void heapifyUp(int index) {
        T item = heap.get(index);
        
        while (index > 0) {
            int parentIndex = (index - 1) / 2;
            T parent = heap.get(parentIndex);
            
            if (item.compareTo(parent) >= 0) {
                break;
            }
            
            heap.set(index, parent);
            index = parentIndex;
        }
        
        heap.set(index, item);
    }
    
    // 아래로 재정렬
    private void heapifyDown(int index) {
        T item = heap.get(index);
        int half = heap.size() / 2;
        
        while (index < half) {
            int leftChild = 2 * index + 1;
            int rightChild = leftChild + 1;
            int smallerChild = leftChild;
            
            if (rightChild < heap.size() && 
                heap.get(rightChild).compareTo(heap.get(leftChild)) < 0) {
                smallerChild = rightChild;
            }
            
            if (item.compareTo(heap.get(smallerChild)) <= 0) {
                break;
            }
            
            heap.set(index, heap.get(smallerChild));
            index = smallerChild;
        }
        
        heap.set(index, item);
    }
    
    public boolean isEmpty() {
        return heap.isEmpty();
    }
    
    public int size() {
        return heap.size();
    }
}
```

## 4.5 큐 활용 예제

```java
public class QueueApplications {
    
    // 슬라이딩 윈도우 최댓값
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || k <= 0) {
            return new int[0];
        }
        
        int n = nums.length;
        int[] result = new int[n - k + 1];
        int ri = 0;
        
        Deque<Integer> deque = new ArrayDeque<>();
        
        for (int i = 0; i < nums.length; i++) {
            // 윈도우 밖의 요소 제거
            while (!deque.isEmpty() && deque.peek() < i - k + 1) {
                deque.poll();
            }
            
            // 현재 요소보다 작은 요소들 제거
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }
            
            deque.offer(i);
            
            if (i >= k - 1) {
                result[ri++] = nums[deque.peek()];
            }
        }
        
        return result;
    }
    
    // 이진 트리 레벨 순회
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            result.add(currentLevel);
        }
        
        return result;
    }
    
    // 최근 호출 카운터
    public class RecentCounter {
        private Queue<Integer> queue;
        
        public RecentCounter() {
            queue = new LinkedList<>();
        }
        
        public int ping(int t) {
            queue.offer(t);
            
            // 3000ms 이전의 요청들 제거
            while (queue.peek() < t - 3000) {
                queue.poll();
            }
            
            return queue.size();
        }
    }
}
```

## 4.6 스택과 큐의 상호 구현

### 큐를 사용한 스택 구현

```java
public class QueueStack<T> {
    private Queue<T> queue1;
    private Queue<T> queue2;
    
    public QueueStack() {
        queue1 = new LinkedList<>();
        queue2 = new LinkedList<>();
    }
    
    // push - O(n)
    public void push(T item) {
        queue2.offer(item);
        
        while (!queue1.isEmpty()) {
            queue2.offer(queue1.poll());
        }
        
        Queue<T> temp = queue1;
        queue1 = queue2;
        queue2 = temp;
    }
    
    // pop - O(1)
    public T pop() {
        if (queue1.isEmpty()) {
            throw new EmptyStackException();
        }
        return queue1.poll();
    }
    
    public T top() {
        if (queue1.isEmpty()) {
            throw new EmptyStackException();
        }
        return queue1.peek();
    }
    
    public boolean empty() {
        return queue1.isEmpty();
    }
}
```

### 스택을 사용한 큐 구현

```java
public class StackQueue<T> {
    private Stack<T> inbox;
    private Stack<T> outbox;
    
    public StackQueue() {
        inbox = new Stack<>();
        outbox = new Stack<>();
    }
    
    // enqueue - O(1)
    public void enqueue(T item) {
        inbox.push(item);
    }
    
    // dequeue - amortized O(1)
    public T dequeue() {
        if (outbox.isEmpty()) {
            while (!inbox.isEmpty()) {
                outbox.push(inbox.pop());
            }
        }
        
        if (outbox.isEmpty()) {
            throw new NoSuchElementException();
        }
        
        return outbox.pop();
    }
    
    public T peek() {
        if (outbox.isEmpty()) {
            while (!inbox.isEmpty()) {
                outbox.push(inbox.pop());
            }
        }
        
        if (outbox.isEmpty()) {
            throw new NoSuchElementException();
        }
        
        return outbox.peek();
    }
    
    public boolean isEmpty() {
        return inbox.isEmpty() && outbox.isEmpty();
    }
}
```

## 4.7 실습 문제

### 문제 1: 유효한 괄호 문자열
주어진 문자열이 유효한 괄호 조합인지 확인하세요. '(', ')', '{', '}', '[', ']'를 포함합니다.

### 문제 2: 스택으로 큐 구현
두 개의 스택을 사용하여 큐를 구현하세요.

### 문제 3: 회전하는 큐
덱을 구현하고, 주어진 연산들을 수행하는 최소 비용을 계산하세요.

### 문제 4: 작업 스케줄러
우선순위 큐를 사용하여 CPU 작업 스케줄러를 구현하세요.

## 4.8 요약

이 장에서는 스택과 큐의 개념과 구현에 대해 학습했습니다:

1. **스택**은 LIFO 구조로 재귀, 백트래킹 문제에 유용
2. **큐**는 FIFO 구조로 순차적 처리가 필요한 경우에 적합
3. **덱**은 양방향 삽입/삭제가 가능한 유연한 구조
4. **우선순위 큐**는 힙을 기반으로 우선순위 처리

다음 장에서는 비선형 자료구조인 트리에 대해 알아보겠습니다.