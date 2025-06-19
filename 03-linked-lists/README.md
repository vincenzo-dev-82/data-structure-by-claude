# Chapter 3: 연결 리스트 (Linked Lists)

## 3.1 연결 리스트 개요

### 연결 리스트란?
연결 리스트는 노드(Node)들이 포인터로 연결된 선형 자료구조입니다. 각 노드는 데이터와 다음 노드를 가리키는 참조(포인터)를 포함합니다.

### 배열 vs 연결 리스트

| 특징 | 배열 | 연결 리스트 |
|------|------|------------|
| 메모리 할당 | 연속적 | 비연속적 |
| 크기 | 고정 | 동적 |
| 접근 시간 | O(1) | O(n) |
| 삽입/삭제 (처음) | O(n) | O(1) |
| 삽입/삭제 (중간) | O(n) | O(n) |
| 메모리 오버헤드 | 낮음 | 높음 (포인터 저장) |
| 캐시 효율성 | 높음 | 낮음 |

### 연결 리스트의 장단점

**장점:**
- 동적 크기 조절
- 효율적인 삽입/삭제 (처음과 끝)
- 메모리 효율적 사용 (필요한 만큼만 할당)

**단점:**
- 랜덤 접근 불가
- 포인터를 위한 추가 메모리 필요
- 캐시 지역성이 낮음
- 역방향 순회 어려움 (단일 연결 리스트)

## 3.2 단일 연결 리스트 (Singly Linked List)

### 기본 구조
각 노드가 다음 노드만을 가리키는 가장 간단한 형태의 연결 리스트입니다.

```java
public class SinglyLinkedList<T> {
    private Node<T> head;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public SinglyLinkedList() {
        this.head = null;
        this.size = 0;
    }
    
    // 리스트의 처음에 노드 추가 - O(1)
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        newNode.next = head;
        head = newNode;
        size++;
    }
    
    // 리스트의 끝에 노드 추가 - O(n)
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            head = newNode;
        } else {
            Node<T> current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
        size++;
    }
    
    // 특정 위치에 노드 추가 - O(n)
    public void add(int index, T data) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == 0) {
            addFirst(data);
            return;
        }
        
        Node<T> newNode = new Node<>(data);
        Node<T> current = head;
        
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        
        newNode.next = current.next;
        current.next = newNode;
        size++;
    }
    
    // 처음 노드 제거 - O(1)
    public T removeFirst() {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        T data = head.data;
        head = head.next;
        size--;
        return data;
    }
    
    // 마지막 노드 제거 - O(n)
    public T removeLast() {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        if (head.next == null) {
            T data = head.data;
            head = null;
            size--;
            return data;
        }
        
        Node<T> current = head;
        while (current.next.next != null) {
            current = current.next;
        }
        
        T data = current.next.data;
        current.next = null;
        size--;
        return data;
    }
    
    // 특정 위치의 노드 제거 - O(n)
    public T remove(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == 0) {
            return removeFirst();
        }
        
        Node<T> current = head;
        for (int i = 0; i < index - 1; i++) {
            current = current.next;
        }
        
        T data = current.next.data;
        current.next = current.next.next;
        size--;
        return data;
    }
    
    // 특정 값을 가진 첫 번째 노드 제거 - O(n)
    public boolean removeValue(T value) {
        if (head == null) {
            return false;
        }
        
        if (head.data.equals(value)) {
            head = head.next;
            size--;
            return true;
        }
        
        Node<T> current = head;
        while (current.next != null) {
            if (current.next.data.equals(value)) {
                current.next = current.next.next;
                size--;
                return true;
            }
            current = current.next;
        }
        
        return false;
    }
    
    // 특정 위치의 값 가져오기 - O(n)
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        Node<T> current = head;
        for (int i = 0; i < index; i++) {
            current = current.next;
        }
        
        return current.data;
    }
    
    // 값 검색 - O(n)
    public boolean contains(T value) {
        Node<T> current = head;
        while (current != null) {
            if (current.data.equals(value)) {
                return true;
            }
            current = current.next;
        }
        return false;
    }
    
    // 리스트 뒤집기 - O(n)
    public void reverse() {
        Node<T> prev = null;
        Node<T> current = head;
        Node<T> next = null;
        
        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        head = prev;
    }
    
    // 중간 노드 찾기 - O(n)
    public T getMiddle() {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        Node<T> slow = head;
        Node<T> fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return slow.data;
    }
    
    // 순환 검사 - O(n)
    public boolean hasCycle() {
        if (head == null) {
            return false;
        }
        
        Node<T> slow = head;
        Node<T> fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            
            if (slow == fast) {
                return true;
            }
        }
        
        return false;
    }
    
    // 크기 반환
    public int size() {
        return size;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return head == null;
    }
    
    // 리스트 출력
    @Override
    public String toString() {
        if (head == null) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder("[");
        Node<T> current = head;
        
        while (current != null) {
            sb.append(current.data);
            if (current.next != null) {
                sb.append(" -> ");
            }
            current = current.next;
        }
        
        sb.append("]");
        return sb.toString();
    }
}
```

## 3.3 이중 연결 리스트 (Doubly Linked List)

### 기본 구조
각 노드가 이전 노드와 다음 노드를 모두 가리키는 연결 리스트입니다.

```java
public class DoublyLinkedList<T> {
    private Node<T> head;
    private Node<T> tail;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> prev;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.prev = null;
            this.next = null;
        }
    }
    
    public DoublyLinkedList() {
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    
    // 리스트의 처음에 노드 추가 - O(1)
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            head = tail = newNode;
        } else {
            newNode.next = head;
            head.prev = newNode;
            head = newNode;
        }
        size++;
    }
    
    // 리스트의 끝에 노드 추가 - O(1)
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (tail == null) {
            head = tail = newNode;
        } else {
            tail.next = newNode;
            newNode.prev = tail;
            tail = newNode;
        }
        size++;
    }
    
    // 특정 위치에 노드 추가 - O(n)
    public void add(int index, T data) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == 0) {
            addFirst(data);
        } else if (index == size) {
            addLast(data);
        } else {
            Node<T> newNode = new Node<>(data);
            Node<T> current = getNodeAt(index);
            
            newNode.prev = current.prev;
            newNode.next = current;
            current.prev.next = newNode;
            current.prev = newNode;
            size++;
        }
    }
    
    // 처음 노드 제거 - O(1)
    public T removeFirst() {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        T data = head.data;
        
        if (head == tail) {
            head = tail = null;
        } else {
            head = head.next;
            head.prev = null;
        }
        size--;
        return data;
    }
    
    // 마지막 노드 제거 - O(1)
    public T removeLast() {
        if (tail == null) {
            throw new NoSuchElementException();
        }
        
        T data = tail.data;
        
        if (head == tail) {
            head = tail = null;
        } else {
            tail = tail.prev;
            tail.next = null;
        }
        size--;
        return data;
    }
    
    // 특정 위치의 노드 제거 - O(n)
    public T remove(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == 0) {
            return removeFirst();
        } else if (index == size - 1) {
            return removeLast();
        } else {
            Node<T> current = getNodeAt(index);
            T data = current.data;
            
            current.prev.next = current.next;
            current.next.prev = current.prev;
            size--;
            return data;
        }
    }
    
    // 특정 위치의 노드 가져오기 (최적화) - O(n)
    private Node<T> getNodeAt(int index) {
        Node<T> current;
        
        // 인덱스가 전체의 절반보다 작으면 앞에서부터 탐색
        if (index < size / 2) {
            current = head;
            for (int i = 0; i < index; i++) {
                current = current.next;
            }
        } else { // 뒤에서부터 탐색
            current = tail;
            for (int i = size - 1; i > index; i--) {
                current = current.prev;
            }
        }
        
        return current;
    }
    
    // 특정 위치의 값 가져오기 - O(n)
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        return getNodeAt(index).data;
    }
    
    // 값 설정 - O(n)
    public T set(int index, T data) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        Node<T> node = getNodeAt(index);
        T oldData = node.data;
        node.data = data;
        return oldData;
    }
    
    // 역방향 순회
    public void reverseTraverse() {
        Node<T> current = tail;
        while (current != null) {
            System.out.print(current.data + " ");
            current = current.prev;
        }
        System.out.println();
    }
    
    // 리스트를 배열로 변환
    @SuppressWarnings("unchecked")
    public T[] toArray() {
        T[] array = (T[]) new Object[size];
        Node<T> current = head;
        
        for (int i = 0; i < size; i++) {
            array[i] = current.data;
            current = current.next;
        }
        
        return array;
    }
    
    // 크기 반환
    public int size() {
        return size;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return size == 0;
    }
    
    // 리스트 출력
    @Override
    public String toString() {
        if (head == null) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder("[");
        Node<T> current = head;
        
        while (current != null) {
            sb.append(current.data);
            if (current.next != null) {
                sb.append(" <-> ");
            }
            current = current.next;
        }
        
        sb.append("]");
        return sb.toString();
    }
}
```

## 3.4 원형 연결 리스트 (Circular Linked List)

### 기본 구조
마지막 노드가 첫 번째 노드를 가리키는 연결 리스트입니다.

```java
public class CircularLinkedList<T> {
    private Node<T> head;
    private Node<T> tail;
    private int size;
    
    private static class Node<T> {
        T data;
        Node<T> next;
        
        Node(T data) {
            this.data = data;
            this.next = null;
        }
    }
    
    public CircularLinkedList() {
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    
    // 리스트의 처음에 노드 추가 - O(1)
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            head = tail = newNode;
            newNode.next = head;
        } else {
            newNode.next = head;
            tail.next = newNode;
            head = newNode;
        }
        size++;
    }
    
    // 리스트의 끝에 노드 추가 - O(1)
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (head == null) {
            head = tail = newNode;
            newNode.next = head;
        } else {
            tail.next = newNode;
            newNode.next = head;
            tail = newNode;
        }
        size++;
    }
    
    // 처음 노드 제거 - O(1)
    public T removeFirst() {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        T data = head.data;
        
        if (head == tail) {
            head = tail = null;
        } else {
            head = head.next;
            tail.next = head;
        }
        size--;
        return data;
    }
    
    // 특정 값을 가진 노드 제거 - O(n)
    public boolean removeValue(T value) {
        if (head == null) {
            return false;
        }
        
        if (head.data.equals(value)) {
            removeFirst();
            return true;
        }
        
        Node<T> current = head;
        while (current.next != head) {
            if (current.next.data.equals(value)) {
                if (current.next == tail) {
                    tail = current;
                }
                current.next = current.next.next;
                size--;
                return true;
            }
            current = current.next;
        }
        
        return false;
    }
    
    // 순회 (지정된 횟수만큼) - O(n)
    public void traverse(int times) {
        if (head == null) {
            return;
        }
        
        Node<T> current = head;
        int count = 0;
        
        while (count < times * size) {
            System.out.print(current.data + " ");
            current = current.next;
            count++;
        }
        System.out.println();
    }
    
    // 요세푸스 문제 해결
    public T josephus(int k) {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        Node<T> current = head;
        Node<T> prev = tail;
        
        while (size > 1) {
            // k-1번 이동
            for (int i = 0; i < k - 1; i++) {
                prev = current;
                current = current.next;
            }
            
            // 현재 노드 제거
            prev.next = current.next;
            if (current == head) {
                head = current.next;
            }
            if (current == tail) {
                tail = prev;
            }
            
            current = current.next;
            size--;
        }
        
        return head.data;
    }
    
    // 리스트 분할 (두 개의 원형 리스트로)
    public CircularLinkedList<T> split() {
        if (size < 2) {
            return new CircularLinkedList<>();
        }
        
        // 중간 지점 찾기
        Node<T> slow = head;
        Node<T> fast = head;
        Node<T> prevSlow = null;
        
        while (fast.next != head && fast.next.next != head) {
            prevSlow = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // 두 번째 리스트 생성
        CircularLinkedList<T> secondList = new CircularLinkedList<>();
        secondList.head = slow.next;
        secondList.tail = tail;
        secondList.tail.next = secondList.head;
        secondList.size = size / 2;
        
        // 첫 번째 리스트 수정
        slow.next = head;
        tail = slow;
        size = (size + 1) / 2;
        
        return secondList;
    }
    
    // 크기 반환
    public int size() {
        return size;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return head == null;
    }
    
    // 리스트 출력
    @Override
    public String toString() {
        if (head == null) {
            return "[]";
        }
        
        StringBuilder sb = new StringBuilder("[");
        Node<T> current = head;
        
        do {
            sb.append(current.data);
            if (current.next != head) {
                sb.append(" -> ");
            }
            current = current.next;
        } while (current != head);
        
        sb.append(" -> (head)]");
        return sb.toString();
    }
}
```

## 3.5 연결 리스트 활용 예제

### LRU 캐시 구현
```java
public class LRUCache {
    private class Node {
        int key;
        int value;
        Node prev;
        Node next;
        
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
    
    private int capacity;
    private Map<Integer, Node> map;
    private Node head;
    private Node tail;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new HashMap<>();
        
        // 더미 노드 사용
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        
        Node node = map.get(key);
        removeNode(node);
        addToHead(node);
        
        return node.value;
    }
    
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.value = value;
            removeNode(node);
            addToHead(node);
        } else {
            Node newNode = new Node(key, value);
            map.put(key, newNode);
            addToHead(newNode);
            
            if (map.size() > capacity) {
                Node tailNode = tail.prev;
                removeNode(tailNode);
                map.remove(tailNode.key);
            }
        }
    }
    
    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void addToHead(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }
}
```

### 두 정렬된 리스트 병합
```java
public class ListMerger {
    
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 더미 노드 사용
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                current.next = l1;
                l1 = l1.next;
            } else {
                current.next = l2;
                l2 = l2.next;
            }
            current = current.next;
        }
        
        // 남은 노드 연결
        if (l1 != null) {
            current.next = l1;
        } else {
            current.next = l2;
        }
        
        return dummy.next;
    }
    
    // k개의 정렬된 리스트 병합
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        
        // 각 리스트의 첫 노드를 우선순위 큐에 추가
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            current.next = node;
            current = current.next;
            
            if (node.next != null) {
                pq.offer(node.next);
            }
        }
        
        return dummy.next;
    }
}
```

### 팰린드롬 연결 리스트 확인
```java
public class PalindromeChecker {
    
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        
        // 1. 중간 지점 찾기
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // 2. 후반부 뒤집기
        ListNode reversedSecondHalf = reverse(slow);
        
        // 3. 비교
        ListNode p1 = head;
        ListNode p2 = reversedSecondHalf;
        
        while (p2 != null) {
            if (p1.val != p2.val) {
                return false;
            }
            p1 = p1.next;
            p2 = p2.next;
        }
        
        return true;
    }
    
    private ListNode reverse(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        
        while (current != null) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
}
```

## 3.6 연결 리스트 알고리즘 패턴

### Two Pointers (투 포인터)
```java
public class TwoPointerTechniques {
    
    // 리스트의 중간 노드 찾기
    public ListNode findMiddle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return slow;
    }
    
    // 끝에서 n번째 노드 찾기
    public ListNode findNthFromEnd(ListNode head, int n) {
        ListNode fast = head;
        ListNode slow = head;
        
        // fast를 n만큼 먼저 이동
        for (int i = 0; i < n; i++) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
        }
        
        // 함께 이동
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        
        return slow;
    }
    
    // 사이클 시작점 찾기
    public ListNode detectCycleStart(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        
        // 사이클 존재 확인
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            
            if (slow == fast) {
                break;
            }
        }
        
        if (fast == null || fast.next == null) {
            return null;  // 사이클 없음
        }
        
        // 사이클 시작점 찾기
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        
        return slow;
    }
}
```

### 더미 노드 활용
```java
public class DummyNodeTechniques {
    
    // 중복 제거
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        while (head != null) {
            if (head.next != null && head.val == head.next.val) {
                int val = head.val;
                while (head != null && head.val == val) {
                    head = head.next;
                }
                prev.next = head;
            } else {
                prev = head;
                head = head.next;
            }
        }
        
        return dummy.next;
    }
    
    // 파티션
    public ListNode partition(ListNode head, int x) {
        ListNode beforeDummy = new ListNode(0);
        ListNode before = beforeDummy;
        ListNode afterDummy = new ListNode(0);
        ListNode after = afterDummy;
        
        while (head != null) {
            if (head.val < x) {
                before.next = head;
                before = before.next;
            } else {
                after.next = head;
                after = after.next;
            }
            head = head.next;
        }
        
        after.next = null;
        before.next = afterDummy.next;
        
        return beforeDummy.next;
    }
}
```

## 3.7 실습 문제

### 문제 1: 연결 리스트 뒤집기
주어진 연결 리스트를 재귀적으로 뒤집는 메서드를 구현하세요.

### 문제 2: 교차점 찾기
두 개의 연결 리스트가 교차하는 노드를 찾는 메서드를 구현하세요.

### 문제 3: 리스트 재정렬
L0 → L1 → ... → Ln-1 → Ln을 L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ... 순서로 재정렬하세요.

### 문제 4: Add Two Numbers
두 연결 리스트로 표현된 숫자를 더하는 메서드를 구현하세요. (각 노드는 한 자리 숫자)

## 3.8 요약

이 장에서는 연결 리스트의 다양한 형태와 구현에 대해 학습했습니다:

1. **단일 연결 리스트**: 가장 기본적인 형태, 한 방향으로만 순회
2. **이중 연결 리스트**: 양방향 순회 가능, 더 많은 메모리 사용
3. **원형 연결 리스트**: 마지막이 처음을 가리킴, 순환 구조
4. **연결 리스트 알고리즘**: Two Pointers, 더미 노드 등 유용한 패턴

연결 리스트는 동적 메모리 할당이 필요한 상황에서 매우 유용한 자료구조입니다. 다음 장에서는 이미 다룬 스택과 큐를 넘어 트리 구조에 대해 알아보겠습니다.