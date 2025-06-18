# Chapter 3: 연결 리스트 (Linked Lists)

## 3.1 연결 리스트란?

### 정의
연결 리스트는 각 노드가 데이터와 다음 노드에 대한 참조(포인터)를 가지는 선형 자료구조입니다. 배열과 달리 메모리상에 연속적으로 저장되지 않습니다.

### 연결 리스트 vs 배열

| 특성 | 배열 | 연결 리스트 |
|------|------|------------|
| 메모리 할당 | 연속적 | 비연속적 |
| 크기 | 고정 | 동적 |
| 접근 시간 | O(1) | O(n) |
| 삽입/삭제 (처음) | O(n) | O(1) |
| 삽입/삭제 (중간) | O(n) | O(n) |
| 메모리 오버헤드 | 낮음 | 높음 (포인터 저장) |
| 캐시 지역성 | 좋음 | 나쁨 |

### 연결 리스트의 장단점

**장점:**
- 동적 크기 조정
- 효율적인 삽입/삭제 (처음과 끝)
- 메모리 효율성 (필요한 만큼만 사용)
- 구현의 유연성

**단점:**
- 임의 접근 불가 (순차 접근만 가능)
- 추가 메모리 필요 (포인터 저장)
- 캐시 성능 저하
- 역방향 순회 어려움 (단일 연결 리스트)

## 3.2 단일 연결 리스트 (Singly Linked List)

### 노드 구조
```java
public class Node<T> {
    T data;
    Node<T> next;
    
    public Node(T data) {
        this.data = data;
        this.next = null;
    }
}
```

### 단일 연결 리스트 구현

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
    
    // 맨 앞에 삽입
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        newNode.next = head;
        head = newNode;
        size++;
    }
    
    // 맨 뒤에 삽입
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
    
    // 특정 위치에 삽입
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
    
    // 맨 앞 요소 제거
    public T removeFirst() {
        if (head == null) {
            throw new NoSuchElementException();
        }
        
        T data = head.data;
        head = head.next;
        size--;
        return data;
    }
    
    // 맨 뒤 요소 제거
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
    
    // 특정 위치의 요소 제거
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
    
    // 값으로 제거
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
    
    // 특정 위치의 요소 반환
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
    
    // 특정 위치의 요소 설정
    public void set(int index, T data) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        Node<T> current = head;
        for (int i = 0; i < index; i++) {
            current = current.next;
        }
        
        current.data = data;
    }
    
    // 검색
    public int indexOf(T value) {
        Node<T> current = head;
        int index = 0;
        
        while (current != null) {
            if (current.data.equals(value)) {
                return index;
            }
            current = current.next;
            index++;
        }
        
        return -1;
    }
    
    // 포함 여부
    public boolean contains(T value) {
        return indexOf(value) != -1;
    }
    
    // 크기
    public int size() {
        return size;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return head == null;
    }
    
    // 전체 삭제
    public void clear() {
        head = null;
        size = 0;
    }
    
    // 리스트 뒤집기
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
    
    // 중간 노드 찾기
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
    
    // 순환 감지
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
    
    // 출력
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

### 노드 구조
```java
public class DoublyNode<T> {
    T data;
    DoublyNode<T> prev;
    DoublyNode<T> next;
    
    public DoublyNode(T data) {
        this.data = data;
        this.prev = null;
        this.next = null;
    }
}
```

### 이중 연결 리스트 구현

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
    
    // 맨 앞에 삽입
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
    
    // 맨 뒤에 삽입
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
    
    // 특정 위치에 삽입
    public void add(int index, T data) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == 0) {
            addFirst(data);
            return;
        }
        
        if (index == size) {
            addLast(data);
            return;
        }
        
        Node<T> newNode = new Node<>(data);
        Node<T> current = getNode(index);
        
        newNode.prev = current.prev;
        newNode.next = current;
        current.prev.next = newNode;
        current.prev = newNode;
        size++;
    }
    
    // 맨 앞 요소 제거
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
    
    // 맨 뒤 요소 제거
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
    
    // 특정 위치의 요소 제거
    public T remove(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (index == 0) {
            return removeFirst();
        }
        
        if (index == size - 1) {
            return removeLast();
        }
        
        Node<T> current = getNode(index);
        T data = current.data;
        
        current.prev.next = current.next;
        current.next.prev = current.prev;
        size--;
        return data;
    }
    
    // 특정 노드 가져오기 (최적화)
    private Node<T> getNode(int index) {
        Node<T> current;
        
        // 앞쪽에서 탐색이 빠른 경우
        if (index < size / 2) {
            current = head;
            for (int i = 0; i < index; i++) {
                current = current.next;
            }
        } 
        // 뒤쪽에서 탐색이 빠른 경우
        else {
            current = tail;
            for (int i = size - 1; i > index; i--) {
                current = current.prev;
            }
        }
        
        return current;
    }
    
    // 특정 위치의 요소 반환
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        return getNode(index).data;
    }
    
    // 특정 위치의 요소 설정
    public void set(int index, T data) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        getNode(index).data = data;
    }
    
    // 역방향 순회
    public void printReverse() {
        Node<T> current = tail;
        System.out.print("[");
        
        while (current != null) {
            System.out.print(current.data);
            if (current.prev != null) {
                System.out.print(" <- ");
            }
            current = current.prev;
        }
        
        System.out.println("]");
    }
    
    // 크기
    public int size() {
        return size;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return size == 0;
    }
    
    // 전체 삭제
    public void clear() {
        head = tail = null;
        size = 0;
    }
    
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

### 원형 연결 리스트의 특징
- 마지막 노드가 첫 번째 노드를 가리킴
- 순환 구조로 끝이 없음
- 어느 노드에서든 전체 리스트 순회 가능

### 원형 연결 리스트 구현

```java
public class CircularLinkedList<T> {
    private Node<T> tail;  // tail만 유지 (head는 tail.next)
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
        this.tail = null;
        this.size = 0;
    }
    
    // 맨 앞에 삽입
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (tail == null) {
            tail = newNode;
            tail.next = tail;
        } else {
            newNode.next = tail.next;
            tail.next = newNode;
        }
        size++;
    }
    
    // 맨 뒤에 삽입
    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        
        if (tail == null) {
            tail = newNode;
            tail.next = tail;
        } else {
            newNode.next = tail.next;
            tail.next = newNode;
            tail = newNode;
        }
        size++;
    }
    
    // 맨 앞 요소 제거
    public T removeFirst() {
        if (tail == null) {
            throw new NoSuchElementException();
        }
        
        Node<T> head = tail.next;
        T data = head.data;
        
        if (tail == head) {
            tail = null;
        } else {
            tail.next = head.next;
        }
        size--;
        return data;
    }
    
    // 맨 뒤 요소 제거
    public T removeLast() {
        if (tail == null) {
            throw new NoSuchElementException();
        }
        
        T data = tail.data;
        
        if (tail.next == tail) {
            tail = null;
        } else {
            Node<T> current = tail.next;
            while (current.next != tail) {
                current = current.next;
            }
            current.next = tail.next;
            tail = current;
        }
        size--;
        return data;
    }
    
    // 회전 (k번 왼쪽으로)
    public void rotate(int k) {
        if (tail == null || k == 0) {
            return;
        }
        
        k = k % size;
        for (int i = 0; i < k; i++) {
            tail = tail.next;
        }
    }
    
    // 출력 (n개까지만)
    public void print(int n) {
        if (tail == null) {
            System.out.println("[]");
            return;
        }
        
        System.out.print("[");
        Node<T> current = tail.next;
        
        for (int i = 0; i < n && i < size; i++) {
            System.out.print(current.data);
            current = current.next;
            if (i < n - 1 && i < size - 1) {
                System.out.print(" -> ");
            }
        }
        
        if (n < size) {
            System.out.print(" -> ...");
        }
        System.out.println("]");
    }
    
    // 크기
    public int size() {
        return size;
    }
    
    // 비어있는지 확인
    public boolean isEmpty() {
        return tail == null;
    }
}
```

## 3.5 연결 리스트 알고리즘

### 두 정렬된 리스트 병합

```java
public class LinkedListAlgorithms {
    
    // 두 정렬된 리스트 병합
    public Node<Integer> mergeTwoLists(Node<Integer> l1, Node<Integer> l2) {
        Node<Integer> dummy = new Node<>(0);
        Node<Integer> current = dummy;
        
        while (l1 != null && l2 != null) {
            if (l1.data <= l2.data) {
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
        }
        if (l2 != null) {
            current.next = l2;
        }
        
        return dummy.next;
    }
    
    // k개씩 그룹으로 뒤집기
    public Node<Integer> reverseKGroup(Node<Integer> head, int k) {
        if (head == null || k == 1) {
            return head;
        }
        
        // k개의 노드가 있는지 확인
        Node<Integer> current = head;
        int count = 0;
        while (current != null && count < k) {
            current = current.next;
            count++;
        }
        
        if (count < k) {
            return head;
        }
        
        // k개 노드 뒤집기
        current = head;
        Node<Integer> prev = null;
        Node<Integer> next = null;
        count = 0;
        
        while (current != null && count < k) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
            count++;
        }
        
        // 재귀적으로 다음 그룹 처리
        if (next != null) {
            head.next = reverseKGroup(next, k);
        }
        
        return prev;
    }
    
    // 팰린드롬 확인
    public boolean isPalindrome(Node<Integer> head) {
        if (head == null || head.next == null) {
            return true;
        }
        
        // 중간 지점 찾기
        Node<Integer> slow = head;
        Node<Integer> fast = head;
        
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        // 후반부 뒤집기
        Node<Integer> secondHalf = reverseList(slow.next);
        
        // 비교
        Node<Integer> p1 = head;
        Node<Integer> p2 = secondHalf;
        boolean isPalindrome = true;
        
        while (p2 != null) {
            if (!p1.data.equals(p2.data)) {
                isPalindrome = false;
                break;
            }
            p1 = p1.next;
            p2 = p2.next;
        }
        
        // 원래 구조로 복원
        slow.next = reverseList(secondHalf);
        
        return isPalindrome;
    }
    
    // 리스트 뒤집기 (헬퍼 메서드)
    private Node<Integer> reverseList(Node<Integer> head) {
        Node<Integer> prev = null;
        Node<Integer> current = head;
        
        while (current != null) {
            Node<Integer> next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
    
    // 교집합 지점 찾기
    public Node<Integer> getIntersectionNode(Node<Integer> headA, Node<Integer> headB) {
        if (headA == null || headB == null) {
            return null;
        }
        
        Node<Integer> p