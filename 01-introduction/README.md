# Chapter 1: 자료구조 개요 (Introduction to Data Structures)

## 1.1 자료구조란 무엇인가?

### 정의
자료구조(Data Structure)는 데이터를 효율적으로 저장하고 조직화하는 방법입니다. 컴퓨터 프로그램에서 데이터를 어떻게 저장하고 접근할지를 결정하는 것은 프로그램의 성능과 효율성에 직접적인 영향을 미칩니다.

### 자료구조의 중요성
1. **효율성**: 적절한 자료구조를 선택하면 프로그램의 실행 속도를 크게 향상시킬 수 있습니다.
2. **메모리 관리**: 메모리를 효율적으로 사용하여 공간을 절약할 수 있습니다.
3. **코드 재사용성**: 잘 설계된 자료구조는 다양한 상황에서 재사용할 수 있습니다.
4. **문제 해결**: 복잡한 문제를 더 쉽게 해결할 수 있는 도구를 제공합니다.

### 자료구조의 분류

```
자료구조
├── 선형 자료구조 (Linear)
│   ├── 배열 (Array)
│   ├── 연결 리스트 (Linked List)
│   ├── 스택 (Stack)
│   └── 큐 (Queue)
└── 비선형 자료구조 (Non-linear)
    ├── 트리 (Tree)
    ├── 그래프 (Graph)
    └── 해시 테이블 (Hash Table)
```

## 1.2 추상 자료형 (Abstract Data Type, ADT)

### ADT란?
추상 자료형은 데이터와 그 데이터에 대한 연산을 추상적으로 정의한 것입니다. 구현 세부사항은 숨기고 인터페이스만을 제공합니다.

### ADT의 특징
1. **캡슐화**: 데이터와 연산을 하나의 단위로 묶음
2. **정보 은닉**: 내부 구현을 숨기고 인터페이스만 노출
3. **재사용성**: 한 번 정의하면 여러 곳에서 사용 가능

### Java에서의 ADT 예제

```java
// Stack ADT 인터페이스 정의
public interface Stack<T> {
    void push(T item);      // 요소 추가
    T pop();                // 요소 제거 및 반환
    T peek();               // 최상단 요소 확인
    boolean isEmpty();      // 비어있는지 확인
    int size();            // 크기 반환
}

// Stack ADT 구현 예제
public class ArrayStack<T> implements Stack<T> {
    private T[] array;
    private int top;
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public ArrayStack() {
        array = (T[]) new Object[DEFAULT_CAPACITY];
        top = -1;
    }
    
    @Override
    public void push(T item) {
        if (top == array.length - 1) {
            resize();
        }
        array[++top] = item;
    }
    
    @Override
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        T item = array[top];
        array[top--] = null; // 메모리 누수 방지
        return item;
    }
    
    @Override
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return array[top];
    }
    
    @Override
    public boolean isEmpty() {
        return top == -1;
    }
    
    @Override
    public int size() {
        return top + 1;
    }
    
    @SuppressWarnings("unchecked")
    private void resize() {
        T[] newArray = (T[]) new Object[array.length * 2];
        System.arraycopy(array, 0, newArray, 0, array.length);
        array = newArray;
    }
}
```

## 1.3 알고리즘 복잡도 분석

### 시간 복잡도 (Time Complexity)
알고리즘이 문제를 해결하는 데 걸리는 시간을 입력 크기에 대한 함수로 표현합니다.

### Big-O 표기법
최악의 경우의 시간 복잡도를 나타내는 표기법입니다.

| 표기법 | 명칭 | 예시 |
|--------|------|------|
| O(1) | 상수 시간 | 배열 인덱스 접근 |
| O(log n) | 로그 시간 | 이진 탐색 |
| O(n) | 선형 시간 | 선형 탐색 |
| O(n log n) | 선형 로그 시간 | 효율적인 정렬 (병합, 퀵) |
| O(n²) | 이차 시간 | 중첩 반복문, 버블 정렬 |
| O(2ⁿ) | 지수 시간 | 재귀적 피보나치 |

### 시간 복잡도 분석 예제

```java
public class ComplexityExamples {
    
    // O(1) - 상수 시간
    public int getFirst(int[] array) {
        return array[0];
    }
    
    // O(n) - 선형 시간
    public int linearSearch(int[] array, int target) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == target) {
                return i;
            }
        }
        return -1;
    }
    
    // O(log n) - 로그 시간
    public int binarySearch(int[] sortedArray, int target) {
        int left = 0;
        int right = sortedArray.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (sortedArray[mid] == target) {
                return mid;
            } else if (sortedArray[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
    
    // O(n²) - 이차 시간
    public void bubbleSort(int[] array) {
        int n = array.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (array[j] > array[j + 1]) {
                    // swap
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }
    
    // O(2ⁿ) - 지수 시간 (비효율적인 예)
    public int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
```

### 공간 복잡도 (Space Complexity)
알고리즘이 실행되는 동안 필요한 메모리 공간을 나타냅니다.

```java
public class SpaceComplexityExamples {
    
    // O(1) 공간 복잡도
    public int sum(int[] array) {
        int total = 0;  // 고정된 공간만 사용
        for (int num : array) {
            total += num;
        }
        return total;
    }
    
    // O(n) 공간 복잡도
    public int[] copyArray(int[] array) {
        int[] newArray = new int[array.length];  // n개의 추가 공간 필요
        for (int i = 0; i < array.length; i++) {
            newArray[i] = array[i];
        }
        return newArray;
    }
    
    // O(n) 공간 복잡도 (재귀 호출 스택)
    public int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);  // n개의 재귀 호출
    }
}
```

## 1.4 자료구조 선택 가이드

### 고려사항
1. **연산의 종류**: 삽입, 삭제, 검색, 정렬 등 어떤 연산이 주로 필요한가?
2. **연산의 빈도**: 각 연산이 얼마나 자주 수행되는가?
3. **데이터의 크기**: 저장할 데이터의 양은 얼마나 되는가?
4. **메모리 제약**: 사용 가능한 메모리는 충분한가?
5. **데이터의 특성**: 정렬되어 있는가? 중복이 허용되는가?

### 자료구조별 특징 요약

| 자료구조 | 접근 | 검색 | 삽입 | 삭제 | 특징 |
|----------|------|------|------|------|------|
| 배열 | O(1) | O(n) | O(n) | O(n) | 인덱스 접근 빠름 |
| 연결 리스트 | O(n) | O(n) | O(1) | O(1) | 동적 크기 |
| 스택 | O(n) | O(n) | O(1) | O(1) | LIFO |
| 큐 | O(n) | O(n) | O(1) | O(1) | FIFO |
| 해시 테이블 | N/A | O(1)* | O(1)* | O(1)* | 빠른 검색 |
| 이진 탐색 트리 | O(log n)* | O(log n)* | O(log n)* | O(log n)* | 정렬된 데이터 |

*평균적인 경우

## 1.5 실습 문제

### 문제 1: 시간 복잡도 분석
다음 메서드의 시간 복잡도를 분석하세요:

```java
public int mysteryFunction(int n) {
    int count = 0;
    for (int i = n; i > 0; i /= 2) {
        for (int j = 0; j < i; j++) {
            count++;
        }
    }
    return count;
}
```

### 문제 2: ADT 구현
Queue ADT를 배열을 사용하여 구현하세요. 다음 메서드를 포함해야 합니다:
- enqueue(T item): 요소 추가
- dequeue(): 요소 제거 및 반환
- peek(): 맨 앞 요소 확인
- isEmpty(): 비어있는지 확인
- size(): 크기 반환

### 문제 3: 효율성 개선
다음 코드를 더 효율적으로 개선하세요:

```java
public boolean hasDuplicate(int[] array) {
    for (int i = 0; i < array.length; i++) {
        for (int j = 0; j < array.length; j++) {
            if (i != j && array[i] == array[j]) {
                return true;
            }
        }
    }
    return false;
}
```

## 1.6 요약

이 장에서는 자료구조의 기본 개념과 중요성을 학습했습니다. 주요 내용은:

1. 자료구조는 데이터를 효율적으로 저장하고 관리하는 방법입니다.
2. ADT는 구현과 인터페이스를 분리하여 추상화를 제공합니다.
3. 알고리즘의 효율성은 시간 복잡도와 공간 복잡도로 측정됩니다.
4. 문제에 적합한 자료구조를 선택하는 것이 중요합니다.

다음 장에서는 가장 기본적인 자료구조인 배열과 문자열에 대해 자세히 알아보겠습니다.