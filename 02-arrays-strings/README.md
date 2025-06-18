# Chapter 2: 배열과 문자열 (Arrays and Strings)

## 2.1 배열 (Array)

### 배열의 정의
배열은 동일한 타입의 데이터를 연속된 메모리 공간에 저장하는 선형 자료구조입니다. 각 요소는 인덱스를 통해 직접 접근할 수 있습니다.

### 배열의 특징
1. **고정된 크기**: 생성 시 크기가 결정되며 변경 불가
2. **연속된 메모리**: 모든 요소가 메모리상에 연속적으로 저장
3. **인덱스 접근**: O(1) 시간에 요소 접근 가능
4. **동일한 타입**: 모든 요소가 같은 데이터 타입

### 배열의 장단점

**장점:**
- 인덱스를 통한 빠른 접근 (O(1))
- 메모리 지역성으로 인한 캐시 효율성
- 구현이 간단함

**단점:**
- 크기가 고정됨
- 삽입/삭제 시 요소 이동 필요 (O(n))
- 메모리 낭비 가능성

### Java에서의 배열 구현

```java
public class ArrayOperations {
    
    // 배열 생성과 초기화
    public void arrayBasics() {
        // 선언과 생성
        int[] numbers = new int[5];  // 크기 5인 정수 배열
        
        // 선언과 동시에 초기화
        int[] primes = {2, 3, 5, 7, 11};
        
        // 2차원 배열
        int[][] matrix = new int[3][3];
        int[][] matrix2 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
    }
    
    // 배열 순회
    public void traverseArray(int[] array) {
        // 전통적인 for 루프
        for (int i = 0; i < array.length; i++) {
            System.out.println("Index " + i + ": " + array[i]);
        }
        
        // 향상된 for 루프 (for-each)
        for (int element : array) {
            System.out.println(element);
        }
    }
    
    // 배열 복사
    public int[] copyArray(int[] original) {
        // 방법 1: 수동 복사
        int[] copy1 = new int[original.length];
        for (int i = 0; i < original.length; i++) {
            copy1[i] = original[i];
        }
        
        // 방법 2: System.arraycopy()
        int[] copy2 = new int[original.length];
        System.arraycopy(original, 0, copy2, 0, original.length);
        
        // 방법 3: Arrays.copyOf()
        int[] copy3 = Arrays.copyOf(original, original.length);
        
        // 방법 4: clone()
        int[] copy4 = original.clone();
        
        return copy3;
    }
    
    // 배열에서 요소 삽입
    public int[] insert(int[] array, int index, int value) {
        if (index < 0 || index > array.length) {
            throw new IndexOutOfBoundsException("Invalid index");
        }
        
        int[] newArray = new int[array.length + 1];
        
        // index 이전 요소들 복사
        for (int i = 0; i < index; i++) {
            newArray[i] = array[i];
        }
        
        // 새 요소 삽입
        newArray[index] = value;
        
        // index 이후 요소들 복사
        for (int i = index; i < array.length; i++) {
            newArray[i + 1] = array[i];
        }
        
        return newArray;
    }
    
    // 배열에서 요소 삭제
    public int[] delete(int[] array, int index) {
        if (index < 0 || index >= array.length) {
            throw new IndexOutOfBoundsException("Invalid index");
        }
        
        int[] newArray = new int[array.length - 1];
        
        // index 이전 요소들 복사
        for (int i = 0; i < index; i++) {
            newArray[i] = array[i];
        }
        
        // index 이후 요소들 복사
        for (int i = index + 1; i < array.length; i++) {
            newArray[i - 1] = array[i];
        }
        
        return newArray;
    }
    
    // 배열 회전
    public void rotateArray(int[] array, int k) {
        int n = array.length;
        k = k % n;  // k가 n보다 큰 경우 처리
        
        // 방법 1: 추가 배열 사용
        int[] temp = new int[n];
        for (int i = 0; i < n; i++) {
            temp[(i + k) % n] = array[i];
        }
        System.arraycopy(temp, 0, array, 0, n);
        
        // 방법 2: 역순 이용 (in-place)
        // 1. 전체 배열 역순
        // 2. 처음 k개 역순
        // 3. 나머지 역순
    }
    
    // 배열 검색
    public int search(int[] array, int target) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
```

## 2.2 다차원 배열

### 2차원 배열
행과 열로 구성된 배열로, 행렬이나 표를 표현할 때 사용됩니다.

```java
public class Matrix {
    private int[][] data;
    private int rows;
    private int cols;
    
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new int[rows][cols];
    }
    
    // 행렬 덧셈
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match");
        }
        
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    // 행렬 곱셈
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                "First matrix columns must equal second matrix rows");
        }
        
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this.cols; k++) {
                    result.data[i][j] += this.data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    // 전치 행렬
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }
    
    // 대각선 순회
    public void printDiagonals() {
        // 주 대각선
        System.out.println("Main diagonal:");
        for (int i = 0; i < Math.min(rows, cols); i++) {
            System.out.print(data[i][i] + " ");
        }
        System.out.println();
        
        // 부 대각선
        System.out.println("Anti-diagonal:");
        for (int i = 0; i < Math.min(rows, cols); i++) {
            System.out.print(data[i][cols - 1 - i] + " ");
        }
        System.out.println();
    }
    
    // 나선형 순회
    public List<Integer> spiralTraverse() {
        List<Integer> result = new ArrayList<>();
        
        int top = 0, bottom = rows - 1;
        int left = 0, right = cols - 1;
        
        while (top <= bottom && left <= right) {
            // 오른쪽으로
            for (int j = left; j <= right; j++) {
                result.add(data[top][j]);
            }
            top++;
            
            // 아래로
            for (int i = top; i <= bottom; i++) {
                result.add(data[i][right]);
            }
            right--;
            
            // 왼쪽으로
            if (top <= bottom) {
                for (int j = right; j >= left; j--) {
                    result.add(data[bottom][j]);
                }
                bottom--;
            }
            
            // 위로
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    result.add(data[i][left]);
                }
                left++;
            }
        }
        
        return result;
    }
}
```

## 2.3 문자열 (String)

### 문자열의 특징
Java에서 String은 불변(immutable) 객체입니다. 한 번 생성된 문자열은 변경할 수 없습니다.

### String vs StringBuilder vs StringBuffer
- **String**: 불변, 스레드 안전
- **StringBuilder**: 가변, 스레드 안전하지 않음, 빠름
- **StringBuffer**: 가변, 스레드 안전, 느림

```java
public class StringOperations {
    
    // 문자열 기본 연산
    public void stringBasics() {
        String str1 = "Hello";
        String str2 = "World";
        
        // 연결
        String concat = str1 + " " + str2;  // "Hello World"
        
        // 길이
        int length = str1.length();  // 5
        
        // 문자 접근
        char ch = str1.charAt(0);  // 'H'
        
        // 부분 문자열
        String sub = str1.substring(1, 4);  // "ell"
        
        // 비교
        boolean equals = str1.equals("Hello");  // true
        int compare = str1.compareTo(str2);  // < 0
        
        // 검색
        int index = str1.indexOf('l');  // 2
        boolean contains = str1.contains("ell");  // true
    }
    
    // 문자열 뒤집기
    public String reverse(String str) {
        // 방법 1: StringBuilder 사용
        return new StringBuilder(str).reverse().toString();
        
        // 방법 2: 문자 배열 사용
        /*
        char[] chars = str.toCharArray();
        int left = 0, right = chars.length - 1;
        
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
        
        return new String(chars);
        */
    }
    
    // 팰린드롬 확인
    public boolean isPalindrome(String str) {
        // 공백과 특수문자 제거, 소문자로 변환
        str = str.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
        
        int left = 0;
        int right = str.length() - 1;
        
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        
        return true;
    }
    
    // 애너그램 확인
    public boolean isAnagram(String s1, String s2) {
        if (s1.length() != s2.length()) {
            return false;
        }
        
        // 방법 1: 정렬 사용
        char[] chars1 = s1.toCharArray();
        char[] chars2 = s2.toCharArray();
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        return Arrays.equals(chars1, chars2);
        
        // 방법 2: 문자 빈도 계산
        /*
        int[] count = new int[256];  // ASCII 문자
        
        for (char c : s1.toCharArray()) {
            count[c]++;
        }
        
        for (char c : s2.toCharArray()) {
            count[c]--;
            if (count[c] < 0) {
                return false;
            }
        }
        
        return true;
        */
    }
    
    // 문자열 압축
    public String compress(String str) {
        if (str == null || str.length() <= 2) {
            return str;
        }
        
        StringBuilder compressed = new StringBuilder();
        int count = 1;
        
        for (int i = 0; i < str.length(); i++) {
            if (i + 1 < str.length() && str.charAt(i) == str.charAt(i + 1)) {
                count++;
            } else {
                compressed.append(str.charAt(i)).append(count);
                count = 1;
            }
        }
        
        return compressed.length() < str.length() ? 
               compressed.toString() : str;
    }
    
    // 문자열에서 첫 번째 유일한 문자 찾기
    public int firstUniqueChar(String str) {
        int[] count = new int[26];  // 소문자만 가정
        
        // 문자 빈도 계산
        for (char c : str.toCharArray()) {
            count[c - 'a']++;
        }
        
        // 첫 번째 유일한 문자 찾기
        for (int i = 0; i < str.length(); i++) {
            if (count[str.charAt(i) - 'a'] == 1) {
                return i;
            }
        }
        
        return -1;
    }
    
    // 가장 긴 공통 접두사
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        
        String prefix = strs[0];
        
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.isEmpty()) {
                    return "";
                }
            }
        }
        
        return prefix;
    }
}
```

## 2.4 동적 배열 (Dynamic Array)

### ArrayList의 구현
Java의 ArrayList는 동적으로 크기가 조정되는 배열입니다.

```java
public class DynamicArray<T> {
    private T[] array;
    private int size;
    private int capacity;
    private static final int DEFAULT_CAPACITY = 10;
    
    @SuppressWarnings("unchecked")
    public DynamicArray() {
        this.capacity = DEFAULT_CAPACITY;
        this.array = (T[]) new Object[capacity];
        this.size = 0;
    }
    
    // 요소 추가
    public void add(T element) {
        if (size == capacity) {
            resize();
        }
        array[size++] = element;
    }
    
    // 특정 위치에 요소 삽입
    public void add(int index, T element) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException();
        }
        
        if (size == capacity) {
            resize();
        }
        
        // 요소들을 오른쪽으로 이동
        for (int i = size; i > index; i--) {
            array[i] = array[i - 1];
        }
        
        array[index] = element;
        size++;
    }
    
    // 요소 접근
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        return array[index];
    }
    
    // 요소 설정
    public T set(int index, T element) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        T oldValue = array[index];
        array[index] = element;
        return oldValue;
    }
    
    // 요소 제거
    public T remove(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException();
        }
        
        T removedElement = array[index];
        
        // 요소들을 왼쪽으로 이동
        for (int i = index; i < size - 1; i++) {
            array[i] = array[i + 1];
        }
        
        array[--size] = null;  // 메모리 누수 방지
        
        // 크기가 용량의 1/4이면 축소
        if (size > 0 && size == capacity / 4) {
            shrink();
        }
        
        return removedElement;
    }
    
    // 크기 확장
    @SuppressWarnings("unchecked")
    private void resize() {
        capacity *= 2;
        T[] newArray = (T[]) new Object[capacity];
        System.arraycopy(array, 0, newArray, 0, size);
        array = newArray;
    }
    
    // 크기 축소
    @SuppressWarnings("unchecked")
    private void shrink() {
        capacity /= 2;
        T[] newArray = (T[]) new Object[capacity];
        System.arraycopy(array, 0, newArray, 0, size);
        array = newArray;
    }
    
    // 검색
    public int indexOf(T element) {
        for (int i = 0; i < size; i++) {
            if (element == null ? array[i] == null : element.equals(array[i])) {
                return i;
            }
        }
        return -1;
    }
    
    // 포함 여부
    public boolean contains(T element) {
        return indexOf(element) != -1;
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
        for (int i = 0; i < size; i++) {
            array[i] = null;
        }
        size = 0;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < size; i++) {
            sb.append(array[i]);
            if (i < size - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
```

## 2.5 배열 알고리즘 패턴

### Two Pointers 기법
배열의 양 끝에서 시작하여 중앙으로 이동하는 패턴입니다.

```java
public class TwoPointers {
    
    // 정렬된 배열에서 두 수의 합 찾기
    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            
            if (sum == target) {
                return new int[]{left, right};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        
        return new int[]{-1, -1};
    }
    
    // 물을 담을 수 있는 최대 용량
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxArea = 0;
        
        while (left < right) {
            int width = right - left;
            int minHeight = Math.min(height[left], height[right]);
            int area = width * minHeight;
            maxArea = Math.max(maxArea, area);
            
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxArea;
    }
}
```

### Sliding Window 기법
고정 크기의 윈도우를 이동시키며 계산하는 패턴입니다.

```java
public class SlidingWindow {
    
    // 크기 k인 부분 배열의 최대 합
    public int maxSumSubarray(int[] arr, int k) {
        if (arr.length < k) {
            return -1;
        }
        
        // 첫 번째 윈도우의 합 계산
        int windowSum = 0;
        for (int i = 0; i < k; i++) {
            windowSum += arr[i];
        }
        
        int maxSum = windowSum;
        
        // 윈도우 슬라이딩
        for (int i = k; i < arr.length; i++) {
            windowSum = windowSum - arr[i - k] + arr[i];
            maxSum = Math.max(maxSum, windowSum);
        }
        
        return maxSum;
    }
    
    // 가장 긴 고유 문자 부분 문자열
    public int lengthOfLongestSubstring(String s) {
        Set<Character> window = new HashSet<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            while (window.contains(s.charAt(right))) {
                window.remove(s.charAt(left));
                left++;
            }
            
            window.add(s.charAt(right));
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
}
```

## 2.6 실습 문제

### 문제 1: 배열 회전
n x n 2D 행렬을 90도 시계방향으로 회전시키는 메서드를 구현하세요. (in-place)

### 문제 2: 중복 제거
정렬된 배열에서 중복을 제거하고 고유한 요소의 개수를 반환하세요. (in-place)

### 문제 3: 문자열 순열
두 문자열이 서로의 순열인지 확인하는 메서드를 구현하세요.

### 문제 4: 부분 배열의 최대 곱
배열에서 연속된 부분 배열의 최대 곱을 찾는 메서드를 구현하세요.

## 2.7 요약

이 장에서는 가장 기본적인 자료구조인 배열과 문자열에 대해 학습했습니다:

1. **배열**은 인덱스 접근이 빠르지만 크기가 고정됨
2. **문자열**은 Java에서 불변 객체로 다뤄짐
3. **동적 배열**은 크기를 자동으로 조정하여 유연성 제공
4. **Two Pointers**와 **Sliding Window**는 배열 문제의 효율적인 해결 패턴

다음 장에서는 동적인 크기 조정이 자유로운 연결 리스트에 대해 알아보겠습니다.