# Chapter 9: 탐색 알고리즘 (Searching Algorithms)

## 9.1 탐색 알고리즘 개요

### 탐색이란?
탐색(Search)은 데이터 집합에서 특정 값을 찾는 과정입니다. 효율적인 탐색은 많은 응용 프로그램의 성능에 직접적인 영향을 미칩니다.

### 탐색 알고리즘의 분류
1. **순차 탐색**: 처음부터 끝까지 하나씩 확인
2. **이진 탐색**: 정렬된 데이터에서 반씩 나누어 탐색
3. **해시 기반 탐색**: 해시 테이블을 이용한 O(1) 탐색
4. **트리 기반 탐색**: BST, B-Tree 등을 이용
5. **그래프 탐색**: DFS, BFS 등

### 탐색 성능 평가
- **시간 복잡도**: 평균/최악의 경우
- **공간 복잡도**: 추가 메모리 사용량
- **전처리 시간**: 자료구조 구성 시간
- **적응성**: 데이터 특성에 따른 성능 변화

## 9.2 선형 탐색 (Linear Search)

### 기본 선형 탐색
가장 단순한 탐색 방법으로, 처음부터 끝까지 순차적으로 확인합니다.

```java
public class LinearSearch {
    
    // 기본 선형 탐색
    public static int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
    
    // 제네릭 선형 탐색
    public static <T> int linearSearch(T[] arr, T target) {
        for (int i = 0; i < arr.length; i++) {
            if (target.equals(arr[i])) {
                return i;
            }
        }
        return -1;
    }
    
    // 보초법을 이용한 선형 탐색
    public static int sentinelLinearSearch(int[] arr, int target) {
        int n = arr.length;
        int last = arr[n - 1];
        
        // 보초 설정
        arr[n - 1] = target;
        
        int i = 0;
        while (arr[i] != target) {
            i++;
        }
        
        // 원래 값 복원
        arr[n - 1] = last;
        
        // 찾은 위치가 마지막이고 원래 값이 target이 아니면 못 찾은 것
        if (i == n - 1 && last != target) {
            return -1;
        }
        
        return i;
    }
    
    // 순서 이동 선형 탐색 (자주 검색되는 요소를 앞으로)
    public static int moveToFrontSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                if (i > 0) {
                    // 찾은 요소를 맨 앞으로 이동
                    int temp = arr[i];
                    System.arraycopy(arr, 0, arr, 1, i);
                    arr[0] = temp;
                    return 0;
                }
                return i;
            }
        }
        return -1;
    }
    
    // 전치 선형 탐색 (찾은 요소를 한 칸 앞으로)
    public static int transposeSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                if (i > 0) {
                    // 이전 요소와 교환
                    int temp = arr[i];
                    arr[i] = arr[i - 1];
                    arr[i - 1] = temp;
                    return i - 1;
                }
                return i;
            }
        }
        return -1;
    }
    
    // 순차 탐색으로 모든 위치 찾기
    public static List<Integer> findAllOccurrences(int[] arr, int target) {
        List<Integer> positions = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                positions.add(i);
            }
        }
        return positions;
    }
    
    // 조건을 만족하는 첫 번째 요소 찾기
    public static <T> int findFirst(T[] arr, Predicate<T> condition) {
        for (int i = 0; i < arr.length; i++) {
            if (condition.test(arr[i])) {
                return i;
            }
        }
        return -1;
    }
    
    // 최댓값/최솟값 찾기
    public static int findMax(int[] arr) {
        if (arr.length == 0) {
            throw new IllegalArgumentException("Array is empty");
        }
        
        int maxIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    public static int findMin(int[] arr) {
        if (arr.length == 0) {
            throw new IllegalArgumentException("Array is empty");
        }
        
        int minIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < arr[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }
    
    // k번째 최댓값/최솟값 찾기 (부분 정렬)
    public static int findKthLargest(int[] arr, int k) {
        if (k <= 0 || k > arr.length) {
            throw new IllegalArgumentException("Invalid k");
        }
        
        // 최소 힙을 사용하여 k개의 최대 요소 유지
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k);
        
        for (int num : arr) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        
        return minHeap.peek();
    }
}
```

## 9.3 이진 탐색 (Binary Search)

### 기본 이진 탐색
정렬된 배열에서 중간값과 비교하여 탐색 범위를 절반씩 줄여나가는 방법입니다.

```java
public class BinarySearch {
    
    // 반복적 이진 탐색
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
    
    // 재귀적 이진 탐색
    public static int recursiveBinarySearch(int[] arr, int target) {
        return recursiveBinarySearch(arr, target, 0, arr.length - 1);
    }
    
    private static int recursiveBinarySearch(int[] arr, int target, int left, int right) {
        if (left > right) {
            return -1;
        }
        
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            return recursiveBinarySearch(arr, target, mid + 1, right);
        } else {
            return recursiveBinarySearch(arr, target, left, mid - 1);
        }
    }
    
    // 제네릭 이진 탐색
    public static <T extends Comparable<T>> int binarySearch(T[] arr, T target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int cmp = arr[mid].compareTo(target);
            
            if (cmp == 0) {
                return mid;
            } else if (cmp < 0) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
    
    // Lower Bound: target 이상인 첫 번째 요소의 위치
    public static int lowerBound(int[] arr, int target) {
        int left = 0;
        int right = arr.length;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }
    
    // Upper Bound: target 초과인 첫 번째 요소의 위치
    public static int upperBound(int[] arr, int target) {
        int left = 0;
        int right = arr.length;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }
    
    // 범위 검색: [start, end] 범위의 요소 개수
    public static int countInRange(int[] arr, int start, int end) {
        int leftIdx = lowerBound(arr, start);
        int rightIdx = upperBound(arr, end);
        return rightIdx - leftIdx;
    }
    
    // 가장 가까운 값 찾기
    public static int findClosest(int[] arr, int target) {
        if (arr.length == 0) {
            return -1;
        }
        
        int left = 0;
        int right = arr.length - 1;
        int closest = 0;
        int minDiff = Integer.MAX_VALUE;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int diff = Math.abs(arr[mid] - target);
            
            if (diff < minDiff) {
                minDiff = diff;
                closest = mid;
            }
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        // 경계 확인
        if (left < arr.length && Math.abs(arr[left] - target) < minDiff) {
            closest = left;
        }
        if (right >= 0 && Math.abs(arr[right] - target) < Math.abs(arr[closest] - target)) {
            closest = right;
        }
        
        return closest;
    }
    
    // 회전된 정렬 배열에서 탐색
    public static int searchRotatedArray(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            }
            
            // 왼쪽 부분이 정렬되어 있는 경우
            if (arr[left] <= arr[mid]) {
                if (target >= arr[left] && target < arr[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // 오른쪽 부분이 정렬되어 있는 경우
            else {
                if (target > arr[mid] && target <= arr[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
    
    // 2D 행렬에서 탐색
    public static boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        
        // 방법 1: 2D를 1D로 취급
        int left = 0;
        int right = rows * cols - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int midValue = matrix[mid / cols][mid % cols];
            
            if (midValue == target) {
                return true;
            } else if (midValue < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return false;
    }
    
    // 피크 요소 찾기
    public static int findPeakElement(int[] arr) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] > arr[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        return left;
    }
}
```

## 9.4 보간 탐색 (Interpolation Search)

### 균등 분포 데이터에 효율적인 탐색
데이터가 균등하게 분포되어 있을 때 더 효율적인 탐색 방법입니다.

```java
public class InterpolationSearch {
    
    // 기본 보간 탐색
    public static int interpolationSearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right && target >= arr[left] && target <= arr[right]) {
            // 균등 분포 가정하에 위치 추정
            if (arr[right] == arr[left]) {
                if (arr[left] == target) {
                    return left;
                }
                break;
            }
            
            int pos = left + ((target - arr[left]) * (right - left)) / 
                            (arr[right] - arr[left]);
            
            if (arr[pos] == target) {
                return pos;
            } else if (arr[pos] < target) {
                left = pos + 1;
            } else {
                right = pos - 1;
            }
        }
        
        return -1;
    }
    
    // 개선된 보간 탐색 (이진 탐색과 결합)
    public static int hybridInterpolationSearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        int iterations = 0;
        int maxIterations = (int)(Math.log(arr.length) / Math.log(2)) + 1;
        
        while (left <= right && target >= arr[left] && target <= arr[right]) {
            iterations++;
            
            // 일정 횟수 후 이진 탐색으로 전환
            if (iterations > maxIterations) {
                return binarySearchInRange(arr, target, left, right);
            }
            
            // 보간 위치 계산
            int pos;
            if (arr[right] == arr[left]) {
                pos = left;
            } else {
                double fraction = (double)(target - arr[left]) / (arr[right] - arr[left]);
                pos = left + (int)(fraction * (right - left));
                
                // 범위 검증
                pos = Math.max(left, Math.min(right, pos));
            }
            
            if (arr[pos] == target) {
                return pos;
            } else if (arr[pos] < target) {
                left = pos + 1;
            } else {
                right = pos - 1;
            }
        }
        
        return -1;
    }
    
    private static int binarySearchInRange(int[] arr, int target, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
    
    // 문자열 배열에서 보간 탐색
    public static int interpolationSearchStrings(String[] arr, String target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            // 빈 문자열 처리
            while (left <= right && arr[left].isEmpty()) {
                left++;
            }
            while (left <= right && arr[right].isEmpty()) {
                right--;
            }
            
            if (left > right) {
                return -1;
            }
            
            // 중간 위치 추정 (첫 문자 기준)
            int mid = left;
            if (!arr[left].equals(arr[right])) {
                char leftChar = arr[left].charAt(0);
                char rightChar = arr[right].charAt(0);
                char targetChar = target.charAt(0);
                
                if (targetChar >= leftChar && targetChar <= rightChar) {
                    double fraction = (double)(targetChar - leftChar) / (rightChar - leftChar);
                    mid = left + (int)(fraction * (right - left));
                }
            }
            
            // 빈 문자열이면 가장 가까운 비어있지 않은 문자열 찾기
            if (arr[mid].isEmpty()) {
                int leftMid = mid - 1;
                int rightMid = mid + 1;
                
                while (true) {
                    if (leftMid < left && rightMid > right) {
                        return -1;
                    }
                    if (leftMid >= left && !arr[leftMid].isEmpty()) {
                        mid = leftMid;
                        break;
                    }
                    if (rightMid <= right && !arr[rightMid].isEmpty()) {
                        mid = rightMid;
                        break;
                    }
                    leftMid--;
                    rightMid++;
                }
            }
            
            int cmp = arr[mid].compareTo(target);
            if (cmp == 0) {
                return mid;
            } else if (cmp < 0) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
}
```

## 9.5 지수 탐색 (Exponential Search)

### 무한 또는 크기를 모르는 배열에서의 탐색

```java
public class ExponentialSearch {
    
    // 기본 지수 탐색
    public static int exponentialSearch(int[] arr, int target) {
        if (arr[0] == target) {
            return 0;
        }
        
        // 범위를 지수적으로 증가시켜 탐색
        int bound = 1;
        while (bound < arr.length && arr[bound] <= target) {
            bound *= 2;
        }
        
        // 찾은 범위에서 이진 탐색
        int left = bound / 2;
        int right = Math.min(bound, arr.length - 1);
        
        return binarySearchInRange(arr, target, left, right);
    }
    
    // 무한 배열에서의 탐색
    public static int searchInfiniteArray(InfiniteArray arr, int target) {
        int low = 0;
        int high = 1;
        
        // 목표값보다 큰 요소를 찾을 때까지 범위 확장
        while (arr.get(high) < target) {
            low = high;
            high *= 2;
        }
        
        // 이진 탐색
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int midValue = arr.get(mid);
            
            if (midValue == target) {
                return mid;
            } else if (midValue < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        return -1;
    }
    
    // 무한 배열 인터페이스
    interface InfiniteArray {
        int get(int index);
    }
    
    // 감소하는 배열에서의 지수 탐색
    public static int exponentialSearchDescending(int[] arr, int target) {
        if (arr[0] == target) {
            return 0;
        }
        
        int bound = 1;
        while (bound < arr.length && arr[bound] >= target) {
            bound *= 2;
        }
        
        int left = bound / 2;
        int right = Math.min(bound, arr.length - 1);
        
        return binarySearchDescending(arr, target, left, right);
    }
    
    private static int binarySearchDescending(int[] arr, int target, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] > target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
    
    private static int binarySearchInRange(int[] arr, int target, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
}
```

## 9.6 점프 탐색 (Jump Search)

### 블록 단위로 점프하며 탐색

```java
public class JumpSearch {
    
    // 기본 점프 탐색
    public static int jumpSearch(int[] arr, int target) {
        int n = arr.length;
        int step = (int)Math.sqrt(n);
        int prev = 0;
        
        // 블록 단위로 점프
        while (arr[Math.min(step, n) - 1] < target) {
            prev = step;
            step += (int)Math.sqrt(n);
            if (prev >= n) {
                return -1;
            }
        }
        
        // 블록 내에서 선형 탐색
        while (arr[prev] < target) {
            prev++;
            if (prev == Math.min(step, n)) {
                return -1;
            }
        }
        
        if (arr[prev] == target) {
            return prev;
        }
        
        return -1;
    }
    
    // 개선된 점프 탐색 (이진 탐색과 결합)
    public static int improvedJumpSearch(int[] arr, int target) {
        int n = arr.length;
        int step = (int)Math.sqrt(n);
        int prev = 0;
        
        // 블록 단위로 점프
        while (arr[Math.min(step, n) - 1] < target) {
            prev = step;
            step += (int)Math.sqrt(n);
            if (prev >= n) {
                return -1;
            }
        }
        
        // 블록 내에서 이진 탐색
        return binarySearchInRange(arr, target, prev, Math.min(step, n) - 1);
    }
    
    // 가변 점프 크기
    public static int variableJumpSearch(int[] arr, int target) {
        int n = arr.length;
        int jump = 1;
        int prev = 0;
        
        // 피보나치 수열로 점프 크기 증가
        int fib1 = 1, fib2 = 1;
        
        while (prev < n && arr[prev] < target) {
            prev += jump;
            if (prev >= n || arr[prev] >= target) {
                break;
            }
            
            // 다음 피보나치 수
            int nextFib = fib1 + fib2;
            fib1 = fib2;
            fib2 = nextFib;
            jump = nextFib;
        }
        
        // 뒤로 돌아가서 선형 탐색
        prev = Math.max(0, prev - jump);
        while (prev < n && arr[prev] < target) {
            prev++;
        }
        
        if (prev < n && arr[prev] == target) {
            return prev;
        }
        
        return -1;
    }
    
    private static int binarySearchInRange(int[] arr, int target, int left, int right) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
}
```

## 9.7 피보나치 탐색 (Fibonacci Search)

### 피보나치 수열을 이용한 탐색

```java
public class FibonacciSearch {
    
    // 기본 피보나치 탐색
    public static int fibonacciSearch(int[] arr, int target) {
        int n = arr.length;
        
        // 피보나치 수 초기화
        int fib2 = 0;  // (m-2)번째 피보나치 수
        int fib1 = 1;  // (m-1)번째 피보나치 수
        int fibM = fib2 + fib1;  // m번째 피보나치 수
        
        // n보다 크거나 같은 가장 작은 피보나치 수 찾기
        while (fibM < n) {
            fib2 = fib1;
            fib1 = fibM;
            fibM = fib2 + fib1;
        }
        
        int offset = -1;
        
        while (fibM > 1) {
            // 유효한 위치 확인
            int i = Math.min(offset + fib2, n - 1);
            
            if (arr[i] < target) {
                fibM = fib1;
                fib1 = fib2;
                fib2 = fibM - fib1;
                offset = i;
            }
            else if (arr[i] > target) {
                fibM = fib2;
                fib1 = fib1 - fib2;
                fib2 = fibM - fib1;
            }
            else {
                return i;
            }
        }
        
        // 마지막 요소 확인
        if (fib1 == 1 && offset + 1 < n && arr[offset + 1] == target) {
            return offset + 1;
        }
        
        return -1;
    }
    
    // 개선된 피보나치 탐색
    public static int improvedFibonacciSearch(int[] arr, int target) {
        int n = arr.length;
        if (n == 0) return -1;
        
        // 피보나치 수열 미리 계산
        List<Integer> fibs = new ArrayList<>();
        fibs.add(0);
        fibs.add(1);
        
        while (fibs.get(fibs.size() - 1) < n) {
            int nextFib = fibs.get(fibs.size() - 1) + fibs.get(fibs.size() - 2);
            fibs.add(nextFib);
        }
        
        int k = fibs.size() - 1;
        int offset = 0;
        
        while (k > 0) {
            int index = Math.min(offset + fibs.get(k - 1), n - 1);
            
            if (target == arr[index]) {
                return index;
            } else if (target > arr[index]) {
                offset = index;
                k = k - 1;
            } else {
                k = k - 2;
            }
        }
        
        return -1;
    }
}
```

## 9.8 삼진 탐색 (Ternary Search)

### 세 부분으로 나누어 탐색

```java
public class TernarySearch {
    
    // 기본 삼진 탐색
    public static int ternarySearch(int[] arr, int target) {
        return ternarySearch(arr, target, 0, arr.length - 1);
    }
    
    private static int ternarySearch(int[] arr, int target, int left, int right) {
        if (left > right) {
            return -1;
        }
        
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;
        
        if (arr[mid1] == target) {
            return mid1;
        }
        if (arr[mid2] == target) {
            return mid2;
        }
        
        if (target < arr[mid1]) {
            return ternarySearch(arr, target, left, mid1 - 1);
        } else if (target > arr[mid2]) {
            return ternarySearch(arr, target, mid2 + 1, right);
        } else {
            return ternarySearch(arr, target, mid1 + 1, mid2 - 1);
        }
    }
    
    // 반복적 삼진 탐색
    public static int iterativeTernarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left <= right) {
            int mid1 = left + (right - left) / 3;
            int mid2 = right - (right - left) / 3;
            
            if (arr[mid1] == target) {
                return mid1;
            }
            if (arr[mid2] == target) {
                return mid2;
            }
            
            if (target < arr[mid1]) {
                right = mid1 - 1;
            } else if (target > arr[mid2]) {
                left = mid2 + 1;
            } else {
                left = mid1 + 1;
                right = mid2 - 1;
            }
        }
        
        return -1;
    }
    
    // 단봉 함수에서 최댓값 찾기
    public static double findMaximum(Function<Double, Double> f, double left, double right, double epsilon) {
        while (right - left > epsilon) {
            double mid1 = left + (right - left) / 3;
            double mid2 = right - (right - left) / 3;
            
            if (f.apply(mid1) < f.apply(mid2)) {
                left = mid1;
            } else {
                right = mid2;
            }
        }
        
        return (left + right) / 2;
    }
}
```

## 9.9 블록 탐색 (Block Search)

### 인덱스 블록을 사용한 탐색

```java
public class BlockSearch {
    
    static class IndexBlock {
        int maxValue;
        int startIndex;
        int endIndex;
        
        IndexBlock(int maxValue, int startIndex, int endIndex) {
            this.maxValue = maxValue;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
        }
    }
    
    // 블록 탐색 구현
    public static int blockSearch(int[] arr, int target, int blockSize) {
        int n = arr.length;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        // 인덱스 블록 생성
        IndexBlock[] indexBlocks = new IndexBlock[numBlocks];
        
        for (int i = 0; i < numBlocks; i++) {
            int start = i * blockSize;
            int end = Math.min(start + blockSize - 1, n - 1);
            int maxValue = arr[start];
            
            for (int j = start + 1; j <= end; j++) {
                maxValue = Math.max(maxValue, arr[j]);
            }
            
            indexBlocks[i] = new IndexBlock(maxValue, start, end);
        }
        
        // 타겟이 속한 블록 찾기
        int blockIndex = -1;
        for (int i = 0; i < numBlocks; i++) {
            if (target <= indexBlocks[i].maxValue) {
                blockIndex = i;
                break;
            }
        }
        
        if (blockIndex == -1) {
            return -1;
        }
        
        // 블록 내에서 선형 탐색
        IndexBlock block = indexBlocks[blockIndex];
        for (int i = block.startIndex; i <= block.endIndex; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        
        return -1;
    }
    
    // 동적 블록 크기
    public static int adaptiveBlockSearch(int[] arr, int target) {
        int n = arr.length;
        int blockSize = (int)Math.sqrt(n);
        
        // 데이터 분포에 따라 블록 크기 조정
        if (isUniformlyDistributed(arr)) {
            blockSize = (int)Math.sqrt(n / 2);
        }
        
        return blockSearch(arr, target, blockSize);
    }
    
    private static boolean isUniformlyDistributed(int[] arr) {
        if (arr.length < 3) return true;
        
        int diff1 = arr[1] - arr[0];
        for (int i = 2; i < Math.min(10, arr.length); i++) {
            int diff = arr[i] - arr[i-1];
            if (Math.abs(diff - diff1) > diff1 * 0.1) {
                return false;
            }
        }
        
        return true;
    }
}
```

## 9.10 해시 기반 탐색

### 해시 테이블을 이용한 O(1) 탐색

```java
public class HashSearch {
    
    // 해시셋을 이용한 존재 확인
    public static boolean contains(int[] arr, int target) {
        Set<Integer> set = new HashSet<>();
        for (int num : arr) {
            set.add(num);
        }
        return set.contains(target);
    }
    
    // 두 수의 합 문제
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        
        return new int[]{-1, -1};
    }
    
    // 세 수의 합 문제
    public static List<List<Integer>> threeSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            
            Set<Integer> seen = new HashSet<>();
            for (int j = i + 1; j < nums.length; j++) {
                int complement = target - nums[i] - nums[j];
                
                if (seen.contains(complement)) {
                    result.add(Arrays.asList(nums[i], complement, nums[j]));
                    
                    while (j + 1 < nums.length && nums[j] == nums[j + 1]) {
                        j++;
                    }
                }
                seen.add(nums[j]);
            }
        }
        
        return result;
    }
    
    // 부분 배열의 합
    public static boolean subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        
        int sum = 0;
        for (int num : nums) {
            sum += num;
            if (map.containsKey(sum - k)) {
                return true;
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        
        return false;
    }
}
```

## 9.11 문자열 탐색 알고리즘

### 패턴 매칭

```java
public class StringSearch {
    
    // 단순 문자열 탐색
    public static int naiveSearch(String text, String pattern) {
        int n = text.length();
        int m = pattern.length();
        
        for (int i = 0; i <= n - m; i++) {
            int j;
            for (j = 0; j < m; j++) {
                if (text.charAt(i + j) != pattern.charAt(j)) {
                    break;
                }
            }
            if (j == m) {
                return i;
            }
        }
        
        return -1;
    }
    
    // KMP 알고리즘
    public static int kmpSearch(String text, String pattern) {
        int n = text.length();
        int m = pattern.length();
        
        if (m == 0) return 0;
        
        // LPS 배열 생성
        int[] lps = computeLPS(pattern);
        
        int i = 0;  // text의 인덱스
        int j = 0;  // pattern의 인덱스
        
        while (i < n) {
            if (text.charAt(i) == pattern.charAt(j)) {
                i++;
                j++;
            }
            
            if (j == m) {
                return i - j;
            } else if (i < n && text.charAt(i) != pattern.charAt(j)) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return -1;
    }
    
    private static int[] computeLPS(String pattern) {
        int m = pattern.length();
        int[] lps = new int[m];
        int len = 0;
        int i = 1;
        
        while (i < m) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        
        return lps;
    }
    
    // Rabin-Karp 알고리즘
    public static int rabinKarpSearch(String text, String pattern) {
        int n = text.length();
        int m = pattern.length();
        int prime = 101;
        int d = 256;  // 문자의 개수
        
        int patternHash = 0;
        int textHash = 0;
        int h = 1;
        
        // h = d^(m-1) % prime
        for (int i = 0; i < m - 1; i++) {
            h = (h * d) % prime;
        }
        
        // 패턴과 텍스트의 첫 윈도우 해시 계산
        for (int i = 0; i < m; i++) {
            patternHash = (d * patternHash + pattern.charAt(i)) % prime;
            textHash = (d * textHash + text.charAt(i)) % prime;
        }
        
        // 슬라이딩 윈도우
        for (int i = 0; i <= n - m; i++) {
            if (patternHash == textHash) {
                // 실제 문자 비교
                int j;
                for (j = 0; j < m; j++) {
                    if (text.charAt(i + j) != pattern.charAt(j)) {
                        break;
                    }
                }
                if (j == m) {
                    return i;
                }
            }
            
            // 다음 윈도우의 해시 계산
            if (i < n - m) {
                textHash = (d * (textHash - text.charAt(i) * h) + text.charAt(i + m)) % prime;
                if (textHash < 0) {
                    textHash += prime;
                }
            }
        }
        
        return -1;
    }
}
```

## 9.12 실습 문제

### 문제 1: 첫 번째와 마지막 위치
정렬된 배열에서 target의 첫 번째와 마지막 위치를 찾으세요.

### 문제 2: 2D 행렬 탐색
행과 열이 모두 정렬된 2D 행렬에서 target을 찾으세요.

### 문제 3: 회전된 배열의 최솟값
회전된 정렬 배열에서 최솟값을 찾으세요.

### 문제 4: 검색 자동완성
주어진 접두사로 시작하는 모든 단어를 효율적으로 찾는 시스템을 구현하세요.

## 9.13 요약

이 장에서는 다양한 탐색 알고리즘에 대해 학습했습니다:

1. **선형 탐색**: O(n) - 단순하지만 비효율적
2. **이진 탐색**: O(log n) - 정렬된 데이터에서 효율적
3. **보간 탐색**: O(log log n) - 균등 분포 데이터에 최적
4. **지수 탐색**: O(log n) - 무한 배열에 유용
5. **점프 탐색**: O(√n) - 블록 단위 탐색
6. **해시 탐색**: O(1) - 가장 빠르지만 추가 공간 필요

적절한 탐색 알고리즘의 선택은 데이터의 특성과 요구사항에 따라 달라집니다. 다음 장에서는 더 복잡한 고급 자료구조에 대해 알아보겠습니다.