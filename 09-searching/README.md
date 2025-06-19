# Chapter 9: 탐색 알고리즘 (Searching Algorithms)

## 9.1 탐색이란?

### 정의
탐색(Search)은 데이터 집합에서 특정 값이나 조건을 만족하는 요소를 찾는 과정입니다. 효율적인 탐색은 많은 응용 프로그램의 핵심 연산입니다.

### 탐색 알고리즘의 분류
1. **순차 탐색**: 처음부터 끝까지 순서대로 확인
2. **이진 탐색**: 정렬된 데이터에서 반씩 나누어 탐색
3. **해시 탐색**: 해시 함수를 이용한 직접 접근
4. **트리 탐색**: 트리 구조를 이용한 탐색
5. **그래프 탐색**: BFS, DFS 등

### 탐색 알고리즘의 성능 평가
- **시간 복잡도**: 탐색에 필요한 비교 횟수
- **공간 복잡도**: 추가 메모리 사용량
- **전처리 시간**: 탐색을 위한 준비 시간
- **데이터 구조**: 필요한 자료구조의 복잡성

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
            if (arr[i].equals(target)) {
                return i;
            }
        }
        return -1;
    }
    
    // 조건을 만족하는 첫 번째 요소 찾기
    public static <T> int linearSearchWithPredicate(T[] arr, Predicate<T> predicate) {
        for (int i = 0; i < arr.length; i++) {
            if (predicate.test(arr[i])) {
                return i;
            }
        }
        return -1;
    }
    
    // 모든 일치하는 인덱스 찾기
    public static List<Integer> linearSearchAll(int[] arr, int target) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                indices.add(i);
            }
        }
        return indices;
    }
    
    // 마지막 일치하는 인덱스 찾기
    public static int linearSearchLast(int[] arr, int target) {
        for (int i = arr.length - 1; i >= 0; i--) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
    
    // 센티널 선형 탐색 (약간의 최적화)
    public static int sentinelLinearSearch(int[] arr, int target) {
        int n = arr.length;
        int last = arr[n - 1];
        
        // 마지막 요소를 타겟으로 설정 (센티널)
        arr[n - 1] = target;
        
        int i = 0;
        while (arr[i] != target) {
            i++;
        }
        
        // 원래 값 복원
        arr[n - 1] = last;
        
        // 찾았는지 확인
        if (i < n - 1 || arr[n - 1] == target) {
            return i;
        }
        
        return -1;
    }
    
    // 정렬된 배열에서의 선형 탐색 (조기 종료)
    public static int orderedLinearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
            if (arr[i] > target) {
                break; // 타겟보다 큰 값을 만나면 종료
            }
        }
        return -1;
    }
    
    // 자체 조직화 선형 탐색 (Move to Front)
    public static class SelfOrganizingList {
        private Node head;
        
        private class Node {
            int data;
            Node next;
            
            Node(int data) {
                this.data = data;
                this.next = null;
            }
        }
        
        public void insert(int data) {
            Node newNode = new Node(data);
            newNode.next = head;
            head = newNode;
        }
        
        public boolean search(int target) {
            Node prev = null;
            Node current = head;
            
            while (current != null) {
                if (current.data == target) {
                    // Move to front
                    if (prev != null) {
                        prev.next = current.next;
                        current.next = head;
                        head = current;
                    }
                    return true;
                }
                prev = current;
                current = current.next;
            }
            
            return false;
        }
    }
    
    // 근사 선형 탐색 (가장 가까운 값 찾기)
    public static int approximateLinearSearch(double[] arr, double target) {
        if (arr.length == 0) return -1;
        
        int closestIndex = 0;
        double minDifference = Math.abs(arr[0] - target);
        
        for (int i = 1; i < arr.length; i++) {
            double difference = Math.abs(arr[i] - target);
            if (difference < minDifference) {
                minDifference = difference;
                closestIndex = i;
            }
        }
        
        return closestIndex;
    }
}
```

## 9.3 이진 탐색 (Binary Search)

### 기본 이진 탐색
정렬된 배열에서 중간값과 비교하여 탐색 범위를 절반으로 줄여가는 알고리즘입니다.

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
    
    // 첫 번째 발생 위치 찾기
    public static int binarySearchFirst(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                result = mid;
                right = mid - 1; // 왼쪽에서 계속 탐색
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    // 마지막 발생 위치 찾기
    public static int binarySearchLast(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        int result = -1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                result = mid;
                left = mid + 1; // 오른쪽에서 계속 탐색
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    // 범위 탐색 (첫 번째와 마지막 위치)
    public static int[] searchRange(int[] arr, int target) {
        int first = binarySearchFirst(arr, target);
        if (first == -1) {
            return new int[]{-1, -1};
        }
        
        int last = binarySearchLast(arr, target);
        return new int[]{first, last};
    }
    
    // Lower Bound (target 이상인 첫 번째 위치)
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
    
    // Upper Bound (target 초과인 첫 번째 위치)
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
    
    // 가장 가까운 값 찾기
    public static int binarySearchClosest(int[] arr, int target) {
        if (arr.length == 0) return -1;
        
        int left = 0;
        int right = arr.length - 1;
        
        // target이 범위를 벗어난 경우
        if (target <= arr[left]) return left;
        if (target >= arr[right]) return right;
        
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
        
        // left와 right가 교차한 후, 더 가까운 값 반환
        if (Math.abs(arr[left] - target) < Math.abs(arr[right] - target)) {
            return left;
        } else {
            return right;
        }
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

### 보간 탐색의 원리
균등하게 분포된 정렬 배열에서 타겟의 예상 위치를 계산하여 탐색합니다.

```java
public class InterpolationSearch {
    
    // 기본 보간 탐색
    public static int interpolationSearch(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        
        while (low <= high && target >= arr[low] && target <= arr[high]) {
            // 배열에 하나의 요소만 있는 경우
            if (low == high) {
                if (arr[low] == target) return low;
                return -1;
            }
            
            // 보간 공식을 사용하여 예상 위치 계산
            int pos = low + ((target - arr[low]) * (high - low)) / 
                           (arr[high] - arr[low]);
            
            if (arr[pos] == target) {
                return pos;
            } else if (arr[pos] < target) {
                low = pos + 1;
            } else {
                high = pos - 1;
            }
        }
        
        return -1;
    }
    
    // 재귀적 보간 탐색
    public static int recursiveInterpolationSearch(int[] arr, int target) {
        return recursiveInterpolationSearch(arr, 0, arr.length - 1, target);
    }
    
    private static int recursiveInterpolationSearch(int[] arr, int low, int high, int target) {
        if (low > high || target < arr[low] || target > arr[high]) {
            return -1;
        }
        
        if (low == high) {
            return arr[low] == target ? low : -1;
        }
        
        int pos = low + ((target - arr[low]) * (high - low)) / 
                       (arr[high] - arr[low]);
        
        if (arr[pos] == target) {
            return pos;
        } else if (arr[pos] < target) {
            return recursiveInterpolationSearch(arr, pos + 1, high, target);
        } else {
            return recursiveInterpolationSearch(arr, low, pos - 1, target);
        }
    }
    
    // 개선된 보간 탐색 (경계 검사 추가)
    public static int improvedInterpolationSearch(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        int iterations = 0;
        int maxIterations = (int)(Math.log(arr.length) / Math.log(2)) + 1;
        
        while (low <= high && target >= arr[low] && target <= arr[high]) {
            iterations++;
            
            // 무한 루프 방지
            if (iterations > maxIterations) {
                // 이진 탐색으로 전환
                return binarySearchFallback(arr, low, high, target);
            }
            
            if (low == high) {
                return arr[low] == target ? low : -1;
            }
            
            // 0으로 나누기 방지
            if (arr[high] == arr[low]) {
                return arr[low] == target ? low : -1;
            }
            
            int pos = low + ((target - arr[low]) * (high - low)) / 
                           (arr[high] - arr[low]);
            
            // 범위 검증
            pos = Math.max(low, Math.min(high, pos));
            
            if (arr[pos] == target) {
                return pos;
            } else if (arr[pos] < target) {
                low = pos + 1;
            } else {
                high = pos - 1;
            }
        }
        
        return -1;
    }
    
    private static int binarySearchFallback(int[] arr, int low, int high, int target) {
        while (low <= high) {
            int mid = low + (high - low) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        return -1;
    }
    
    // 실수 배열에서의 보간 탐색
    public static int interpolationSearchDouble(double[] arr, double target, double epsilon) {
        int low = 0;
        int high = arr.length - 1;
        
        while (low <= high && target >= arr[low] && target <= arr[high]) {
            if (low == high) {
                if (Math.abs(arr[low] - target) <= epsilon) {
                    return low;
                }
                return -1;
            }
            
            // 보간 공식
            int pos = (int)(low + ((target - arr[low]) * (high - low)) / 
                                 (arr[high] - arr[low]));
            
            pos = Math.max(low, Math.min(high, pos));
            
            if (Math.abs(arr[pos] - target) <= epsilon) {
                return pos;
            } else if (arr[pos] < target) {
                low = pos + 1;
            } else {
                high = pos - 1;
            }
        }
        
        return -1;
    }
}
```

## 9.5 지수 탐색 (Exponential Search)

### 지수 탐색의 원리
범위를 지수적으로 증가시켜 타겟이 있을 범위를 찾은 후 이진 탐색을 수행합니다.

```java
public class ExponentialSearch {
    
    // 기본 지수 탐색
    public static int exponentialSearch(int[] arr, int target) {
        int n = arr.length;
        
        // 첫 번째 요소 확인
        if (arr[0] == target) {
            return 0;
        }
        
        // 범위 찾기
        int i = 1;
        while (i < n && arr[i] <= target) {
            i *= 2;
        }
        
        // 이진 탐색 수행
        return binarySearchInRange(arr, target, i / 2, Math.min(i, n - 1));
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
    
    // 무한 배열에서의 탐색
    public static int searchInfiniteArray(InfiniteArray arr, int target) {
        // 범위 찾기
        int low = 0;
        int high = 1;
        
        while (arr.get(high) < target) {
            low = high;
            high *= 2;
        }
        
        // 이진 탐색
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int midVal = arr.get(mid);
            
            if (midVal == target) {
                return mid;
            } else if (midVal < target) {
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
    
    // 개선된 지수 탐색 (양방향)
    public static int bidirectionalExponentialSearch(int[] arr, int target) {
        int n = arr.length;
        int mid = n / 2;
        
        // 중간값 확인
        if (arr[mid] == target) {
            return mid;
        }
        
        // 타겟이 중간값보다 작으면 왼쪽에서 탐색
        if (target < arr[mid]) {
            int i = mid / 2;
            while (i >= 0 && arr[i] > target) {
                i /= 2;
            }
            return binarySearchInRange(arr, target, i, mid - 1);
        } 
        // 타겟이 중간값보다 크면 오른쪽에서 탐색
        else {
            int i = mid + (n - mid) / 2;
            while (i < n && arr[i] < target) {
                int next = i + (n - i) / 2;
                if (next == i) break;
                i = next;
            }
            return binarySearchInRange(arr, target, mid + 1, Math.min(i, n - 1));
        }
    }
    
    // 피보나치 탐색
    public static int fibonacciSearch(int[] arr, int target) {
        int n = arr.length;
        
        // 피보나치 수 초기화
        int fib2 = 0; // (m-2)번째 피보나치 수
        int fib1 = 1; // (m-1)번째 피보나치 수
        int fibM = fib2 + fib1; // m번째 피보나치 수
        
        // n보다 크거나 같은 가장 작은 피보나치 수 찾기
        while (fibM < n) {
            fib2 = fib1;
            fib1 = fibM;
            fibM = fib2 + fib1;
        }
        
        int offset = -1;
        
        while (fibM > 1) {
            int i = Math.min(offset + fib2, n - 1);
            
            if (arr[i] < target) {
                fibM = fib1;
                fib1 = fib2;
                fib2 = fibM - fib1;
                offset = i;
            } else if (arr[i] > target) {
                fibM = fib2;
                fib1 = fib1 - fib2;
                fib2 = fibM - fib1;
            } else {
                return i;
            }
        }
        
        // 마지막 요소 확인
        if (fib1 == 1 && offset + 1 < n && arr[offset + 1] == target) {
            return offset + 1;
        }
        
        return -1;
    }
}
```

## 9.6 점프 탐색 (Jump Search)

### 점프 탐색의 원리
일정한 간격으로 점프하면서 타겟을 포함하는 블록을 찾은 후 선형 탐색합니다.

```java
public class JumpSearch {
    
    // 기본 점프 탐색
    public static int jumpSearch(int[] arr, int target) {
        int n = arr.length;
        int step = (int)Math.sqrt(n);
        int prev = 0;
        
        // 타겟을 포함하는 블록 찾기
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
    
    // 최적화된 점프 탐색 (가변 스텝)
    public static int optimizedJumpSearch(int[] arr, int target) {
        int n = arr.length;
        int step = (int)Math.sqrt(n);
        int prev = 0;
        
        // 점프 단계
        while (prev < n && arr[prev] < target) {
            int next = Math.min(prev + step, n - 1);
            
            if (arr[next] >= target) {
                // 이진 탐색으로 정확한 위치 찾기
                return binarySearchInRange(arr, target, prev, next);
            }
            
            prev = next;
            
            // 동적 스텝 조정
            if (prev < n / 2) {
                step = (int)Math.sqrt(n - prev);
            }
        }
        
        return prev < n && arr[prev] == target ? prev : -1;
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
    
    // 역방향 점프 탐색
    public static int reverseJumpSearch(int[] arr, int target) {
        int n = arr.length;
        int step = (int)Math.sqrt(n);
        int current = n - 1;
        
        // 뒤에서부터 점프
        while (current >= 0 && arr[current] > target) {
            current -= step;
        }
        
        if (current < 0) {
            current = 0;
        }
        
        // 선형 탐색
        while (current < n && arr[current] <= target) {
            if (arr[current] == target) {
                return current;
            }
            current++;
        }
        
        return -1;
    }
    
    // 2차원 배열에서의 점프 탐색
    public static int[] jumpSearch2D(int[][] matrix, int target) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int stepRow = (int)Math.sqrt(rows);
        int stepCol = (int)Math.sqrt(cols);
        
        // 행 단위로 점프
        int row = 0;
        while (row < rows && matrix[row][cols - 1] < target) {
            row += stepRow;
        }
        
        if (row >= rows) {
            row = rows - 1;
        }
        
        // 해당 행에서 열 단위로 점프
        int startRow = Math.max(0, row - stepRow + 1);
        for (int r = startRow; r <= row && r < rows; r++) {
            int col = 0;
            while (col < cols && matrix[r][col] < target) {
                col += stepCol;
            }
            
            // 블록 내에서 선형 탐색
            int startCol = Math.max(0, col - stepCol + 1);
            for (int c = startCol; c <= col && c < cols; c++) {
                if (matrix[r][c] == target) {
                    return new int[]{r, c};
                }
            }
        }
        
        return new int[]{-1, -1};
    }
}
```

## 9.7 삼진 탐색 (Ternary Search)

### 삼진 탐색의 원리
정렬된 배열을 3등분하여 탐색 범위를 줄여가는 알고리즘입니다.

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
    
    // 단봉 배열에서 최댓값 찾기
    public static int findPeakTernary(int[] arr) {
        int left = 0;
        int right = arr.length - 1;
        
        while (left < right) {
            int mid1 = left + (right - left) / 3;
            int mid2 = right - (right - left) / 3;
            
            if (arr[mid1] < arr[mid2]) {
                left = mid1 + 1;
            } else {
                right = mid2 - 1;
            }
        }
        
        return left;
    }
    
    // 일반화된 k-진 탐색
    public static int kArySearch(int[] arr, int target, int k) {
        return kArySearchHelper(arr, target, 0, arr.length - 1, k);
    }
    
    private static int kArySearchHelper(int[] arr, int target, int left, int right, int k) {
        if (left > right) {
            return -1;
        }
        
        if (left == right) {
            return arr[left] == target ? left : -1;
        }
        
        // k개의 분할점 계산
        int[] points = new int[k - 1];
        for (int i = 0; i < k - 1; i++) {
            points[i] = left + (i + 1) * (right - left) / k;
        }
        
        // 각 분할점에서 확인
        for (int i = 0; i < k - 1; i++) {
            if (arr[points[i]] == target) {
                return points[i];
            }
        }
        
        // 적절한 구간 선택
        if (target < arr[points[0]]) {
            return kArySearchHelper(arr, target, left, points[0] - 1, k);
        }
        
        for (int i = 0; i < k - 2; i++) {
            if (target > arr[points[i]] && target < arr[points[i + 1]]) {
                return kArySearchHelper(arr, target, points[i] + 1, points[i + 1] - 1, k);
            }
        }
        
        return kArySearchHelper(arr, target, points[k - 2] + 1, right, k);
    }
}
```

## 9.8 해시 탐색 (Hash Search)

### 해시 테이블을 이용한 탐색
해시 함수를 통해 O(1) 평균 시간에 탐색을 수행합니다.

```java
public class HashSearch {
    
    // 간단한 해시 테이블 구현
    static class SimpleHashTable {
        private static class Entry {
            int key;
            String value;
            Entry next;
            
            Entry(int key, String value) {
                this.key = key;
                this.value = value;
            }
        }
        
        private Entry[] table;
        private int size;
        
        public SimpleHashTable(int capacity) {
            table = new Entry[capacity];
            size = 0;
        }
        
        private int hash(int key) {
            return Math.abs(key % table.length);
        }
        
        public void put(int key, String value) {
            int index = hash(key);
            Entry newEntry = new Entry(key, value);
            
            if (table[index] == null) {
                table[index] = newEntry;
            } else {
                Entry current = table[index];
                while (current.next != null) {
                    if (current.key == key) {
                        current.value = value;
                        return;
                    }
                    current = current.next;
                }
                if (current.key == key) {
                    current.value = value;
                } else {
                    current.next = newEntry;
                }
            }
            size++;
        }
        
        public String search(int key) {
            int index = hash(key);
            Entry current = table[index];
            
            while (current != null) {
                if (current.key == key) {
                    return current.value;
                }
                current = current.next;
            }
            
            return null;
        }
    }
    
    // Cuckoo 해싱
    static class CuckooHashTable {
        private int[] table1;
        private int[] table2;
        private int size;
        private int capacity;
        private static final int MAX_ITERATIONS = 100;
        private static final int EMPTY = Integer.MIN_VALUE;
        
        public CuckooHashTable(int capacity) {
            this.capacity = capacity;
            this.table1 = new int[capacity];
            this.table2 = new int[capacity];
            Arrays.fill(table1, EMPTY);
            Arrays.fill(table2, EMPTY);
            this.size = 0;
        }
        
        private int hash1(int key) {
            return Math.abs(key % capacity);
        }
        
        private int hash2(int key) {
            return Math.abs((key / capacity) % capacity);
        }
        
        public boolean insert(int key) {
            if (search(key)) {
                return false; // 이미 존재
            }
            
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                int pos1 = hash1(key);
                if (table1[pos1] == EMPTY) {
                    table1[pos1] = key;
                    size++;
                    return true;
                }
                
                // Swap and try table2
                int temp = table1[pos1];
                table1[pos1] = key;
                key = temp;
                
                int pos2 = hash2(key);
                if (table2[pos2] == EMPTY) {
                    table2[pos2] = key;
                    size++;
                    return true;
                }
                
                // Swap and continue
                temp = table2[pos2];
                table2[pos2] = key;
                key = temp;
            }
            
            // Rehashing needed
            rehash();
            return insert(key);
        }
        
        public boolean search(int key) {
            return table1[hash1(key)] == key || table2[hash2(key)] == key;
        }
        
        private void rehash() {
            int[] oldTable1 = table1;
            int[] oldTable2 = table2;
            
            capacity *= 2;
            table1 = new int[capacity];
            table2 = new int[capacity];
            Arrays.fill(table1, EMPTY);
            Arrays.fill(table2, EMPTY);
            size = 0;
            
            for (int key : oldTable1) {
                if (key != EMPTY) {
                    insert(key);
                }
            }
            
            for (int key : oldTable2) {
                if (key != EMPTY) {
                    insert(key);
                }
            }
        }
    }
    
    // Robin Hood 해싱
    static class RobinHoodHashTable {
        private static class Entry {
            int key;
            String value;
            int distance; // 이상적인 위치로부터의 거리
            
            Entry(int key, String value, int distance) {
                this.key = key;
                this.value = value;
                this.distance = distance;
            }
        }
        
        private Entry[] table;
        private int size;
        private int capacity;
        private static final Entry DELETED = new Entry(-1, null, -1);
        
        public RobinHoodHashTable(int capacity) {
            this.capacity = capacity;
            this.table = new Entry[capacity];
            this.size = 0;
        }
        
        private int hash(int key) {
            return Math.abs(key % capacity);
        }
        
        public void insert(int key, String value) {
            int index = hash(key);
            int distance = 0;
            Entry newEntry = new Entry(key, value, distance);
            
            while (table[index] != null && table[index] != DELETED) {
                if (table[index].key == key) {
                    table[index].value = value;
                    return;
                }
                
                // Robin Hood: 가난한 것을 우선
                if (table[index].distance < distance) {
                    Entry temp = table[index];
                    table[index] = newEntry;
                    newEntry = temp;
                    distance = newEntry.distance;
                }
                
                index = (index + 1) % capacity;
                distance++;
                newEntry.distance = distance;
            }
            
            table[index] = newEntry;
            size++;
            
            if (size > capacity * 0.9) {
                resize();
            }
        }
        
        public String search(int key) {
            int index = hash(key);
            int distance = 0;
            
            while (table[index] != null) {
                if (table[index] != DELETED && table[index].key == key) {
                    return table[index].value;
                }
                
                // 현재 거리가 저장된 거리보다 크면 존재하지 않음
                if (table[index].distance < distance) {
                    break;
                }
                
                index = (index + 1) % capacity;
                distance++;
            }
            
            return null;
        }
        
        private void resize() {
            Entry[] oldTable = table;
            capacity *= 2;
            table = new Entry[capacity];
            size = 0;
            
            for (Entry entry : oldTable) {
                if (entry != null && entry != DELETED) {
                    insert(entry.key, entry.value);
                }
            }
        }
    }
}
```

## 9.9 특수 탐색 알고리즘

### 패턴 매칭 알고리즘

```java
public class PatternMatching {
    
    // KMP (Knuth-Morris-Pratt) 알고리즘
    public static List<Integer> kmpSearch(String text, String pattern) {
        List<Integer> matches = new ArrayList<>();
        int n = text.length();
        int m = pattern.length();
        
        if (m == 0) return matches;
        
        // LPS 배열 구성
        int[] lps = computeLPS(pattern);
        
        int i = 0; // text index
        int j = 0; // pattern index
        
        while (i < n) {
            if (text.charAt(i) == pattern.charAt(j)) {
                i++;
                j++;
            }
            
            if (j == m) {
                matches.add(i - j);
                j = lps[j - 1];
            } else if (i < n && text.charAt(i) != pattern.charAt(j)) {
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
        
        return matches;
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
    
    // Boyer-Moore 알고리즘
    public static List<Integer> boyerMooreSearch(String text, String pattern) {
        List<Integer> matches = new ArrayList<>();
        int n = text.length();
        int m = pattern.length();
        
        if (m == 0) return matches;
        
        // Bad character 테이블
        int[] badChar = buildBadCharTable(pattern);
        
        int s = 0; // shift
        while (s <= n - m) {
            int j = m - 1;
            
            while (j >= 0 && pattern.charAt(j) == text.charAt(s + j)) {
                j--;
            }
            
            if (j < 0) {
                matches.add(s);
                s += (s + m < n) ? m - badChar[text.charAt(s + m)] : 1;
            } else {
                s += Math.max(1, j - badChar[text.charAt(s + j)]);
            }
        }
        
        return matches;
    }
    
    private static int[] buildBadCharTable(String pattern) {
        int[] badChar = new int[256];
        Arrays.fill(badChar, -1);
        
        for (int i = 0; i < pattern.length(); i++) {
            badChar[pattern.charAt(i)] = i;
        }
        
        return badChar;
    }
    
    // Rabin-Karp 알고리즘
    public static List<Integer> rabinKarpSearch(String text, String pattern) {
        List<Integer> matches = new ArrayList<>();
        int n = text.length();
        int m = pattern.length();
        
        if (m == 0 || m > n) return matches;
        
        long patternHash = 0;
        long textHash = 0;
        long h = 1;
        int prime = 101;
        int d = 256; // 문자 집합 크기
        
        // h = d^(m-1) % prime
        for (int i = 0; i < m - 1; i++) {
            h = (h * d) % prime;
        }
        
        // 초기 해시값 계산
        for (int i = 0; i < m; i++) {
            patternHash = (d * patternHash + pattern.charAt(i)) % prime;
            textHash = (d * textHash + text.charAt(i)) % prime;
        }
        
        // 슬라이딩 윈도우
        for (int i = 0; i <= n - m; i++) {
            if (patternHash == textHash) {
                // 실제 문자열 비교
                if (text.substring(i, i + m).equals(pattern)) {
                    matches.add(i);
                }
            }
            
            if (i < n - m) {
                textHash = (d * (textHash - text.charAt(i) * h) + text.charAt(i + m)) % prime;
                if (textHash < 0) {
                    textHash += prime;
                }
            }
        }
        
        return matches;
    }
}
```

### 이차원 배열 탐색

```java
public class Matrix2DSearch {
    
    // 행과 열이 모두 정렬된 2D 행렬에서 탐색
    public static boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        int row = 0;
        int col = cols - 1;
        
        // 오른쪽 위에서 시작
        while (row < rows && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                col--;
            } else {
                row++;
            }
        }
        
        return false;
    }
    
    // Saddleback Search
    public static int[] saddlebackSearch(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return new int[]{-1, -1};
        }
        
        int rows = matrix.length;
        int cols = matrix[0].length;
        int row = rows - 1;
        int col = 0;
        
        // 왼쪽 아래에서 시작
        while (row >= 0 && col < cols) {
            if (matrix[row][col] == target) {
                return new int[]{row, col};
            } else if (matrix[row][col] > target) {
                row--;
            } else {
                col++;
            }
        }
        
        return new int[]{-1, -1};
    }
    
    // 분할 정복을 이용한 2D 탐색
    public static boolean divideConquerSearch(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        
        return searchHelper(matrix, target, 0, 0, 
                          matrix.length - 1, matrix[0].length - 1);
    }
    
    private static boolean searchHelper(int[][] matrix, int target,
                                      int rowStart, int colStart,
                                      int rowEnd, int colEnd) {
        if (rowStart > rowEnd || colStart > colEnd) {
            return false;
        }
        
        if (target < matrix[rowStart][colStart] || 
            target > matrix[rowEnd][colEnd]) {
            return false;
        }
        
        int midRow = rowStart + (rowEnd - rowStart) / 2;
        int midCol = colStart + (colEnd - colStart) / 2;
        
        if (matrix[midRow][midCol] == target) {
            return true;
        } else if (matrix[midRow][midCol] > target) {
            return searchHelper(matrix, target, rowStart, colStart, midRow - 1, colEnd) ||
                   searchHelper(matrix, target, midRow, colStart, rowEnd, midCol - 1);
        } else {
            return searchHelper(matrix, target, rowStart, midCol + 1, midRow, colEnd) ||
                   searchHelper(matrix, target, midRow + 1, colStart, rowEnd, colEnd);
        }
    }
}
```

## 9.10 실습 문제

### 문제 1: 회전된 정렬 배열의 최솟값
회전된 정렬 배열에서 최솟값을 O(log n) 시간에 찾으세요.

### 문제 2: 두 정렬 배열의 중간값
크기가 다른 두 정렬 배열의 중간값을 O(log(min(m,n))) 시간에 찾으세요.

### 문제 3: k번째 가장 가까운 점
원점에서 k번째로 가까운 점들을 효율적으로 찾으세요.

### 문제 4: 스카이라인 문제
건물들의 윤곽선을 효율적으로 계산하세요.

## 9.11 요약

이 장에서는 다양한 탐색 알고리즘에 대해 학습했습니다:

1. **선형 탐색**: O(n) - 단순하지만 정렬되지 않은 데이터에 유용
2. **이진 탐색**: O(log n) - 정렬된 데이터에서 매우 효율적
3. **보간 탐색**: O(log log n) - 균등 분포 데이터에 최적
4. **지수 탐색**: O(log n) - 무한 배열이나 큰 배열에 유용
5. **해시 탐색**: O(1) 평균 - 빠른 검색이 필요할 때
6. **패턴 매칭**: 문자열 검색에 특화된 알고리즘들

적절한 탐색 알고리즘의 선택은 데이터의 특성과 요구사항에 따라 달라집니다. 다음 장에서는 더 복잡한 고급 자료구조에 대해 알아보겠습니다.