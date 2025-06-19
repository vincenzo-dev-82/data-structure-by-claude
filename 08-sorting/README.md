# Chapter 8: 정렬 알고리즘 (Sorting Algorithms)

## 8.1 정렬이란?

### 정의
정렬은 데이터를 특정 순서(오름차순 또는 내림차순)로 재배열하는 과정입니다. 정렬은 검색, 데이터 분석, 중복 제거 등 많은 알고리즘의 전처리 단계로 사용됩니다.

### 정렬 알고리즘의 분류
1. **비교 기반 정렬**: 요소 간 비교를 통해 정렬
   - 버블 정렬, 선택 정렬, 삽입 정렬, 병합 정렬, 퀵 정렬, 힙 정렬
2. **비비교 기반 정렬**: 요소의 특성을 이용한 정렬
   - 계수 정렬, 기수 정렬, 버킷 정렬

### 정렬 알고리즘의 특성
- **시간 복잡도**: 평균/최선/최악의 경우
- **공간 복잡도**: 추가 메모리 사용량
- **안정성(Stability)**: 같은 값의 상대적 순서 유지
- **제자리 정렬(In-place)**: 추가 메모리 거의 사용 안 함
- **적응적(Adaptive)**: 부분 정렬된 데이터에 효율적

## 8.2 기초 정렬 알고리즘

### 버블 정렬 (Bubble Sort)
인접한 두 요소를 비교하여 정렬하는 가장 단순한 정렬 알고리즘입니다.

```java
public class BubbleSort {
    
    // 기본 버블 정렬
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    // swap
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
    
    // 최적화된 버블 정렬 (조기 종료)
    public static void optimizedBubbleSort(int[] arr) {
        int n = arr.length;
        boolean swapped;
        
        for (int i = 0; i < n - 1; i++) {
            swapped = false;
            
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    swapped = true;
                }
            }
            
            // 교환이 없었다면 이미 정렬됨
            if (!swapped) {
                break;
            }
        }
    }
    
    // 칵테일 정렬 (양방향 버블 정렬)
    public static void cocktailSort(int[] arr) {
        boolean swapped = true;
        int start = 0;
        int end = arr.length - 1;
        
        while (swapped) {
            swapped = false;
            
            // 왼쪽에서 오른쪽으로
            for (int i = start; i < end; i++) {
                if (arr[i] > arr[i + 1]) {
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                    swapped = true;
                }
            }
            
            if (!swapped) break;
            
            end--;
            swapped = false;
            
            // 오른쪽에서 왼쪽으로
            for (int i = end; i > start; i--) {
                if (arr[i] < arr[i - 1]) {
                    int temp = arr[i];
                    arr[i] = arr[i - 1];
                    arr[i - 1] = temp;
                    swapped = true;
                }
            }
            
            start++;
        }
    }
    
    // 제네릭 버블 정렬
    public static <T extends Comparable<T>> void bubbleSort(T[] arr) {
        int n = arr.length;
        
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j].compareTo(arr[j + 1]) > 0) {
                    T temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}
```

### 선택 정렬 (Selection Sort)
매번 최솟값을 찾아 앞으로 이동시키는 정렬 알고리즘입니다.

```java
public class SelectionSort {
    
    // 기본 선택 정렬
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            
            // 최솟값 찾기
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            
            // 최솟값과 교환
            if (minIdx != i) {
                int temp = arr[i];
                arr[i] = arr[minIdx];
                arr[minIdx] = temp;
            }
        }
    }
    
    // 이중 선택 정렬 (최소/최대 동시 선택)
    public static void doubleSelectionSort(int[] arr) {
        int n = arr.length;
        int left = 0;
        int right = n - 1;
        
        while (left < right) {
            int minIdx = left;
            int maxIdx = right;
            
            // 구간의 최소값과 최대값 찾기
            for (int i = left; i <= right; i++) {
                if (arr[i] < arr[minIdx]) {
                    minIdx = i;
                }
                if (arr[i] > arr[maxIdx]) {
                    maxIdx = i;
                }
            }
            
            // 최소값을 왼쪽 끝으로
            if (minIdx != left) {
                int temp = arr[left];
                arr[left] = arr[minIdx];
                arr[minIdx] = temp;
            }
            
            // 최대값이 왼쪽 끝에 있었다면 위치 조정
            if (maxIdx == left) {
                maxIdx = minIdx;
            }
            
            // 최대값을 오른쪽 끝으로
            if (maxIdx != right) {
                int temp = arr[right];
                arr[right] = arr[maxIdx];
                arr[maxIdx] = temp;
            }
            
            left++;
            right--;
        }
    }
    
    // 안정적인 선택 정렬
    public static void stableSelectionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            
            // 최솟값을 앞으로 이동 (shift 방식)
            int minValue = arr[minIdx];
            while (minIdx > i) {
                arr[minIdx] = arr[minIdx - 1];
                minIdx--;
            }
            arr[i] = minValue;
        }
    }
}
```

### 삽입 정렬 (Insertion Sort)
정렬된 부분에 새로운 요소를 적절한 위치에 삽입하는 정렬 알고리즘입니다.

```java
public class InsertionSort {
    
    // 기본 삽입 정렬
    public static void insertionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            
            // key보다 큰 요소들을 오른쪽으로 이동
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            
            arr[j + 1] = key;
        }
    }
    
    // 이진 삽입 정렬
    public static void binaryInsertionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int left = 0;
            int right = i;
            
            // 이진 탐색으로 삽입 위치 찾기
            while (left < right) {
                int mid = (left + right) / 2;
                if (arr[mid] > key) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            
            // 요소들을 오른쪽으로 이동
            for (int j = i - 1; j >= left; j--) {
                arr[j + 1] = arr[j];
            }
            
            arr[left] = key;
        }
    }
    
    // 재귀적 삽입 정렬
    public static void recursiveInsertionSort(int[] arr, int n) {
        if (n <= 1) {
            return;
        }
        
        // 처음 n-1개 요소를 정렬
        recursiveInsertionSort(arr, n - 1);
        
        // 마지막 요소를 정렬된 배열에 삽입
        int last = arr[n - 1];
        int j = n - 2;
        
        while (j >= 0 && arr[j] > last) {
            arr[j + 1] = arr[j];
            j--;
        }
        
        arr[j + 1] = last;
    }
    
    // 연결 리스트를 위한 삽입 정렬
    public static ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = head;
        
        while (current != null) {
            ListNode next = current.next;
            ListNode prev = dummy;
            
            // 삽입 위치 찾기
            while (prev.next != null && prev.next.val < current.val) {
                prev = prev.next;
            }
            
            // 노드 삽입
            current.next = prev.next;
            prev.next = current;
            
            current = next;
        }
        
        return dummy.next;
    }
    
    static class ListNode {
        int val;
        ListNode next;
        ListNode(int val) {
            this.val = val;
        }
    }
}
```

## 8.3 고급 정렬 알고리즘

### 병합 정렬 (Merge Sort)
분할 정복 방식을 사용하는 안정적인 정렬 알고리즘입니다.

```java
public class MergeSort {
    
    // 기본 병합 정렬
    public static void mergeSort(int[] arr) {
        if (arr.length < 2) {
            return;
        }
        
        mergeSort(arr, 0, arr.length - 1);
    }
    
    private static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            
            // 분할
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            
            // 병합
            merge(arr, left, mid, right);
        }
    }
    
    private static void merge(int[] arr, int left, int mid, int right) {
        // 임시 배열 생성
        int[] temp = new int[right - left + 1];
        
        int i = left;      // 왼쪽 부분 배열 인덱스
        int j = mid + 1;   // 오른쪽 부분 배열 인덱스
        int k = 0;         // 임시 배열 인덱스
        
        // 두 부분 배열을 병합
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }
        
        // 남은 요소들 복사
        while (i <= mid) {
            temp[k++] = arr[i++];
        }
        
        while (j <= right) {
            temp[k++] = arr[j++];
        }
        
        // 임시 배열을 원본 배열에 복사
        for (i = 0; i < temp.length; i++) {
            arr[left + i] = temp[i];
        }
    }
    
    // 반복적 병합 정렬 (Bottom-up)
    public static void iterativeMergeSort(int[] arr) {
        int n = arr.length;
        
        // 부분 배열의 크기를 1부터 시작하여 2배씩 증가
        for (int size = 1; size < n; size *= 2) {
            // 현재 크기의 부분 배열들을 병합
            for (int start = 0; start < n - 1; start += 2 * size) {
                int mid = Math.min(start + size - 1, n - 1);
                int end = Math.min(start + 2 * size - 1, n - 1);
                
                if (mid < end) {
                    merge(arr, start, mid, end);
                }
            }
        }
    }
    
    // 3-way 병합 정렬
    public static void threewayMergeSort(int[] arr) {
        if (arr.length < 2) {
            return;
        }
        
        threewayMergeSort(arr, 0, arr.length - 1);
    }
    
    private static void threewayMergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid1 = left + (right - left) / 3;
            int mid2 = left + 2 * (right - left) / 3;
            
            threewayMergeSort(arr, left, mid1);
            threewayMergeSort(arr, mid1 + 1, mid2);
            threewayMergeSort(arr, mid2 + 1, right);
            
            threewayMerge(arr, left, mid1, mid2, right);
        }
    }
    
    private static void threewayMerge(int[] arr, int left, int mid1, int mid2, int right) {
        int[] temp = new int[right - left + 1];
        int i = left, j = mid1 + 1, k = mid2 + 1, l = 0;
        
        // 세 부분을 병합
        while (i <= mid1 && j <= mid2 && k <= right) {
            if (arr[i] <= arr[j] && arr[i] <= arr[k]) {
                temp[l++] = arr[i++];
            } else if (arr[j] <= arr[k]) {
                temp[l++] = arr[j++];
            } else {
                temp[l++] = arr[k++];
            }
        }
        
        // 두 부분이 남은 경우
        while (i <= mid1 && j <= mid2) {
            temp[l++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
        }
        
        while (j <= mid2 && k <= right) {
            temp[l++] = arr[j] <= arr[k] ? arr[j++] : arr[k++];
        }
        
        while (i <= mid1 && k <= right) {
            temp[l++] = arr[i] <= arr[k] ? arr[i++] : arr[k++];
        }
        
        // 한 부분만 남은 경우
        while (i <= mid1) temp[l++] = arr[i++];
        while (j <= mid2) temp[l++] = arr[j++];
        while (k <= right) temp[l++] = arr[k++];
        
        // 결과 복사
        System.arraycopy(temp, 0, arr, left, temp.length);
    }
    
    // 자연 병합 정렬 (Natural Merge Sort)
    public static void naturalMergeSort(int[] arr) {
        int n = arr.length;
        if (n < 2) return;
        
        while (true) {
            int start = 0;
            boolean merged = false;
            
            while (start < n) {
                int mid = findRunEnd(arr, start, n);
                if (mid >= n - 1) break;
                
                int end = findRunEnd(arr, mid + 1, n);
                merge(arr, start, mid, Math.min(end, n - 1));
                merged = true;
                start = end + 1;
            }
            
            if (!merged) break;
        }
    }
    
    private static int findRunEnd(int[] arr, int start, int n) {
        if (start >= n - 1) return n - 1;
        
        int i = start;
        while (i < n - 1 && arr[i] <= arr[i + 1]) {
            i++;
        }
        return i;
    }
}
```

### 퀵 정렬 (Quick Sort)
피벗을 기준으로 분할하여 정렬하는 효율적인 정렬 알고리즘입니다.

```java
public class QuickSort {
    
    // 기본 퀵 정렬
    public static void quickSort(int[] arr) {
        quickSort(arr, 0, arr.length - 1);
    }
    
    private static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    // Lomuto 분할 방식
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr, i, j);
            }
        }
        
        swap(arr, i + 1, high);
        return i + 1;
    }
    
    // Hoare 분할 방식
    private static int hoarePartition(int[] arr, int low, int high) {
        int pivot = arr[low];
        int i = low - 1;
        int j = high + 1;
        
        while (true) {
            do {
                i++;
            } while (arr[i] < pivot);
            
            do {
                j--;
            } while (arr[j] > pivot);
            
            if (i >= j) {
                return j;
            }
            
            swap(arr, i, j);
        }
    }
    
    // 3-way 퀵 정렬 (중복 요소가 많을 때 효율적)
    public static void threewayQuickSort(int[] arr) {
        threewayQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void threewayQuickSort(int[] arr, int low, int high) {
        if (low >= high) {
            return;
        }
        
        int[] partitions = threewayPartition(arr, low, high);
        threewayQuickSort(arr, low, partitions[0] - 1);
        threewayQuickSort(arr, partitions[1] + 1, high);
    }
    
    private static int[] threewayPartition(int[] arr, int low, int high) {
        int pivot = arr[low];
        int i = low;
        int lt = low;
        int gt = high;
        
        while (i <= gt) {
            if (arr[i] < pivot) {
                swap(arr, lt++, i++);
            } else if (arr[i] > pivot) {
                swap(arr, i, gt--);
            } else {
                i++;
            }
        }
        
        return new int[]{lt, gt};
    }
    
    // 랜덤 피벗 퀵 정렬
    public static void randomizedQuickSort(int[] arr) {
        randomizedQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void randomizedQuickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = randomizedPartition(arr, low, high);
            
            randomizedQuickSort(arr, low, pi - 1);
            randomizedQuickSort(arr, pi + 1, high);
        }
    }
    
    private static int randomizedPartition(int[] arr, int low, int high) {
        int randomIndex = low + (int)(Math.random() * (high - low + 1));
        swap(arr, randomIndex, high);
        return partition(arr, low, high);
    }
    
    // 반복적 퀵 정렬
    public static void iterativeQuickSort(int[] arr) {
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        stack.push(arr.length - 1);
        
        while (!stack.isEmpty()) {
            int high = stack.pop();
            int low = stack.pop();
            
            if (low < high) {
                int pi = partition(arr, low, high);
                
                // 더 작은 부분을 먼저 스택에 넣어 스택 크기 최소화
                if (pi - low < high - pi) {
                    stack.push(pi + 1);
                    stack.push(high);
                    stack.push(low);
                    stack.push(pi - 1);
                } else {
                    stack.push(low);
                    stack.push(pi - 1);
                    stack.push(pi + 1);
                    stack.push(high);
                }
            }
        }
    }
    
    // Dual-Pivot 퀵 정렬
    public static void dualPivotQuickSort(int[] arr) {
        dualPivotQuickSort(arr, 0, arr.length - 1);
    }
    
    private static void dualPivotQuickSort(int[] arr, int low, int high) {
        if (low < high) {
            int[] pivots = dualPivotPartition(arr, low, high);
            
            dualPivotQuickSort(arr, low, pivots[0] - 1);
            dualPivotQuickSort(arr, pivots[0] + 1, pivots[1] - 1);
            dualPivotQuickSort(arr, pivots[1] + 1, high);
        }
    }
    
    private static int[] dualPivotPartition(int[] arr, int low, int high) {
        if (arr[low] > arr[high]) {
            swap(arr, low, high);
        }
        
        int p1 = arr[low];
        int p2 = arr[high];
        
        int i = low + 1;
        int lt = low + 1;
        int gt = high - 1;
        
        while (i <= gt) {
            if (arr[i] < p1) {
                swap(arr, i++, lt++);
            } else if (arr[i] > p2) {
                swap(arr, i, gt--);
            } else {
                i++;
            }
        }
        
        swap(arr, low, --lt);
        swap(arr, high, ++gt);
        
        return new int[]{lt, gt};
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

### 힙 정렬 (Heap Sort)
힙 자료구조를 이용한 정렬 알고리즘입니다.

```java
public class HeapSort {
    
    // 기본 힙 정렬
    public static void heapSort(int[] arr) {
        int n = arr.length;
        
        // 힙 구성 (Build heap)
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        
        // 요소를 하나씩 추출
        for (int i = n - 1; i > 0; i--) {
            // 루트(최댓값)를 끝으로 이동
            swap(arr, 0, i);
            
            // 축소된 힙에서 heapify
            heapify(arr, i, 0);
        }
    }
    
    private static void heapify(int[] arr, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        // 왼쪽 자식이 더 크면
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        // 오른쪽 자식이 더 크면
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        // 최댓값이 루트가 아니면
        if (largest != i) {
            swap(arr, i, largest);
            
            // 재귀적으로 하위 트리에 heapify 적용
            heapify(arr, n, largest);
        }
    }
    
    // 반복적 heapify
    private static void iterativeHeapify(int[] arr, int n, int i) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            
            if (left < n && arr[left] > arr[largest]) {
                largest = left;
            }
            
            if (right < n && arr[right] > arr[largest]) {
                largest = right;
            }
            
            if (largest == i) {
                break;
            }
            
            swap(arr, i, largest);
            i = largest;
        }
    }
    
    // Min Heap을 사용한 내림차순 정렬
    public static void heapSortDescending(int[] arr) {
        int n = arr.length;
        
        // Min heap 구성
        for (int i = n / 2 - 1; i >= 0; i--) {
            minHeapify(arr, n, i);
        }
        
        // 요소를 하나씩 추출
        for (int i = n - 1; i > 0; i--) {
            swap(arr, 0, i);
            minHeapify(arr, i, 0);
        }
    }
    
    private static void minHeapify(int[] arr, int n, int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] < arr[smallest]) {
            smallest = left;
        }
        
        if (right < n && arr[right] < arr[smallest]) {
            smallest = right;
        }
        
        if (smallest != i) {
            swap(arr, i, smallest);
            minHeapify(arr, n, smallest);
        }
    }
    
    // k번째 작은/큰 요소 찾기
    public static int findKthLargest(int[] arr, int k) {
        int n = arr.length;
        
        // 전체 배열을 최대 힙으로 구성
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(arr, n, i);
        }
        
        // k-1번 최댓값 제거
        for (int i = 0; i < k - 1; i++) {
            swap(arr, 0, n - 1 - i);
            heapify(arr, n - 1 - i, 0);
        }
        
        return arr[0];
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

## 8.4 비비교 정렬 알고리즘

### 계수 정렬 (Counting Sort)
요소의 개수를 세어 정렬하는 알고리즘입니다.

```java
public class CountingSort {
    
    // 기본 계수 정렬 (음이 아닌 정수)
    public static void countingSort(int[] arr) {
        if (arr.length == 0) return;
        
        // 최댓값 찾기
        int max = arr[0];
        for (int num : arr) {
            max = Math.max(max, num);
        }
        
        // 카운트 배열 생성
        int[] count = new int[max + 1];
        
        // 각 요소의 개수 세기
        for (int num : arr) {
            count[num]++;
        }
        
        // 누적 합 계산
        for (int i = 1; i <= max; i++) {
            count[i] += count[i - 1];
        }
        
        // 결과 배열 생성
        int[] output = new int[arr.length];
        
        // 뒤에서부터 순회하여 안정성 보장
        for (int i = arr.length - 1; i >= 0; i--) {
            output[count[arr[i]] - 1] = arr[i];
            count[arr[i]]--;
        }
        
        // 원본 배열에 복사
        System.arraycopy(output, 0, arr, 0, arr.length);
    }
    
    // 음수를 포함한 계수 정렬
    public static void countingSortWithNegative(int[] arr) {
        if (arr.length == 0) return;
        
        int min = arr[0];
        int max = arr[0];
        
        // 최솟값과 최댓값 찾기
        for (int num : arr) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        
        int range = max - min + 1;
        int[] count = new int[range];
        
        // 개수 세기 (오프셋 적용)
        for (int num : arr) {
            count[num - min]++;
        }
        
        // 결과 배열에 정렬된 값 저장
        int index = 0;
        for (int i = 0; i < range; i++) {
            while (count[i] > 0) {
                arr[index++] = i + min;
                count[i]--;
            }
        }
    }
    
    // 문자열을 위한 계수 정렬
    public static void countingSortForStrings(String[] arr, int position) {
        int n = arr.length;
        String[] output = new String[n];
        int[] count = new int[256]; // ASCII 문자
        
        // 특정 위치의 문자 개수 세기
        for (String str : arr) {
            char ch = position < str.length() ? str.charAt(position) : 0;
            count[ch]++;
        }
        
        // 누적 합 계산
        for (int i = 1; i < 256; i++) {
            count[i] += count[i - 1];
        }
        
        // 안정적으로 정렬
        for (int i = n - 1; i >= 0; i--) {
            char ch = position < arr[i].length() ? arr[i].charAt(position) : 0;
            output[count[ch] - 1] = arr[i];
            count[ch]--;
        }
        
        System.arraycopy(output, 0, arr, 0, n);
    }
}
```

### 기수 정렬 (Radix Sort)
자릿수별로 정렬을 반복하는 알고리즘입니다.

```java
public class RadixSort {
    
    // LSD 기수 정렬 (Least Significant Digit)
    public static void radixSort(int[] arr) {
        if (arr.length == 0) return;
        
        // 최댓값 찾기
        int max = arr[0];
        for (int num : arr) {
            max = Math.max(max, Math.abs(num));
        }
        
        // 각 자릿수에 대해 계수 정렬 수행
        for (int exp = 1; max / exp > 0; exp *= 10) {
            countingSortByDigit(arr, exp);
        }
        
        // 음수 처리
        handleNegatives(arr);
    }
    
    private static void countingSortByDigit(int[] arr, int exp) {
        int n = arr.length;
        int[] output = new int[n];
        int[] count = new int[10];
        
        // 각 자릿수의 개수 세기
        for (int i = 0; i < n; i++) {
            int digit = (Math.abs(arr[i]) / exp) % 10;
            count[digit]++;
        }
        
        // 누적 합 계산
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        
        // 안정적으로 정렬
        for (int i = n - 1; i >= 0; i--) {
            int digit = (Math.abs(arr[i]) / exp) % 10;
            output[count[digit] - 1] = arr[i];
            count[digit]--;
        }
        
        System.arraycopy(output, 0, arr, 0, n);
    }
    
    private static void handleNegatives(int[] arr) {
        int negativeCount = 0;
        for (int num : arr) {
            if (num < 0) negativeCount++;
        }
        
        if (negativeCount == 0) return;
        
        // 음수와 양수 분리
        int[] negatives = new int[negativeCount];
        int[] positives = new int[arr.length - negativeCount];
        int negIdx = 0, posIdx = 0;
        
        for (int num : arr) {
            if (num < 0) {
                negatives[negIdx++] = num;
            } else {
                positives[posIdx++] = num;
            }
        }
        
        // 음수 배열 뒤집기
        for (int i = 0; i < negativeCount / 2; i++) {
            int temp = negatives[i];
            negatives[i] = negatives[negativeCount - 1 - i];
            negatives[negativeCount - 1 - i] = temp;
        }
        
        // 결과 병합
        System.arraycopy(negatives, 0, arr, 0, negativeCount);
        System.arraycopy(positives, 0, arr, negativeCount, positives.length);
    }
    
    // MSD 기수 정렬 (Most Significant Digit)
    public static void msdRadixSort(int[] arr) {
        if (arr.length == 0) return;
        
        int max = arr[0];
        for (int num : arr) {
            max = Math.max(max, Math.abs(num));
        }
        
        int maxDigits = (int)Math.log10(max) + 1;
        msdRadixSort(arr, 0, arr.length - 1, (int)Math.pow(10, maxDigits - 1));
    }
    
    private static void msdRadixSort(int[] arr, int low, int high, int exp) {
        if (low >= high || exp == 0) return;
        
        int[] count = new int[10];
        int[] temp = new int[high - low + 1];
        
        // 자릿수별 개수 세기
        for (int i = low; i <= high; i++) {
            int digit = (Math.abs(arr[i]) / exp) % 10;
            count[digit]++;
        }
        
        // 시작 인덱스 계산
        int[] start = new int[10];
        start[0] = 0;
        for (int i = 1; i < 10; i++) {
            start[i] = start[i - 1] + count[i - 1];
        }
        
        // 임시 배열에 정렬
        for (int i = low; i <= high; i++) {
            int digit = (Math.abs(arr[i]) / exp) % 10;
            temp[start[digit]++] = arr[i];
        }
        
        // 원본 배열에 복사
        for (int i = 0; i < temp.length; i++) {
            arr[low + i] = temp[i];
        }
        
        // 각 버킷에 대해 재귀적으로 정렬
        int bucketStart = low;
        for (int i = 0; i < 10; i++) {
            if (count[i] > 1) {
                msdRadixSort(arr, bucketStart, bucketStart + count[i] - 1, exp / 10);
            }
            bucketStart += count[i];
        }
    }
    
    // 문자열을 위한 기수 정렬
    public static void radixSortStrings(String[] arr) {
        if (arr.length == 0) return;
        
        // 최대 길이 찾기
        int maxLen = 0;
        for (String str : arr) {
            maxLen = Math.max(maxLen, str.length());
        }
        
        // LSD 방식으로 정렬
        for (int pos = maxLen - 1; pos >= 0; pos--) {
            CountingSort.countingSortForStrings(arr, pos);
        }
    }
}
```

### 버킷 정렬 (Bucket Sort)
데이터를 여러 버킷으로 분할한 후 각 버킷을 정렬하는 알고리즘입니다.

```java
public class BucketSort {
    
    // 기본 버킷 정렬 (0과 1 사이의 실수)
    public static void bucketSort(double[] arr) {
        int n = arr.length;
        if (n <= 1) return;
        
        // 버킷 생성
        @SuppressWarnings("unchecked")
        List<Double>[] buckets = new List[n];
        for (int i = 0; i < n; i++) {
            buckets[i] = new ArrayList<>();
        }
        
        // 요소를 버킷에 분배
        for (double num : arr) {
            int bucketIndex = (int)(num * n);
            if (bucketIndex == n) bucketIndex = n - 1;
            buckets[bucketIndex].add(num);
        }
        
        // 각 버킷 정렬
        int index = 0;
        for (List<Double> bucket : buckets) {
            Collections.sort(bucket);
            for (double num : bucket) {
                arr[index++] = num;
            }
        }
    }
    
    // 정수를 위한 버킷 정렬
    public static void bucketSortIntegers(int[] arr) {
        if (arr.length <= 1) return;
        
        // 최솟값과 최댓값 찾기
        int min = arr[0], max = arr[0];
        for (int num : arr) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        
        // 버킷 개수와 크기 계산
        int bucketCount = (int)Math.sqrt(arr.length);
        int bucketSize = (max - min) / bucketCount + 1;
        
        // 버킷 생성
        @SuppressWarnings("unchecked")
        List<Integer>[] buckets = new List[bucketCount];
        for (int i = 0; i < bucketCount; i++) {
            buckets[i] = new ArrayList<>();
        }
        
        // 요소를 버킷에 분배
        for (int num : arr) {
            int bucketIndex = (num - min) / bucketSize;
            if (bucketIndex >= bucketCount) bucketIndex = bucketCount - 1;
            buckets[bucketIndex].add(num);
        }
        
        // 각 버킷을 삽입 정렬로 정렬
        int index = 0;
        for (List<Integer> bucket : buckets) {
            insertionSort(bucket);
            for (int num : bucket) {
                arr[index++] = num;
            }
        }
    }
    
    private static void insertionSort(List<Integer> list) {
        for (int i = 1; i < list.size(); i++) {
            int key = list.get(i);
            int j = i - 1;
            
            while (j >= 0 && list.get(j) > key) {
                list.set(j + 1, list.get(j));
                j--;
            }
            
            list.set(j + 1, key);
        }
    }
    
    // 일반적인 버킷 정렬
    public static <T extends Comparable<T>> void genericBucketSort(T[] arr, 
                                                                   Function<T, Double> mappingFunction) {
        int n = arr.length;
        if (n <= 1) return;
        
        @SuppressWarnings("unchecked")
        List<T>[] buckets = new List[n];
        for (int i = 0; i < n; i++) {
            buckets[i] = new ArrayList<>();
        }
        
        // 매핑 함수를 사용하여 버킷 인덱스 계산
        for (T item : arr) {
            double mappedValue = mappingFunction.apply(item);
            int bucketIndex = (int)(mappedValue * n);
            if (bucketIndex >= n) bucketIndex = n - 1;
            if (bucketIndex < 0) bucketIndex = 0;
            buckets[bucketIndex].add(item);
        }
        
        // 각 버킷 정렬
        int index = 0;
        for (List<T> bucket : buckets) {
            Collections.sort(bucket);
            for (T item : bucket) {
                arr[index++] = item;
            }
        }
    }
}
```

## 8.5 하이브리드 정렬 알고리즘

### 팀 정렬 (Tim Sort)
병합 정렬과 삽입 정렬을 결합한 하이브리드 정렬 알고리즘입니다.

```java
public class TimSort {
    private static final int MIN_MERGE = 32;
    
    public static void timSort(int[] arr) {
        int n = arr.length;
        int minRun = getMinRun(n);
        
        // 작은 run들을 삽입 정렬로 정렬
        for (int i = 0; i < n; i += minRun) {
            insertionSort(arr, i, Math.min(i + minRun - 1, n - 1));
        }
        
        // 병합 시작
        for (int size = minRun; size < n; size *= 2) {
            for (int start = 0; start < n; start += size * 2) {
                int mid = start + size - 1;
                int end = Math.min(start + size * 2 - 1, n - 1);
                
                if (mid < end) {
                    merge(arr, start, mid, end);
                }
            }
        }
    }
    
    private static int getMinRun(int n) {
        int r = 0;
        while (n >= MIN_MERGE) {
            r |= (n & 1);
            n >>= 1;
        }
        return n + r;
    }
    
    private static void insertionSort(int[] arr, int left, int right) {
        for (int i = left + 1; i <= right; i++) {
            int key = arr[i];
            int j = i - 1;
            
            while (j >= left && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            
            arr[j + 1] = key;
        }
    }
    
    private static void merge(int[] arr, int left, int mid, int right) {
        int[] leftArr = Arrays.copyOfRange(arr, left, mid + 1);
        int[] rightArr = Arrays.copyOfRange(arr, mid + 1, right + 1);
        
        int i = 0, j = 0, k = left;
        
        while (i < leftArr.length && j < rightArr.length) {
            if (leftArr[i] <= rightArr[j]) {
                arr[k++] = leftArr[i++];
            } else {
                arr[k++] = rightArr[j++];
            }
        }
        
        while (i < leftArr.length) {
            arr[k++] = leftArr[i++];
        }
        
        while (j < rightArr.length) {
            arr[k++] = rightArr[j++];
        }
    }
}
```

### Intro Sort
퀵 정렬, 힙 정렬, 삽입 정렬을 결합한 하이브리드 정렬입니다.

```java
public class IntroSort {
    private static final int INSERTION_THRESHOLD = 16;
    
    public static void introSort(int[] arr) {
        int maxDepth = (int)(Math.log(arr.length) * 2);
        introSortUtil(arr, 0, arr.length - 1, maxDepth);
    }
    
    private static void introSortUtil(int[] arr, int low, int high, int depthLimit) {
        int size = high - low + 1;
        
        if (size < INSERTION_THRESHOLD) {
            insertionSort(arr, low, high);
        } else if (depthLimit == 0) {
            heapSort(arr, low, high);
        } else {
            int pivot = partition(arr, low, high);
            introSortUtil(arr, low, pivot - 1, depthLimit - 1);
            introSortUtil(arr, pivot + 1, high, depthLimit - 1);
        }
    }
    
    private static void insertionSort(int[] arr, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            int key = arr[i];
            int j = i - 1;
            
            while (j >= low && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            
            arr[j + 1] = key;
        }
    }
    
    private static void heapSort(int[] arr, int low, int high) {
        // 부분 배열을 힙으로 변환
        for (int i = (high - low + 1) / 2 - 1; i >= 0; i--) {
            heapify(arr, low, high, low + i);
        }
        
        // 힙 정렬
        for (int i = high; i > low; i--) {
            swap(arr, low, i);
            heapify(arr, low, i - 1, low);
        }
    }
    
    private static void heapify(int[] arr, int low, int high, int i) {
        int largest = i;
        int left = 2 * (i - low) + 1 + low;
        int right = 2 * (i - low) + 2 + low;
        
        if (left <= high && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right <= high && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest != i) {
            swap(arr, i, largest);
            heapify(arr, low, high, largest);
        }
    }
    
    private static int partition(int[] arr, int low, int high) {
        // Median-of-three pivot selection
        int mid = low + (high - low) / 2;
        if (arr[mid] < arr[low]) swap(arr, low, mid);
        if (arr[high] < arr[low]) swap(arr, low, high);
        if (arr[high] < arr[mid]) swap(arr, mid, high);
        
        swap(arr, mid, high - 1);
        int pivot = arr[high - 1];
        
        int i = low;
        int j = high - 1;
        
        while (true) {
            while (arr[++i] < pivot);
            while (arr[--j] > pivot);
            
            if (i >= j) break;
            swap(arr, i, j);
        }
        
        swap(arr, i, high - 1);
        return i;
    }
    
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

## 8.6 정렬 알고리즘 분석 및 선택

### 정렬 알고리즘 비교

| 알고리즘 | 최선 | 평균 | 최악 | 공간 복잡도 | 안정성 | 특징 |
|---------|------|------|------|------------|--------|------|
| 버블 정렬 | O(n) | O(n²) | O(n²) | O(1) | 안정 | 구현 간단 |
| 선택 정렬 | O(n²) | O(n²) | O(n²) | O(1) | 불안정 | 교환 횟수 적음 |
| 삽입 정렬 | O(n) | O(n²) | O(n²) | O(1) | 안정 | 부분 정렬에 효율적 |
| 병합 정렬 | O(n log n) | O(n log n) | O(n log n) | O(n) | 안정 | 일정한 성능 |
| 퀵 정렬 | O(n log n) | O(n log n) | O(n²) | O(log n) | 불안정 | 평균적으로 가장 빠름 |
| 힙 정렬 | O(n log n) | O(n log n) | O(n log n) | O(1) | 불안정 | 일정한 성능 |
| 계수 정렬 | O(n+k) | O(n+k) | O(n+k) | O(k) | 안정 | 정수에만 사용 |
| 기수 정렬 | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | 안정 | 자릿수가 적을 때 효율적 |
| 버킷 정렬 | O(n+k) | O(n+k) | O(n²) | O(n+k) | 안정 | 균등 분포에 효율적 |

### 정렬 알고리즘 선택 가이드

```java
public class SortingAlgorithmSelector {
    
    public static void adaptiveSort(int[] arr) {
        int n = arr.length;
        
        // 작은 배열: 삽입 정렬
        if (n < 10) {
            InsertionSort.insertionSort(arr);
            return;
        }
        
        // 부분 정렬 확인
        if (isNearlySorted(arr)) {
            InsertionSort.insertionSort(arr);
            return;
        }
        
        // 범위가 작은 정수: 계수 정렬
        int min = arr[0], max = arr[0];
        for (int num : arr) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        
        if (max - min < n * 2) {
            CountingSort.countingSortWithNegative(arr);
            return;
        }
        
        // 일반적인 경우: 퀵 정렬 또는 팀 정렬
        if (n < 1000) {
            QuickSort.randomizedQuickSort(arr);
        } else {
            TimSort.timSort(arr);
        }
    }
    
    private static boolean isNearlySorted(int[] arr) {
        int inversions = 0;
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                inversions++;
                if (inversions > arr.length / 10) {
                    return false;
                }
            }
        }
        return true;
    }
}
```

## 8.7 실습 문제

### 문제 1: K번째 작은 수
정렬되지 않은 배열에서 k번째로 작은 수를 O(n) 평균 시간에 찾으세요.

### 문제 2: 정렬된 배열 병합
k개의 정렬된 배열을 하나의 정렬된 배열로 병합하세요.

### 문제 3: 특수 정렬
0, 1, 2로만 이루어진 배열을 O(n) 시간에 정렬하세요.

### 문제 4: 안정적인 정렬
주어진 정렬 알고리즘을 안정적으로 만드는 방법을 구현하세요.

## 8.8 요약

이 장에서는 다양한 정렬 알고리즘에 대해 학습했습니다:

1. **기초 정렬**: 버블, 선택, 삽입 정렬 - 간단하지만 O(n²)
2. **고급 정렬**: 병합, 퀵, 힙 정렬 - O(n log n) 성능
3. **비비교 정렬**: 계수, 기수, 버킷 정렬 - 특정 조건에서 O(n)
4. **하이브리드 정렬**: 팀 정렬, Intro Sort - 실제 사용되는 최적화된 알고리즘

정렬은 많은 알고리즘의 기초가 되는 중요한 연산입니다. 다음 장에서는 정렬된 데이터에서 효율적으로 원소를 찾는 탐색 알고리즘에 대해 알아보겠습니다.