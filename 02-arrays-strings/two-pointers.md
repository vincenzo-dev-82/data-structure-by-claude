# Two Pointers (투 포인터) 기법

## 개요

Two Pointers 기법은 배열이나 문자열 문제를 해결할 때 사용하는 효율적인 알고리즘 패턴입니다. 두 개의 포인터를 사용하여 데이터를 순회하면서 특정 조건을 만족하는 답을 찾습니다.

## 핵심 개념

- **두 개의 포인터**: 배열의 서로 다른 위치를 가리키는 두 개의 인덱스
- **포인터 이동**: 특정 조건에 따라 포인터를 이동시키며 탐색
- **시간 복잡도 개선**: O(n²)를 O(n)으로 개선 가능

## 언제 사용하나요?

1. **정렬된 배열**에서 특정 조건을 만족하는 쌍 찾기
2. **회문(Palindrome)** 검사
3. **중복 제거** 문제
4. **구간 합** 문제
5. **병합** 작업

## 패턴 유형

### 1. 양 끝에서 시작 (Opposite Direction)
```
[1, 2, 3, 4, 5, 6, 7]
 ↑                 ↑
left             right
```

### 2. 같은 방향 (Same Direction)
```
[1, 2, 3, 4, 5, 6, 7]
 ↑  ↑
slow fast
```

## 시간 복잡도
- 일반적으로 **O(n)**
- 중첩 루프를 피하여 O(n²)를 O(n)으로 개선

## 공간 복잡도
- 일반적으로 **O(1)**
- 추가 배열을 사용하지 않고 포인터만 사용

---

## 예제 1: Two Sum (정렬된 배열에서 두 수의 합)

정렬된 배열에서 합이 target이 되는 두 수의 인덱스를 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>
#include <stdlib.h>

int* twoSum(int* numbers, int numbersSize, int target, int* returnSize) {
    int* result = (int*)malloc(2 * sizeof(int));
    *returnSize = 2;
    
    int left = 0;
    int right = numbersSize - 1;
    
    while (left < right) {
        int sum = numbers[left] + numbers[right];
        
        if (sum == target) {
            result[0] = left + 1;  // 1-indexed
            result[1] = right + 1;
            return result;
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    
    return result;
}

// 테스트
int main() {
    int numbers[] = {2, 7, 11, 15};
    int target = 9;
    int returnSize;
    
    int* result = twoSum(numbers, 4, target, &returnSize);
    printf("Indices: [%d, %d]\n", result[0], result[1]);
    
    free(result);
    return 0;
}
```

### Java 구현
```java
public class TwoPointers {
    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            
            if (sum == target) {
                return new int[]{left + 1, right + 1}; // 1-indexed
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        
        return new int[]{-1, -1};
    }
    
    // 테스트
    public static void main(String[] args) {
        TwoPointers tp = new TwoPointers();
        int[] numbers = {2, 7, 11, 15};
        int target = 9;
        
        int[] result = tp.twoSum(numbers, target);
        System.out.println("Indices: [" + result[0] + ", " + result[1] + "]");
    }
}
```

### Python 구현
```python
def two_sum(numbers, target):
    left = 0
    right = len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return [-1, -1]

# 테스트
if __name__ == "__main__":
    numbers = [2, 7, 11, 15]
    target = 9
    
    result = two_sum(numbers, target)
    print(f"Indices: {result}")
```

---

## 예제 2: Container With Most Water (물 담기)

높이가 다른 막대들 중에서 가장 많은 물을 담을 수 있는 두 막대를 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

int maxArea(int* height, int heightSize) {
    int left = 0;
    int right = heightSize - 1;
    int maxWater = 0;
    
    while (left < right) {
        int width = right - left;
        int currentHeight = min(height[left], height[right]);
        int area = width * currentHeight;
        maxWater = max(maxWater, area);
        
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    
    return maxWater;
}

// 테스트
int main() {
    int height[] = {1, 8, 6, 2, 5, 4, 8, 3, 7};
    int size = sizeof(height) / sizeof(height[0]);
    
    int result = maxArea(height, size);
    printf("Maximum water: %d\n", result);
    
    return 0;
}
```

### Java 구현
```java
public class ContainerWithMostWater {
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxWater = 0;
        
        while (left < right) {
            int width = right - left;
            int currentHeight = Math.min(height[left], height[right]);
            int area = width * currentHeight;
            maxWater = Math.max(maxWater, area);
            
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxWater;
    }
    
    // 테스트
    public static void main(String[] args) {
        ContainerWithMostWater container = new ContainerWithMostWater();
        int[] height = {1, 8, 6, 2, 5, 4, 8, 3, 7};
        
        int result = container.maxArea(height);
        System.out.println("Maximum water: " + result);
    }
}
```

### Python 구현
```python
def max_area(height):
    left = 0
    right = len(height) - 1
    max_water = 0
    
    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        area = width * current_height
        max_water = max(max_water, area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# 테스트
if __name__ == "__main__":
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    
    result = max_area(height)
    print(f"Maximum water: {result}")
```

---

## 예제 3: Valid Palindrome (회문 검사)

문자열이 회문인지 확인하는 문제입니다. (알파벳과 숫자만 고려)

### C 구현
```c
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

bool isPalindrome(char* s) {
    int left = 0;
    int right = strlen(s) - 1;
    
    while (left < right) {
        // 왼쪽 포인터: 알파벳/숫자가 아니면 건너뛰기
        while (left < right && !isalnum(s[left])) {
            left++;
        }
        
        // 오른쪽 포인터: 알파벳/숫자가 아니면 건너뛰기
        while (left < right && !isalnum(s[right])) {
            right--;
        }
        
        // 대소문자 구분 없이 비교
        if (tolower(s[left]) != tolower(s[right])) {
            return false;
        }
        
        left++;
        right--;
    }
    
    return true;
}

// 테스트
int main() {
    char* s1 = "A man, a plan, a canal: Panama";
    char* s2 = "race a car";
    
    printf("\"%s\" is palindrome: %s\n", s1, isPalindrome(s1) ? "true" : "false");
    printf("\"%s\" is palindrome: %s\n", s2, isPalindrome(s2) ? "true" : "false");
    
    return 0;
}
```

### Java 구현
```java
public class ValidPalindrome {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        
        while (left < right) {
            // 왼쪽 포인터: 알파벳/숫자가 아니면 건너뛰기
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            
            // 오른쪽 포인터: 알파벳/숫자가 아니면 건너뛰기
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            
            // 대소문자 구분 없이 비교
            if (Character.toLowerCase(s.charAt(left)) != 
                Character.toLowerCase(s.charAt(right))) {
                return false;
            }
            
            left++;
            right--;
        }
        
        return true;
    }
    
    // 테스트
    public static void main(String[] args) {
        ValidPalindrome vp = new ValidPalindrome();
        
        String s1 = "A man, a plan, a canal: Panama";
        String s2 = "race a car";
        
        System.out.println("\"" + s1 + "\" is palindrome: " + vp.isPalindrome(s1));
        System.out.println("\"" + s2 + "\" is palindrome: " + vp.isPalindrome(s2));
    }
}
```

### Python 구현
```python
def is_palindrome(s):
    left = 0
    right = len(s) - 1
    
    while left < right:
        # 왼쪽 포인터: 알파벳/숫자가 아니면 건너뛰기
        while left < right and not s[left].isalnum():
            left += 1
        
        # 오른쪽 포인터: 알파벳/숫자가 아니면 건너뛰기
        while left < right and not s[right].isalnum():
            right -= 1
        
        # 대소문자 구분 없이 비교
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

# 테스트
if __name__ == "__main__":
    s1 = "A man, a plan, a canal: Panama"
    s2 = "race a car"
    
    print(f'"{s1}" is palindrome: {is_palindrome(s1)}')
    print(f'"{s2}" is palindrome: {is_palindrome(s2)}')
```

---

## 예제 4: 3Sum (세 수의 합)

배열에서 합이 0이 되는 세 수의 조합을 모두 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>
#include <stdlib.h>

int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int** threeSum(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    qsort(nums, numsSize, sizeof(int), compare);
    
    int capacity = 1000;
    int** result = (int**)malloc(capacity * sizeof(int*));
    *returnColumnSizes = (int*)malloc(capacity * sizeof(int));
    *returnSize = 0;
    
    for (int i = 0; i < numsSize - 2; i++) {
        // 중복 건너뛰기
        if (i > 0 && nums[i] == nums[i-1]) continue;
        
        int left = i + 1;
        int right = numsSize - 1;
        
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            
            if (sum == 0) {
                result[*returnSize] = (int*)malloc(3 * sizeof(int));
                result[*returnSize][0] = nums[i];
                result[*returnSize][1] = nums[left];
                result[*returnSize][2] = nums[right];
                (*returnColumnSizes)[*returnSize] = 3;
                (*returnSize)++;
                
                // 중복 건너뛰기
                while (left < right && nums[left] == nums[left+1]) left++;
                while (left < right && nums[right] == nums[right-1]) right--;
                
                left++;
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                right--;
            }
        }
    }
    
    return result;
}

// 테스트
int main() {
    int nums[] = {-1, 0, 1, 2, -1, -4};
    int numsSize = 6;
    int returnSize;
    int* returnColumnSizes;
    
    int** result = threeSum(nums, numsSize, &returnSize, &returnColumnSizes);
    
    printf("3Sum results:\n");
    for (int i = 0; i < returnSize; i++) {
        printf("[%d, %d, %d]\n", result[i][0], result[i][1], result[i][2]);
        free(result[i]);
    }
    
    free(result);
    free(returnColumnSizes);
    return 0;
}
```

### Java 구현
```java
import java.util.*;

public class ThreeSum {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length - 2; i++) {
            // 중복 건너뛰기
            if (i > 0 && nums[i] == nums[i-1]) continue;
            
            int left = i + 1;
            int right = nums.length - 1;
            
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                
                if (sum == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    
                    // 중복 건너뛰기
                    while (left < right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;
                    
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        
        return result;
    }
    
    // 테스트
    public static void main(String[] args) {
        ThreeSum ts = new ThreeSum();
        int[] nums = {-1, 0, 1, 2, -1, -4};
        
        List<List<Integer>> result = ts.threeSum(nums);
        System.out.println("3Sum results:");
        for (List<Integer> triplet : result) {
            System.out.println(triplet);
        }
    }
}
```

### Python 구현
```python
def three_sum(nums):
    result = []
    nums.sort()
    
    for i in range(len(nums) - 2):
        # 중복 건너뛰기
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left = i + 1
        right = len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # 중복 건너뛰기
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result

# 테스트
if __name__ == "__main__":
    nums = [-1, 0, 1, 2, -1, -4]
    
    result = three_sum(nums)
    print("3Sum results:")
    for triplet in result:
        print(triplet)
```

---

## 추가 연습 문제

1. **Remove Duplicates from Sorted Array**: 정렬된 배열에서 중복을 제거
2. **Valid Palindrome II**: 최대 한 문자를 삭제하여 회문 만들기
3. **Merge Sorted Array**: 두 정렬된 배열 병합
4. **4Sum**: 네 수의 합이 target이 되는 조합 찾기
5. **Trapping Rain Water**: 빗물 담기 문제

## 핵심 요약

- Two Pointers는 배열/문자열 탐색을 효율적으로 만드는 기법
- 정렬된 데이터에서 특히 유용
- O(n²)를 O(n)으로 개선 가능
- 포인터 이동 조건을 명확히 정의하는 것이 중요
- 중복 처리에 주의