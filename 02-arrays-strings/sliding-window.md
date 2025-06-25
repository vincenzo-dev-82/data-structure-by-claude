# Sliding Window (슬라이딩 윈도우) 기법

## 개요

Sliding Window 기법은 배열이나 문자열에서 연속된 요소들의 부분집합을 효율적으로 처리하는 알고리즘 패턴입니다. 고정 크기 또는 가변 크기의 "윈도우"를 이동시키며 문제를 해결합니다.

## 핵심 개념

- **윈도우**: 배열/문자열의 연속된 부분 집합
- **슬라이딩**: 윈도우를 한 칸씩 이동
- **효율성**: 중복 계산을 피하고 이전 결과를 재사용

## 윈도우 유형

### 1. 고정 크기 윈도우 (Fixed Size Window)
```
크기 3인 윈도우:
[1, 2, 3, 4, 5, 6]
 └─────┘
    └─────┘
       └─────┘
```

### 2. 가변 크기 윈도우 (Variable Size Window)
```
조건에 따라 확장/축소:
[1, 2, 3, 4, 5, 6]
 └──┘
 └─────┘
    └──────┘
```

## 언제 사용하나요?

1. **부분 배열**의 합/평균/최대/최소 찾기
2. **특정 조건**을 만족하는 가장 긴/짧은 부분 배열
3. **문자열 패턴** 매칭
4. **중복 문자** 관련 문제
5. **K개의 연속된 요소** 처리

## 시간 복잡도
- 일반적으로 **O(n)**
- 중첩 루프를 피하여 O(n×k)를 O(n)으로 개선

## 공간 복잡도
- 고정 크기: **O(1)**
- 가변 크기: **O(k)** (k는 윈도우 크기 또는 고유 요소 수)

---

## 예제 1: Maximum Sum Subarray of Size K (크기 K인 부분 배열의 최대 합)

고정 크기 K의 연속된 부분 배열 중 최대 합을 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>
#include <limits.h>

int maxSumSubarray(int* arr, int n, int k) {
    if (n < k) {
        return -1;
    }
    
    // 첫 번째 윈도우의 합 계산
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    int maxSum = windowSum;
    
    // 윈도우 슬라이딩
    for (int i = k; i < n; i++) {
        // 이전 윈도우의 첫 요소를 빼고 새 요소를 더함
        windowSum = windowSum - arr[i - k] + arr[i];
        if (windowSum > maxSum) {
            maxSum = windowSum;
        }
    }
    
    return maxSum;
}

// 테스트
int main() {
    int arr[] = {1, 4, 2, 10, 23, 3, 1, 0, 20};
    int n = sizeof(arr) / sizeof(arr[0]);
    int k = 4;
    
    int result = maxSumSubarray(arr, n, k);
    printf("Maximum sum of subarray of size %d: %d\n", k, result);
    
    return 0;
}
```

### Java 구현
```java
public class MaxSumSubarray {
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
            // 이전 윈도우의 첫 요소를 빼고 새 요소를 더함
            windowSum = windowSum - arr[i - k] + arr[i];
            maxSum = Math.max(maxSum, windowSum);
        }
        
        return maxSum;
    }
    
    // 테스트
    public static void main(String[] args) {
        MaxSumSubarray mss = new MaxSumSubarray();
        int[] arr = {1, 4, 2, 10, 23, 3, 1, 0, 20};
        int k = 4;
        
        int result = mss.maxSumSubarray(arr, k);
        System.out.println("Maximum sum of subarray of size " + k + ": " + result);
    }
}
```

### Python 구현
```python
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return -1
    
    # 첫 번째 윈도우의 합 계산
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # 윈도우 슬라이딩
    for i in range(k, len(arr)):
        # 이전 윈도우의 첫 요소를 빼고 새 요소를 더함
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# 테스트
if __name__ == "__main__":
    arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
    k = 4
    
    result = max_sum_subarray(arr, k)
    print(f"Maximum sum of subarray of size {k}: {result}")
```

---

## 예제 2: Longest Substring Without Repeating Characters (중복 없는 가장 긴 부분 문자열)

가변 크기 윈도우를 사용하여 중복 문자가 없는 가장 긴 부분 문자열을 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int lengthOfLongestSubstring(char* s) {
    int charSet[128] = {0};  // ASCII 문자
    int maxLength = 0;
    int left = 0;
    
    for (int right = 0; s[right] != '\0'; right++) {
        // 중복 문자가 있으면 왼쪽 포인터 이동
        while (charSet[(int)s[right]] > 0) {
            charSet[(int)s[left]]--;
            left++;
        }
        
        charSet[(int)s[right]]++;
        maxLength = max(maxLength, right - left + 1);
    }
    
    return maxLength;
}

// 테스트
int main() {
    char* s1 = "abcabcbb";
    char* s2 = "bbbbb";
    char* s3 = "pwwkew";
    
    printf("\"%s\" -> %d\n", s1, lengthOfLongestSubstring(s1));
    printf("\"%s\" -> %d\n", s2, lengthOfLongestSubstring(s2));
    printf("\"%s\" -> %d\n", s3, lengthOfLongestSubstring(s3));
    
    return 0;
}
```

### Java 구현
```java
import java.util.*;

public class LongestSubstring {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> window = new HashSet<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            // 중복 문자가 있으면 왼쪽 포인터 이동
            while (window.contains(s.charAt(right))) {
                window.remove(s.charAt(left));
                left++;
            }
            
            window.add(s.charAt(right));
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
    
    // 최적화된 버전 (HashMap 사용)
    public int lengthOfLongestSubstringOptimized(String s) {
        Map<Character, Integer> charIndex = new HashMap<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);
            
            // 중복 문자가 현재 윈도우 내에 있으면
            if (charIndex.containsKey(c) && charIndex.get(c) >= left) {
                left = charIndex.get(c) + 1;
            }
            
            charIndex.put(c, right);
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
    
    // 테스트
    public static void main(String[] args) {
        LongestSubstring ls = new LongestSubstring();
        
        String[] testCases = {"abcabcbb", "bbbbb", "pwwkew"};
        
        for (String s : testCases) {
            System.out.println("\"" + s + "\" -> " + ls.lengthOfLongestSubstring(s));
        }
    }
}
```

### Python 구현
```python
def length_of_longest_substring(s):
    char_set = set()
    max_length = 0
    left = 0
    
    for right in range(len(s)):
        # 중복 문자가 있으면 왼쪽 포인터 이동
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

def length_of_longest_substring_optimized(s):
    """최적화된 버전 (딕셔너리 사용)"""
    char_index = {}
    max_length = 0
    left = 0
    
    for right in range(len(s)):
        # 중복 문자가 현재 윈도우 내에 있으면
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1
        
        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

# 테스트
if __name__ == "__main__":
    test_cases = ["abcabcbb", "bbbbb", "pwwkew"]
    
    for s in test_cases:
        print(f'"{s}" -> {length_of_longest_substring(s)}')
```

---

## 예제 3: Minimum Window Substring (최소 윈도우 부분 문자열)

문자열 S에서 T의 모든 문자를 포함하는 최소 길이의 부분 문자열을 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>
#include <string.h>
#include <limits.h>

char* minWindow(char* s, char* t) {
    if (strlen(s) == 0 || strlen(t) == 0) {
        return "";
    }
    
    int charCount[128] = {0};
    int required = 0;
    
    // t의 문자 빈도 계산
    for (int i = 0; t[i] != '\0'; i++) {
        if (charCount[(int)t[i]] == 0) {
            required++;
        }
        charCount[(int)t[i]]++;
    }
    
    int left = 0, right = 0;
    int formed = 0;
    int windowCounts[128] = {0};
    
    int minLen = INT_MAX;
    int minLeft = 0;
    
    while (right < strlen(s)) {
        char c = s[right];
        windowCounts[(int)c]++;
        
        if (charCount[(int)c] > 0 && windowCounts[(int)c] == charCount[(int)c]) {
            formed++;
        }
        
        // 모든 문자를 포함하면 윈도우 축소 시도
        while (left <= right && formed == required) {
            c = s[left];
            
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minLeft = left;
            }
            
            windowCounts[(int)c]--;
            if (charCount[(int)c] > 0 && windowCounts[(int)c] < charCount[(int)c]) {
                formed--;
            }
            
            left++;
        }
        
        right++;
    }
    
    if (minLen == INT_MAX) {
        return "";
    }
    
    static char result[10000];
    strncpy(result, s + minLeft, minLen);
    result[minLen] = '\0';
    
    return result;
}

// 테스트
int main() {
    char* s = "ADOBECODEBANC";
    char* t = "ABC";
    
    char* result = minWindow(s, t);
    printf("Minimum window substring: \"%s\"\n", result);
    
    return 0;
}
```

### Java 구현
```java
import java.util.*;

public class MinimumWindowSubstring {
    public String minWindow(String s, String t) {
        if (s.length() == 0 || t.length() == 0) {
            return "";
        }
        
        // t의 문자 빈도 계산
        Map<Character, Integer> dictT = new HashMap<>();
        for (char c : t.toCharArray()) {
            dictT.put(c, dictT.getOrDefault(c, 0) + 1);
        }
        
        int required = dictT.size();
        int left = 0, right = 0;
        int formed = 0;
        
        Map<Character, Integer> windowCounts = new HashMap<>();
        
        // 결과 저장
        int[] ans = {-1, 0, 0}; // {길이, 왼쪽, 오른쪽}
        
        while (right < s.length()) {
            char c = s.charAt(right);
            windowCounts.put(c, windowCounts.getOrDefault(c, 0) + 1);
            
            if (dictT.containsKey(c) && 
                windowCounts.get(c).intValue() == dictT.get(c).intValue()) {
                formed++;
            }
            
            // 모든 문자를 포함하면 윈도우 축소 시도
            while (left <= right && formed == required) {
                c = s.charAt(left);
                
                if (ans[0] == -1 || right - left + 1 < ans[0]) {
                    ans[0] = right - left + 1;
                    ans[1] = left;
                    ans[2] = right;
                }
                
                windowCounts.put(c, windowCounts.get(c) - 1);
                if (dictT.containsKey(c) && 
                    windowCounts.get(c).intValue() < dictT.get(c).intValue()) {
                    formed--;
                }
                
                left++;
            }
            
            right++;
        }
        
        return ans[0] == -1 ? "" : s.substring(ans[1], ans[2] + 1);
    }
    
    // 테스트
    public static void main(String[] args) {
        MinimumWindowSubstring mws = new MinimumWindowSubstring();
        
        String s = "ADOBECODEBANC";
        String t = "ABC";
        
        String result = mws.minWindow(s, t);
        System.out.println("Minimum window substring: \"" + result + "\"");
    }
}
```

### Python 구현
```python
from collections import Counter, defaultdict

def min_window(s, t):
    if not t or not s:
        return ""
    
    # t의 문자 빈도 계산
    dict_t = Counter(t)
    required = len(dict_t)
    
    left = right = 0
    formed = 0
    
    window_counts = defaultdict(int)
    
    # 결과 저장: (길이, 왼쪽, 오른쪽)
    ans = float("inf"), None, None
    
    while right < len(s):
        character = s[right]
        window_counts[character] += 1
        
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1
        
        # 모든 문자를 포함하면 윈도우 축소 시도
        while left <= right and formed == required:
            character = s[left]
            
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]

# 테스트
if __name__ == "__main__":
    s = "ADOBECODEBANC"
    t = "ABC"
    
    result = min_window(s, t)
    print(f'Minimum window substring: "{result}"')
```

---

## 예제 4: Longest Substring with K Distinct Characters (K개의 서로 다른 문자를 가진 가장 긴 부분 문자열)

최대 K개의 서로 다른 문자를 포함하는 가장 긴 부분 문자열을 찾는 문제입니다.

### C 구현
```c
#include <stdio.h>
#include <string.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int lengthOfLongestSubstringKDistinct(char* s, int k) {
    if (k == 0) return 0;
    
    int charCount[128] = {0};
    int distinctCount = 0;
    int maxLength = 0;
    int left = 0;
    
    for (int right = 0; s[right] != '\0'; right++) {
        // 새로운 문자 추가
        if (charCount[(int)s[right]] == 0) {
            distinctCount++;
        }
        charCount[(int)s[right]]++;
        
        // K개를 초과하면 왼쪽 포인터 이동
        while (distinctCount > k) {
            charCount[(int)s[left]]--;
            if (charCount[(int)s[left]] == 0) {
                distinctCount--;
            }
            left++;
        }
        
        maxLength = max(maxLength, right - left + 1);
    }
    
    return maxLength;
}

// 테스트
int main() {
    char* s1 = "eceba";
    int k1 = 2;
    
    char* s2 = "aa";
    int k2 = 1;
    
    printf("\"%s\" with k=%d -> %d\n", s1, k1, 
           lengthOfLongestSubstringKDistinct(s1, k1));
    printf("\"%s\" with k=%d -> %d\n", s2, k2, 
           lengthOfLongestSubstringKDistinct(s2, k2));
    
    return 0;
}
```

### Java 구현
```java
import java.util.*;

public class LongestSubstringKDistinct {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (k == 0) return 0;
        
        Map<Character, Integer> charCount = new HashMap<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char rightChar = s.charAt(right);
            charCount.put(rightChar, charCount.getOrDefault(rightChar, 0) + 1);
            
            // K개를 초과하면 왼쪽 포인터 이동
            while (charCount.size() > k) {
                char leftChar = s.charAt(left);
                charCount.put(leftChar, charCount.get(leftChar) - 1);
                
                if (charCount.get(leftChar) == 0) {
                    charCount.remove(leftChar);
                }
                left++;
            }
            
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
    
    // 테스트
    public static void main(String[] args) {
        LongestSubstringKDistinct lskd = new LongestSubstringKDistinct();
        
        System.out.println("\"eceba\" with k=2 -> " + 
                          lskd.lengthOfLongestSubstringKDistinct("eceba", 2));
        System.out.println("\"aa\" with k=1 -> " + 
                          lskd.lengthOfLongestSubstringKDistinct("aa", 1));
    }
}
```

### Python 구현
```python
from collections import defaultdict

def length_of_longest_substring_k_distinct(s, k):
    if k == 0:
        return 0
    
    char_count = defaultdict(int)
    max_length = 0
    left = 0
    
    for right in range(len(s)):
        # 새로운 문자 추가
        char_count[s[right]] += 1
        
        # K개를 초과하면 왼쪽 포인터 이동
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

# 테스트
if __name__ == "__main__":
    test_cases = [
        ("eceba", 2),
        ("aa", 1),
        ("a", 2)
    ]
    
    for s, k in test_cases:
        result = length_of_longest_substring_k_distinct(s, k)
        print(f'"{s}" with k={k} -> {result}')
```

---

## 추가 연습 문제

### 고정 크기 윈도우
1. **Average of Subarrays**: 크기 K인 모든 부분 배열의 평균 계산
2. **Max of All Subarrays**: 크기 K인 모든 부분 배열의 최댓값
3. **First Negative in Window**: 각 윈도우의 첫 번째 음수

### 가변 크기 윈도우
1. **Longest Subarray with Sum K**: 합이 K인 가장 긴 부분 배열
2. **Fruits into Baskets**: 최대 2종류의 과일을 담는 문제
3. **Max Consecutive Ones III**: 최대 K개의 0을 1로 바꿀 수 있을 때 연속된 1의 최대 길이

### 문자열 패턴
1. **Find All Anagrams**: 문자열에서 모든 애너그램 찾기
2. **Permutation in String**: 순열이 부분 문자열로 존재하는지 확인
3. **Repeated DNA Sequences**: 반복되는 DNA 시퀀스 찾기

## 핵심 팁

### 1. 윈도우 크기 결정
- **고정**: 문제에서 명시된 크기 K
- **가변**: 특정 조건을 만족할 때까지 확장/축소

### 2. 데이터 구조 선택
- **배열/해시맵**: 문자/숫자 빈도 추적
- **큐/덱**: 윈도우 내 최대/최소값 추적
- **집합**: 중복 확인

### 3. 최적화 전략
- 이전 계산 결과 재사용
- 불필요한 재계산 방지
- 적절한 자료구조로 연산 복잡도 감소

## 요약

- Sliding Window는 연속된 부분 집합을 효율적으로 처리
- 고정/가변 크기 윈도우를 상황에 맞게 선택
- O(n×k)를 O(n)으로 개선 가능
- 윈도우 내 상태를 효율적으로 관리하는 것이 핵심
- 문자열 패턴, 부분 배열 문제에 특히 유용