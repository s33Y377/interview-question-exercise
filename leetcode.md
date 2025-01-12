The sliding window technique is a powerful and efficient approach used to solve problems that involve arrays or lists, especially when dealing with contiguous subarrays or substrings. The idea is to maintain a "window" that slides over the data, adjusting its size or position as needed.

### Key Steps of the Sliding Window Technique:
1. **Initialize the window**: Start by defining the window size and the initial position.
2. **Expand the window**: Move the window’s boundary to include new elements as needed.
3. **Shrink the window**: Remove elements from the window when necessary (e.g., when the window becomes too large or does not satisfy the problem's conditions).
4. **Update the result**: Keep track of the result as the window slides over the array.

This approach is commonly used to optimize problems that involve finding a subarray or substring that satisfies a certain condition.

### Example Problem: Maximum Sum of Subarray of Size K

**Problem:**
Given an array of integers, find the maximum sum of a subarray of size `k`.

#### Approach using Sliding Window:

1. **Initialize the window**: Start by calculating the sum of the first `k` elements.
2. **Slide the window**: For each subsequent element, slide the window one step to the right by adding the next element in the array and removing the element that is no longer in the window.
3. **Update the result**: Keep track of the maximum sum found as the window slides.

#### Example:

Input: 
```
arr = [2, 1, 5, 1, 3, 2]
k = 3
```

Output: 
```
9  (The maximum sum is 2 + 5 + 1 = 9)
```

#### Solution:

1. **Initialize**: Start by calculating the sum of the first 3 elements:
   - `current_sum = 2 + 1 + 5 = 8`
   - `max_sum = 8`
   
2. **Slide the window**:
   - Move the window by 1 step: Remove the first element (2), add the next element (1).
   - New sum = `8 - 2 + 1 = 7`
   - `max_sum` remains 8.

   - Move the window by 1 step: Remove the first element (1), add the next element (3).
   - New sum = `7 - 1 + 3 = 9`
   - `max_sum = 9` (updated).

   - Move the window by 1 step: Remove the first element (5), add the next element (2).
   - New sum = `9 - 5 + 2 = 6`
   - `max_sum` remains 9.

3. **Final result**: The maximum sum found is `9`.

#### Code Implementation:

```python
def max_sum_subarray(arr, k):
    # Step 1: Initialize the sum of the first window of size k
    current_sum = sum(arr[:k])
    max_sum = current_sum
    
    # Step 2: Slide the window over the rest of the array
    for i in range(k, len(arr)):
        current_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Example usage:
arr = [2, 1, 5, 1, 3, 2]
k = 3
print(max_sum_subarray(arr, k))  # Output: 9
```

### Why Sliding Window?
- **Time Efficiency**: The sliding window technique reduces the time complexity from O(n*k) (if you used nested loops) to O(n), because we only calculate the sum for each element once.
- **Space Efficiency**: It uses only a constant amount of extra space, apart from the input array.

This is a typical example of the sliding window technique, which can be adapted for various other problems, such as finding the longest substring without repeating characters or finding the minimum length subarray with a sum greater than or equal to a given value.


---


Here’s a collection of some commonly solved **Sliding Window** problems from **LeetCode**, along with their Python solutions. These problems generally involve subarrays or substrings, and the sliding window technique helps to optimize the brute force approach.

### 1. **LeetCode #3 - Longest Substring Without Repeating Characters**
**Problem:**
Given a string `s`, find the length of the longest substring without repeating characters.

**Solution using Sliding Window:**

```python
def lengthOfLongestSubstring(s: str) -> int:
    window = set()  # To store unique characters
    left = 0  # Left pointer of the window
    max_length = 0
    
    for right in range(len(s)):  # Right pointer
        while s[right] in window:
            window.remove(s[left])  # Remove the left character and move the left pointer
            left += 1
        window.add(s[right])  # Add the current character to the window
        max_length = max(max_length, right - left + 1)  # Update the max length
    
    return max_length
```

### 2. **LeetCode #76 - Minimum Window Substring**
**Problem:**
Given two strings `s` and `t`, return the minimum window substring of `s` which contains all the characters in `t`.

**Solution using Sliding Window:**

```python
from collections import Counter

def minWindow(s: str, t: str) -> str:
    if not s or not t:
        return ""
    
    t_count = Counter(t)  # Count of characters in t
    window_count = Counter()  # Count of characters in the current window
    left = 0
    right = 0
    min_length = float('inf')
    min_start = 0
    required = len(t_count)
    formed = 0
    
    while right < len(s):
        char = s[right]
        window_count[char] += 1
        
        if window_count[char] == t_count[char]:
            formed += 1
        
        while left <= right and formed == required:
            # Update the minimum window length
            if right - left + 1 < min_length:
                min_length = right - left + 1
                min_start = left
            
            # Shrink the window from the left
            window_count[s[left]] -= 1
            if window_count[s[left]] < t_count[s[left]]:
                formed -= 1
            left += 1
        
        right += 1
    
    return s[min_start:min_start + min_length] if min_length != float('inf') else ""
```

### 3. **LeetCode #209 - Minimum Size Subarray Sum**
**Problem:**
Given an array of positive integers `nums` and a positive integer `target`, find the length of the smallest contiguous subarray whose sum is greater than or equal to `target`.

**Solution using Sliding Window:**

```python
def minSubArrayLen(target: int, nums: list) -> int:
    left = 0
    current_sum = 0
    min_length = float('inf')
    
    for right in range(len(nums)):
        current_sum += nums[right]
        
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0
```

### 4. **LeetCode #340 - Longest Substring with At Most K Distinct Characters**
**Problem:**
Given a string `s` and an integer `k`, find the length of the longest substring that contains at most `k` distinct characters.

**Solution using Sliding Window:**

```python
from collections import defaultdict

def longestKSubstr(s: str, k: int) -> int:
    char_count = defaultdict(int)
    left = 0
    max_len = -1
    
    for right in range(len(s)):
        char_count[s[right]] += 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        if len(char_count) == k:
            max_len = max(max_len, right - left + 1)
    
    return max_len
```

### 5. **LeetCode #283 - Move Zeroes**
**Problem:**
Given an array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

**Solution using Sliding Window:**

```python
def moveZeroes(nums: list) -> None:
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
```

### 6. **LeetCode #674 - Longest Continuous Increasing Subsequence**
**Problem:**
Given an unsorted array of integers, find the length of the longest continuous increasing subsequence.

**Solution using Sliding Window:**

```python
def findLengthOfLCIS(nums: list) -> int:
    if not nums:
        return 0
    
    max_len = 1
    left = 0
    
    for right in range(1, len(nums)):
        if nums[right] <= nums[right - 1]:
            left = right  # Start new window
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### 7. **LeetCode #992 - Subarrays with K Different Integers**
**Problem:**
Given an array of integers `nums` and an integer `k`, return the number of subarrays with exactly `k` different integers.

**Solution using Sliding Window:**

```python
from collections import defaultdict

def subarraysWithKDistinct(nums: list, k: int) -> int:
    def at_most_k(k):
        count = defaultdict(int)
        left = 0
        result = 0
        for right in range(len(nums)):
            count[nums[right]] += 1
            while len(count) > k:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1
            result += right - left + 1
        return result
    
    return at_most_k(k) - at_most_k(k - 1)
```

### Conclusion:
These problems are all commonly solved using the **sliding window** technique, where you maintain a window (a subarray or substring) and adjust the window size dynamically to satisfy the given condition. The sliding window technique helps to reduce the time complexity compared to brute force solutions, especially for problems involving subarrays or substrings.


---
---

