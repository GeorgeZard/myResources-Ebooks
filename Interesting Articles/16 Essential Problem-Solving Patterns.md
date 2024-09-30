

Saurabh Kurve — Read time: 5 minutes

---

Data Structures and Algorithms (DSA) are crucial for efficient problem-solving. Here are 16 key patterns, with use cases and examples, to help tackle real-world problems. This guide includes concise Java examples to demonstrate each pattern in action.

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#1-sliding-window-pattern)1. **Sliding Window Pattern**

Used to track a subset of data that shifts over time, commonly in arrays or strings.

- **Use Case**: Maximum sum of subarrays.
- **Example**: Maximum sum of subarray of size K.

```
public int maxSubArraySum(int[] arr, int k) {
    int maxSum = 0, windowSum = 0;
    for (int i = 0; i < arr.length; i++) {
        windowSum += arr[i];
        if (i >= k - 1) {
            maxSum = Math.max(maxSum, windowSum);
            windowSum -= arr[i - (k - 1)];
        }
    }
    return maxSum;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#2-two-pointer-pattern)2. **Two Pointer Pattern**

Two pointers work towards a solution by converging from different ends of an array.

- **Use Case**: Find pairs in a sorted array.
- **Example**: Find two numbers that sum up to a target.

```
public int[] twoSum(int[] arr, int target) {
    int left = 0, right = arr.length - 1;
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) return new int[]{left, right};
        else if (sum < target) left++;
        else right--;
    }
    return new int[]{};
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#3-fast-amp-slow-pointers-pattern)3. **Fast & Slow Pointers Pattern**

Two pointers move at different speeds to detect cycles in sequences.

- **Use Case**: Detect cycles in linked lists.
- **Example**: Check if a linked list has a cycle.

```
public boolean hasCycle(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#4-merge-intervals-pattern)4. **Merge Intervals Pattern**

This pattern merges overlapping intervals.

- **Use Case**: Scheduling meetings.
- **Example**: Merge overlapping intervals.

```
public int[][] merge(int[][] intervals) {
    Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
    List<int[]> merged = new ArrayList<>();
    for (int[] interval : intervals) {
        if (merged.isEmpty() || merged.get(merged.size() - 1)[1] < interval[0]) {
            merged.add(interval);
        } else {
            merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], interval[1]);
        }
    }
    return merged.toArray(new int[merged.size()][]);
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#5-cyclic-sort-pattern)5. **Cyclic Sort Pattern**

Sort numbers when elements fall within a range.

- **Use Case**: Finding missing numbers.
- **Example**: Find the missing number from 1 to N.

```
public int findMissingNumber(int[] nums) {
    int i = 0;
    while (i < nums.length) {
        if (nums[i] != i && nums[i] < nums.length) {
            int temp = nums[nums[i]];
            nums[nums[i]] = nums[i];
            nums[i] = temp;
        } else {
            i++;
        }
    }
    for (i = 0; i < nums.length; i++) {
        if (nums[i] != i) return i;
    }
    return nums.length;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#6-inplace-reversal-of-linked-list-pattern)6. **In-Place Reversal of Linked List Pattern**

Reverse a linked list in-place.

- **Use Case**: Reversing a sublist of a linked list.
- **Example**: Reverse a linked list.

```
public ListNode reverseList(ListNode head) {
    ListNode prev = null, current = head;
    while (current != null) {
        ListNode next = current.next;
        current.next = prev;
        prev = current;
        current = next;
    }
    return prev;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#7-tree-breadthfirst-search-bfs-pattern)7. **Tree Breadth-First Search (BFS) Pattern**

Explore nodes level by level in a tree.

- **Use Case**: Level-order traversal.
- **Example**: Traverse a binary tree level by level.

```
public List<List<Integer>> bfs(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();
    if (root != null) queue.add(root);
    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();
        for (int i = 0; i < levelSize; i++) {
            TreeNode node = queue.poll();
            currentLevel.add(node.val);
            if (node.left != null) queue.add(node.left);
            if (node.right != null) queue.add(node.right);
        }
        result.add(currentLevel);
    }
    return result;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#8-depthfirst-search-dfs-pattern)8. **Depth-First Search (DFS) Pattern**

Explore as deep as possible along a branch before backtracking.

- **Use Case**: Searching in trees or graphs.
- **Example**: Finding all root-to-leaf paths.

```
public void dfs(TreeNode node, List<Integer> path, List<List<Integer>> result) {
    if (node == null) return;
    path.add(node.val);
    if (node.left == null && node.right == null) result.add(new ArrayList<>(path));
    dfs(node.left, path, result);
    dfs(node.right, path, result);
    path.remove(path.size() - 1);
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#9-two-heap-pattern)9. **Two Heap Pattern**

Use two heaps to maintain dynamic datasets.

- **Use Case**: Finding the median in a data stream.
- **Example**: Find the median of a stream of numbers.

```
class MedianFinder {
    private PriorityQueue<Integer> low = new PriorityQueue<>(Collections.reverseOrder());
    private PriorityQueue<Integer> high = new PriorityQueue<>();

    public void addNum(int num) {
        low.offer(num);
        high.offer(low.poll());
        if (low.size() < high.size()) low.offer(high.poll());
    }

    public double findMedian() {
        return low.size() > high.size() ? low.peek() : (low.peek() + high.peek()) / 2.0;
    }
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#10-subsets-pattern)10. **Subsets Pattern**

Generate all possible subsets.

- **Use Case**: Combination and permutation problems.
- **Example**: Find all subsets of a set.

```
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    result.add(new ArrayList<>());
    for (int num : nums) {
        int size = result.size();
        for (int i = 0; i < size; i++) {
            List<Integer> subset = new ArrayList<>(result.get(i));
            subset.add(num);
            result.add(subset);
        }
    }
    return result;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#11-modified-binary-search-pattern)11. **Modified Binary Search Pattern**

Search in a rotated or partially sorted array.

- **Use Case**: Finding an element in rotated arrays.
- **Example**: Search for a target in a rotated sorted array.

```
public int search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } else {
            if (target > nums[mid] && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#12-bitwise-xor-pattern)12. **Bitwise XOR Pattern**

Solve problems involving pairs using XOR.

- **Use Case**: Finding unique numbers.
- **Example**: Find the single number in an array.

```
public int singleNumber(int[] nums) {
    int result = 0;
    for (int num : nums) result ^= num;
    return result;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#13-top-k-elements-pattern)13. **Top 'K' Elements Pattern**

Use heaps to find the top K elements in a dataset.

- **Use Case**: Finding top K frequent elements.
- **Example**: Find the K most frequent numbers.

```
public List<Integer> topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> countMap = new HashMap<>();
    for (int num : nums) countMap.put(num, countMap.getOrDefault(num, 0) + 1);
    PriorityQueue<Map.Entry<Integer, Integer>> heap = new PriorityQueue<>(Comparator.comparingInt(Map.Entry::getValue));
    for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
        heap.offer(entry);
        if (heap.size() > k) heap.poll();
    }
    return heap.stream().map(Map.Entry::getKey).collect(Collectors.toList());
}
```

---

1. **K-Way Merge Pattern** Merge multiple sorted arrays efficiently.

- **Use Case**: Merging K sorted lists.
- **Example**: Merge K sorted linked lists.

```
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a.val));
    ListNode dummy = new ListNode(0), tail = dummy;
    for (ListNode list : lists) if (list != null) heap.offer(list);
    while (!heap.isEmpty()) {
        tail.next = heap.poll();
        tail = tail.next;
        if (tail.next != null) heap.offer(tail.next);
    }
    return dummy.next;
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#15-01-knapsack-dynamic-programming-pattern)15. **0/1 Knapsack Dynamic Programming Pattern**

Optimize selection under constraints.

- **Use Case**: Resource allocation.
- **Example**: Solve the 0/1 knapsack problem.

```
public int knapsack(int[] weights, int[] values, int capacity) {
    int n = weights.length;
    int[][] dp = new int[n + 1][capacity + 1];
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= capacity; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = Math.max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    return dp[n][capacity];
}
```

---

### [](https://dev.to/saurabhkurve/16-essential-problem-solving-patterns-31p2?ref=dailydev#16-topological-sort-graph-pattern)16. **Topological Sort Graph Pattern**

Find a valid task order in Directed Acyclic Graphs (DAG).

- **Use Case**: Course scheduling.
- **Example**: Find the correct order of courses.

```
public int[] findOrder(int numCourses, int[][] prerequisites) {
    List<Integer>[] graph = new ArrayList[numCourses];
    int[] inDegree = new int[numCourses];
    for (int i = 0; i < numCourses; i++) graph[i] = new ArrayList<>();
    for (int[] pre : prerequisites) {
        graph[pre[1]].add(pre[0]);
        inDegree[pre[0]]++;
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) queue.add(i);
    int[] result = new int[numCourses];
    int idx = 0;
    while (!queue.isEmpty()) {
        int course = queue.poll();
        result[idx++] = course;
        for (int next : graph[course]) {
            inDegree[next]--;
            if (inDegree[next] == 0) queue.add(next);
        }
    }
    return idx == numCourses ? result : new int[0];
}
```

---

These 16 problem-solving patterns are crucial for mastering DSA. Each pattern can be applied to a wide range of real-world problems, providing an efficient path to optimal solutions.