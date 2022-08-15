# https://leetcode.cn/problems/longest-palindromic-substring/
from typing import List, Optional


def longestPalindrome(self, s: str) -> str:
    dp = []
    length = len(s)
    res = s[0]
    for i in range(0, length):
        tdp = []
        for j in range(0, length):
            if i == j:
                tdp.append(True)
            elif i == j - 1 and s[i] == s[j]:
                tdp.append(True)
                res = s[i:j + 1]
            else:
                tdp.append(False)
        dp.append(tdp)
    for i in range(length, -1, -1):
        for j in range(i + 2, length):
            if dp[i + 1][j - 1] and s[i] == s[j]:
                dp[i][j] = True
                if j - i + 1 > len(res):
                    res = s[i:j + 1]
    print(dp)
    return res


# https://leetcode.cn/problems/generate-parentheses/
def generateParenthesis(self, n: int) -> List[str]:
    dp = {"()"}
    for i in range(1, n):
        tdp = set()
        for x in dp:
            x = "(" + x
            balance = 0
            l = len(x)
            for j in range(0, l):
                if x[j] == '(':
                    balance += 1
                else:
                    balance -= 1
                if balance == 1:
                    tdp.add(x[0:j + 1] + ")" + x[j + 1:l + 1])
        dp = tdp
    return list(dp)


# https://leetcode.cn/problems/jump-game-ii/
def jump(self, nums: List[int]) -> int:
    lenth = len(nums)
    dp = [10000] * lenth
    dp[0] = 0
    for i in range(0, lenth - 1):
        for j in range(i, min(i + nums[i] + 1, lenth)):
            dp[j] = min(dp[i] + 1, dp[j])
    return dp[lenth - 1]


def canJump(self, nums: List[int]) -> bool:
    lenth = len(nums)
    edge = nums[0]
    for i in range(1, lenth - 1):
        if i > edge:
            break
        else:
            edge = max(nums[i] + i, edge)
    if edge >= lenth - 1:
        return True
    return False


def uniquePaths(self, m: int, n: int) -> int:
    dp = [1] * n
    for i in range(1, m):
        tdp = [1]
        for j in range(1, n):
            tdp.append(tdp[j - 1] + dp[j])
        dp = tdp
    return dp[n - 1]


# https://leetcode.cn/problems/unique-paths-ii/
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    length = len(obstacleGrid[0])
    dp = [0] * length
    for i in range(0, length):
        if obstacleGrid[0][i] == 1:
            break
        dp[i] = 1
    for i in range(1, len(obstacleGrid)):
        tdp = [dp[0]]
        if obstacleGrid[i][0] == 1:
            tdp[0] = 0
        for j in range(1, length):
            if obstacleGrid[i][j] == 1:
                tdp.append(0)
            else:
                tdp.append(tdp[j - 1] + dp[j])
        dp = tdp
    return dp[length - 1]


def minPathSum(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    for i in range(1, n):
        grid[0][i] += grid[0][i - 1]
    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    return grid[m - 1][n - 1]


# https://leetcode.cn/problems/decode-ways/
def numDecodings(self, s: str) -> int:
    if s[0] == '0':
        return 0
    a, b, res = 1, 1, 1
    length = len(s)
    for i in range(1, length):
        res = 0
        if s[i] != '0':
            res += b
        if int(s[i - 1:i + 1]) <= 26 and s[i - 1] != '0':
            res += a
        a, b = b, res
    return res


# https://leetcode.cn/problems/unique-binary-search-trees/
def numTrees(self, n: int) -> int:
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    return dp[n]


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def generateTrees(self, n: int) -> List[TreeNode]:
    def generateTree(start: int, end: int) -> List[TreeNode]:
        if start > end:
            return [None]
        allTrees = []
        for i in range(start, end + 1):
            leftTrees = generateTree(start, i - 1)
            rightTrees = generateTree(i + 1, end)
            for l in leftTrees:
                for r in rightTrees:
                    t = TreeNode(i)
                    t.left = l
                    t.right = r
                    allTrees.append(t)
        return allTrees

    return generateTree(1, n)


## https://leetcode.cn/problems/interleaving-string/
def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    pdp = [True]
    for i in range(1, n + 1):
        pdp.append(pdp[i - 1] and s2[i - 1] == s3[i - 1])
    for i in range(1, m + 1):
        dp = [pdp[0] and s1[i - 1] == s3[i - 1]]
        for j in range(1, n + 1):
            dp.append((pdp[j] and s1[i - 1] == s3[i + j - 1]) or (dp[j - 1] and s2[j - 1] == s3[i + j - 1]))
        pdp = dp
    return pdp[n]


## https://leetcode.cn/problems/word-break/
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    length, sl = [], len(s)
    dp = [False] * (sl + 1)
    for i in range(0, len(wordDict)):
        length.append(len(wordDict[i]))
        if s[0:length[i]] == wordDict[i]:
            dp[length[i]] = True
    for i in range(1, sl + 1):
        for j in range(0, len(wordDict)):
            if dp[i - 1] is True and i + length[j] - 1 <= sl and s[i - 1:i + length[j] - 1] == wordDict[j]:
                dp[i + length[j] - 1] = True
    return dp[sl]


## https://leetcode.cn/problems/maximum-product-subarray/
def maxProduct(self, nums: List[int]) -> int:
    res, maxnum, minnum = nums[0], max(0, nums[0]), min(0, nums[0])
    for i in range(1, len(nums)):
        tmp = maxnum
        maxnum = max(nums[i], maxnum * nums[i], minnum * nums[i])
        minnum = min(nums[i], minnum * nums[i], tmp * nums[i])
        res = max(res, maxnum)
    return res


def rob(self, nums: List[int]) -> int:
    dp0, dp1, length = [0], [nums[0]], len(nums)
    for i in range(1, length):
        dp0.append(max(dp1[i - 1], dp0[i - 1]))
        dp1.append(dp0[i - 1] + nums[i])
    return max(dp1[length - 1], dp0[length - 1])


def rob2(self, nums: List[int]) -> int:
    dp0, dp1, length = [0], [nums[0]], len(nums)
    if length <= 2:
        return max(nums)
    for i in range(1, length - 1):
        dp0.append(max(dp1[i - 1], dp0[i - 1]))
        dp1.append(dp0[i - 1] + nums[i])
    res = max(dp1[length - 2], dp0[length - 2])
    dp0, dp1 = [0], [nums[1]]
    for i in range(2, length):
        dp0.append(max(dp1[i - 2], dp0[i - 2]))
        dp1.append(dp0[i - 2] + nums[i])
    res = max(dp1[length - 2], dp0[length - 2], res)
    return res


# https://leetcode.cn/problems/maximal-square/
def maximalSquare(self, matrix: List[List[str]]) -> int:
    m, n, res = len(matrix), len(matrix[0]), 0
    for i in range(0, m):
        for j in range(0, n):
            matrix[i][j] = int(matrix[i][j])
            res = max(res, matrix[i][j])
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            if matrix[i][j] != 0:
                matrix[i][j] = min(matrix[i + 1][j], matrix[i][j + 1], matrix[i + 1][j + 1]) + 1
            res = max(res, matrix[i][j])
    return res * res


# https://leetcode.cn/problems/ugly-number-ii/
def nthUglyNumber(self, n: int) -> int:
    dp, res = {2, 3, 5}, 1
    for i in range(1, n):
        tmp = dp.pop()
        dp.add(tmp)
        for j in dp:
            if j < tmp:
                tmp = j
        res = tmp
        dp.remove(res)
        dp.add(res * 2)
        dp.add(res * 3)
        dp.add(res * 5)
    return res


# https://leetcode.cn/problems/perfect-squares/
def numSquares(self, n: int) -> int:
    dp = [0]
    for i in range(1, n + 1):
        minNum, j = 10000, 1
        while j * j <= i:
            minNum = min(minNum, dp[i - j * j] + 1)
            j += 1
        dp.append(minNum)
    return dp[n]


# https://leetcode.cn/problems/longest-increasing-subsequence/
def lengthOfLIS(self, nums: List[int]) -> int:
    dp, res = [1], 1
    for i in range(1, len(nums)):
        pre_max = 0
        for j in range(0, i):
            if nums[i] > nums[j]:
                pre_max = max(pre_max, dp[j])
        dp.append(pre_max + 1)
        res = max(res, dp[i])
    return res


# https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/
def maxProfit(self, prices: List[int]) -> int:
    dp, l = [0], len(prices)
    for i in range(1, l):
        max_profit = 0
        for j in range(0, i):
            if prices[i] > prices[j]:
                if j >= 2:
                    max_profit = max(max_profit, prices[i] - prices[j] + dp[j - 2])
                else:
                    max_profit = max(max_profit, prices[i] - prices[j])
        dp.append(max(max_profit, dp[i - 1]))
    return dp[l - 1]


def maxProfit2(self, prices: List[int]) -> int:
    dp0, dp1, dp2 = 0, 0, -prices[0]
    for i in range(1, len(prices)):
        tmp = dp0
        dp0 = max(dp0, dp1, dp2 + prices[i])
        dp2 = max(dp1 - prices[i], dp2)
        dp1 = tmp
    return dp0


# https://leetcode.cn/problems/super-ugly-number/
def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
    l = len(primes)
    dp, points = [1], [0] * l
    for i in range(1, n):
        min_num = dp[points[0]] * primes[0]
        for j in range(1, l):
            num = dp[points[j]] * primes[j]
            if num < min_num:
                min_num = num
        dp.append(min_num)
        for p in range(0, l):
            if dp[points[p]] * primes[p] == min_num:
                points[p] += 1
    return dp[n - 1]


# https://leetcode.cn/problems/coin-change/
def coinChange(self, coins: List[int], amount: int) -> int:
    dp, l = [0], len(coins)
    for i in range(1, amount + 1):
        min_num = amount * 2
        for j in range(0, l):
            index = i - coins[j]
            if index >= 0 and dp[index] != -1:
                min_num = min(min_num, dp[index] + 1)
        if min_num == amount * 2:
            dp.append(-1)
        else:
            dp.append(min_num)
    return dp[amount]

# https://leetcode.cn/problems/house-robber-iii/
def rob3(self, root: Optional[TreeNode]) -> int:
    f, g = {}, {}
    def dfs(self, root: Optional[TreeNode]):
        if root is None:
            return
        dfs(self, root.left)
        dfs(self, root.right)
        f.update({root: root.val + max(g.get(root.left, 0), g.get(root.right, 0))})
        g.update({root: max(f.get(root.left, 0), g.get(root.left, 0)) + max(f.get(root.right, 0), g.get(root.right, 0))})
    dfs(self, root)
    return max(f.get(root, 0), g.get(root, 0))


def rob31(self, root: Optional[TreeNode]) -> int:
    def DFS(root):
        if not root:
            return 0, 0
        # 后序遍历
        leftchild_steal, leftchild_nosteal = DFS(root.left)
        rightchild_steal, rightchild_nosteal = DFS(root.right)

        # 偷当前node，则最大收益为【投当前节点+不偷左右子树】
        steal = root.val + leftchild_nosteal + rightchild_nosteal
        # 不偷当前node，则可以偷左右子树
        nosteal = max(leftchild_steal, leftchild_nosteal) + max(rightchild_steal, rightchild_nosteal)
        return steal, nosteal

    return max(DFS(root))

# https://leetcode.cn/problems/integer-break/
def integerBreak(self, n: int) -> int:
    dp = [0, 0]
    for i in range(2, n+1):
        max_sum = 0
        for j in range(i-1, 0, -1):
            max_sum = max(max_sum, (i-j) * dp[j], (i-j) * j)
        dp.append(max_sum)
    print(dp)
    return dp[n]

# https://leetcode.cn/problems/count-numbers-with-unique-digits/
def countNumbersWithUniqueDigits(self, n: int) -> int:
    dp = [1]
    for i in range(1, n+1):
        total, index = 9, 9
        for j in range(1, i):
            total *= index
            index -= 1
        dp.append(total + dp[i-1])
    return dp[n]

# https://leetcode.cn/problems/wiggle-subsequence/
def wiggleMaxLength(self, nums: List[int]) -> int:
    dp, wiggle, res = [], [], 0
    for i in range(1, len(nums)):
        wiggle.append(nums[i] - nums[i-1])
    for i in range(0, len(wiggle)):
        current_max = 0
        for j in range(0, i):
            if wiggle[i] * wiggle[j] < 0:
                current_max = max(current_max, dp[j] + 1)
        if wiggle[i] != 0:
            dp.append(max(current_max, 1))
        else:
            dp.append(current_max)
        res = max(res, dp[i])
    return res + 1

# https://leetcode.cn/problems/combination-sum-iv/
def combinationSum4(self, nums: List[int], target: int) -> int:
    dp = [1]
    for i in range(1, target+1):
        current_max = 0
        for num in nums:
            if i - num >= 0:
                current_max += dp[i-num]
        dp.append(current_max)
    return dp[target]

# https://leetcode.cn/problems/integer-replacement/
def integerReplacement1(self, n: int) -> int:
    dp = [0, 0]
    for i in range(2, n+1):
        if i % 2 == 0:
            dp.append(dp[i//2] + 1)
        else:
            dp.append(min(dp[i-1] + 1, dp[(i+1)//2] + 2))
    return dp[n]

def integerReplacement(self, n: int) -> int:
    memo = {}
    def integerReplacement2(self, n: int) -> int:
        if n == 1:
            return 0
        if memo.get(n, 0) == 0:
            if n % 2 == 0:
                memo.update({n: 1 + integerReplacement2(self, n // 2)})
            else:
                memo.update({n: 2 + min(integerReplacement2(self, n // 2), integerReplacement2(self, n // 2 + 1))})
        return memo.get(n)
    return integerReplacement2(self, n)

# https://leetcode.cn/problems/arithmetic-slices/
def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    dp, l, res = [0, 0], len(nums), 0
    if l < 3:
        return 0
    for i in range(2, l):
        if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
            dp.append(max(dp[i-1] + 1, 3))
        else:
            dp.append(0)
        res += max(0, dp[i] - 2)
    return res

# https://leetcode.cn/problems/partition-equal-subset-sum/
def canPartition(self, nums: List[int]) -> bool:
    nums_len, target, dp = len(nums), 0, {}
    for i in range(0, nums_len):
        target += nums[i]
    if nums_len < 2 or target % 2 == 1:
        return False
    def partition(self, target: int, crut_loc: int) -> bool:
        if target == 0:
            return True
        if target < 0 or crut_loc >= nums_len:
            return False
        value = dp.get(target, 0)
        if value != 0:
            return value
        dp.update({target: partition(self, target - nums[crut_loc], crut_loc + 1) or partition(self, target, crut_loc + 1)})
        return dp.get(target)
    return partition(self, target // 2, 0)

# https://leetcode.cn/problems/non-overlapping-intervals/
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    itv_len = len(intervals)
    for i in range(0, itv_len):
        index = 0
        for j in range(0, itv_len - i):
            if intervals[j][0] > intervals[index][0]:
                index = j
        tmp = intervals[index]
        intervals[index], intervals[itv_len - i - 1] = intervals[itv_len - i - 1], tmp
    dp, res = [], 0
    for i in range(0, itv_len):
        max_len = 1
        for j in range(0, i):
            if intervals[i][0] >= intervals[j][1]:
                max_len = max(max_len, dp[j] + 1)
        dp.append(max_len)
        res = max(dp[i], res)
    return itv_len - res

if __name__ == '__main__':
    intervals = [[1,3],[2,3],[3,4],[1,2], [3, 1]]
    intervals.sort()
    print(intervals)
