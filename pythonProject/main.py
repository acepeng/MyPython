# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def countBits(self, n: int) -> List[int]:
    res = [0]
    if n == 0:
        return res
    res.append(1)
    i = x = 2
    while i <= n:
        if i >= (2 * x):
            x *= 2
        res.append(res[i - 2 * x] + 1)
        i += 1
    return res


# 偶数就赢，与前一个数是输还是赢相关
def divisorGame(self, n: int) -> bool:
    res = [False]
    flag = False
    for i in range(2, n + 1):
        j = 1
        while j < i:
            if i % j == 0 and not res[i - j - 1]:
                flag = True
                break
            j += 1
        res.append(flag)
        flag = False
    return res[n - 1]


def tribonacci(self, n: int) -> int:
    if n == 0:
        return 0
    if n <= 2:
        return 1
    res = a = 0
    b = c = 1
    while n > 2:
        n -= 1
        res = a + b + c
        a, b, c = b, c, res
    return res


def getMaximumGenerated(self, n: int) -> int:
    resList = [0, 1]
    if n <= 1:
        return resList[n]
    res = i = 1
    while i < (n + 1) // 2:
        resList.append(resList[i])
        resList.append(resList[i] + resList[i + 1])
        res = max(resList[i] + resList[i + 1], res, resList[i])
        i += 1
    if n % 2 == 0:
        res = max(resList[i], res)
    return res


def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
    pre_dp = [0] * n
    pre_dp[0] = 1
    for i in range(k):
        dp = [0] * n
        for edge in relation:
            dp[edge[1]] += pre_dp[edge[0]]
        pre_dp = dp
    return dp[n - 1]

def leastMinutes(self, n: int) -> int:
    i = res = 1
    while i < n:
        i *= 2
        res += 1
    return res

def fib(self, n: int) -> int:
    if n <= 1:
        return n
    a, b, res = 0, 1, 0
    for i in range(2, n+1):
        res = (a+b) % 1000000007
        a, b = b, res
    return res

def numWays(self, n: int) -> int:
    if n <= 1:
        return 1
    a, b, res = 1, 1, 0
    for i in range(2, n+1):
        res = (a+b) % 1000000007
        a, b = b, res
    return res

# https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
def maxSubArray(self, nums: List[int]) -> int:
    res = nums[0]
    for i in range(1, len(nums)):
        nums[i] = max(nums[i]+nums[i-1], nums[i])
        res = max(res, nums[i])
    return res

def countBits2(self, n: int) -> List[int]:
    res = [0]
    if n == 0:
        return res
    res.append(1)
    index = 2
    for i in range(2, n+1):
        if i == 2*index:
            index *= 2
        res.append(res[i-index]+1)
    return res

def minCostClimbingStairs(self, cost: List[int]) -> int:
    length = len(cost)
    a, b = 0, 0
    for i in range(2, length + 1):
        temp = min(b + cost[i - 1], a + cost[i - 2])
        a, b = b, temp
    return b

# https://leetcode.cn/problems/reverse-bits-lcci/
def reverseBits(self, num: int) -> int:
    dp = [0]
    i = 1
    while num != 0:
        if num % 2 == 0:
            dp.append(i)
        num //= 2
        i += 1
    dp.append(i)
    res = 0
    for i in range(1, len(dp)-1):
        res = max(res, dp[i+1]-dp[i-1]-1)
    if res == 0:
        return i
    return res

# https://leetcode.cn/problems/three-steps-problem-lcci/
def waysToStep(self, n: int) -> int:
    if n <= 2:
        return n
    a, b, c, res = 1, 1, 2, 0
    for i in range(3, n+1):
        res = (a+b+c) % 1000000007
        a, b, c = b, c, res
    return res

# https://leetcode.cn/problems/the-masseuse-lcci/
def massage(self, nums: List[int]) -> int:
    if len(nums) == 0:
        return 0
    length = len(nums)
    dp0, dp1 = 0, nums[0]
    for i in range(1, length):
        tdp0 = max(dp0, dp1)
        tdp1 = dp0 + nums[i]
        dp0, dp1 = tdp0, tdp1
    return max(dp0, dp1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cost = [2,1,4,5,3,1,1,3]
    print(massage(1, cost))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
