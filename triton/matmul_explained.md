# Triton Matmul 小学生版讲解

这份文档专门解释 `triton/matmul.py` 里最绕的部分：

- 为什么要提 L2 cache
- 为什么要用 `GROUP_M`
- `pid`、`pid_m`、`pid_n` 到底是什么
- 后面的 index 为什么这么算
- `a_desc.load(...)`、`b_desc.load(...)`、`c_desc.store(...)` 在拿哪块数据

配套动画页面：打开 [`matmul_group_visualization.html`](./matmul_group_visualization.html)，可以看到普通顺序和 `GROUP_M = 2` 的执行顺序差异。

先记住一句话：

> 这个 kernel 不是一次算完整个矩阵，而是把结果矩阵 `C` 切成很多小方块。每个 Triton program 只负责算一个小方块。

## 1. 先忘掉 GPU，想象在切蛋糕

矩阵乘法是：

```text
C = A @ B
```

形状是：

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

你可以把 `C` 想象成一张大蛋糕。

Triton 不会让一个人吃完整张蛋糕。它会把蛋糕切成很多小块：

```text
C 被切成很多 tile:

        n=0     n=1     n=2
m=0   C[0,0]  C[0,1]  C[0,2]
m=1   C[1,0]  C[1,1]  C[1,2]
m=2   C[2,0]  C[2,1]  C[2,2]
m=3   C[3,0]  C[3,1]  C[3,2]
```

这里：

- `m` 表示第几行 tile
- `n` 表示第几列 tile
- `C[m,n]` 表示结果矩阵 `C` 的一小块

每个 Triton program 就像一个小工人。

```text
一个 program = 一个工人
一个工人 = 算一个 C tile
```

所以代码里的：

```python
pid = tl.program_id(0)
```

意思是：

```text
当前这个工人的编号是多少？
```

比如：

```text
pid = 0 是第 0 个工人
pid = 1 是第 1 个工人
pid = 2 是第 2 个工人
```

但是光知道 `pid = 7` 没用。我们真正想知道的是：

```text
第 7 个工人应该算 C 的哪一块？
```

也就是要从一维编号 `pid` 算出二维坐标：

```text
pid -> (pid_m, pid_n)
```

## 2. `num_pid_m` 和 `num_pid_n` 是什么

代码：

```python
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
```

意思是：

```text
num_pid_m = C 在 M 方向被切成多少行 tile
num_pid_n = C 在 N 方向被切成多少列 tile
```

例如：

```text
M = 1000
BLOCK_M = 128
```

那 M 方向大概需要：

```text
ceil(1000 / 128) = 8
```

也就是：

```text
num_pid_m = 8
```

这里用 `tl.cdiv` 是向上取整。因为最后一块可能不满，但也要有一个 program 去处理。

比如 1000 行，每块 128 行：

```text
前 7 块: 128 行
最后 1 块: 剩下 104 行
```

最后那块虽然不满，也必须算。

## 3. 最普通的顺序是什么

假设结果矩阵 `C` 被切成：

```text
num_pid_m = 4
num_pid_n = 3
```

也就是：

```text
        n=0     n=1     n=2
m=0   C[0,0]  C[0,1]  C[0,2]
m=1   C[1,0]  C[1,1]  C[1,2]
m=2   C[2,0]  C[2,1]  C[2,2]
m=3   C[3,0]  C[3,1]  C[3,2]
```

最容易想到的顺序是按行走：

```text
pid 0  -> C[0,0]
pid 1  -> C[0,1]
pid 2  -> C[0,2]
pid 3  -> C[1,0]
pid 4  -> C[1,1]
pid 5  -> C[1,2]
pid 6  -> C[2,0]
pid 7  -> C[2,1]
pid 8  -> C[2,2]
pid 9  -> C[3,0]
pid 10 -> C[3,1]
pid 11 -> C[3,2]
```

这个顺序很好懂，但不一定最快。

因为它对 L2 cache 不太友好。

## 4. L2 cache 是什么，说人话

GPU 读数据有很多层。

你可以想象成：

```text
显存: 很大的仓库，但是离工人远，拿东西慢
L2 cache: 小一点的临时货架，离工人近，拿东西快
```

如果一个工人刚刚把一箱牛奶从大仓库搬到临时货架上，下一个工人也要同一箱牛奶，那就赚了。

因为第二个工人不用再跑去大仓库。

这就是 cache 的意义：

```text
最近用过的数据，如果马上又要用，就可能还在 cache 里。
```

所以我们希望：

```text
会用同一块数据的 program 尽量挨着执行。
```

## 5. 算一个 C tile 需要哪些数据

要算 `C[m,n]` 这一块，需要：

```text
A 的第 m 行 tile
B 的第 n 列 tile
```

更具体一点，先不要看一整块 tile，只看 `C` 里面的一个小格子。

比如要算：

```text
C[2, 5]
```

它不是凭空来的。它来自：

```text
A 的第 2 行
B 的第 5 列
```

也就是：

```text
C[2,5] =
  A[2,0] * B[0,5]
+ A[2,1] * B[1,5]
+ A[2,2] * B[2,5]
+ A[2,3] * B[3,5]
+ ...
```

注意中间那个数字一直在变：

```text
A[2,0] * B[0,5]
A[2,1] * B[1,5]
A[2,2] * B[2,5]
```

这个一直变化的中间数字，就是 `K` 方向。

所以矩阵乘法本质上是：

```text
固定 C 的行 m
固定 C 的列 n
沿着 K 方向一路乘过去，然后加起来
```

写成抽象形式就是：

```text
C[m,n] =
  A[m, k=0] * B[k=0, n]
+ A[m, k=1] * B[k=1, n]
+ A[m, k=2] * B[k=2, n]
+ ...
```

上面只是一个小格子。

但 Triton 一个 program 不只算一个小格子，它一次算一小块 `C`。

比如一个 program 要算：

```text
C tile: [BLOCK_M, BLOCK_N]
```

也就是：

```text
BLOCK_M 行
BLOCK_N 列
```

为了算这一整块 `C tile`，它每次会拿：

```text
A tile: [BLOCK_M, BLOCK_K]
B tile: [BLOCK_K, BLOCK_N]
```

为什么形状是这样？

因为：

```text
A 要提供 C tile 的这些行，所以 A 要拿 BLOCK_M 行。
B 要提供 C tile 的这些列，所以 B 要拿 BLOCK_N 列。
中间相乘累加沿着 K 方向走，但一次只走 BLOCK_K 这么长。
```

所以一次小乘法长这样：

```text
[BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
```

也就是：

```text
A 的一小块 @ B 的一小块 = C 的一小块贡献
```

注意这里说的是“一小块贡献”，不是完整答案。

因为 K 方向通常很长，`BLOCK_K` 只是其中一段。

所以一个 `C[m,n]` tile 会反复加载很多次：

```text
第 0 段 K: A tile [BLOCK_M, BLOCK_K] 和 B tile [BLOCK_K, BLOCK_N]
第 1 段 K: A tile [BLOCK_M, BLOCK_K] 和 B tile [BLOCK_K, BLOCK_N]
第 2 段 K: A tile [BLOCK_M, BLOCK_K] 和 B tile [BLOCK_K, BLOCK_N]
...
```

对应代码：

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
    b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
    acc = tl.dot(a, b, acc)
```

这段的意思是：

```text
沿着 K 方向，一块一块拿 A 和 B。
每次拿一块 A、一块 B，做一次小矩阵乘法。
把结果加到 acc 里。
```

## 6. 为什么普通顺序不够好

普通按行顺序是：

```text
C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2], ...
```

看前三个：

```text
C[0,0] 需要 A[0,:] 和 B[:,0]
C[0,1] 需要 A[0,:] 和 B[:,1]
C[0,2] 需要 A[0,:] 和 B[:,2]
```

优点：

```text
A[0,:] 可以复用
```

缺点：

```text
B 每次都换一列
```

也就是说：

```text
刚刚用完 B[:,0]
马上就不用它了
换成 B[:,1]
再换成 B[:,2]
```

等到下次需要 `B[:,0]`，已经是 `C[1,0]` 了：

```text
C[0,0] 用 B[:,0]
C[0,1] 用 B[:,1]
C[0,2] 用 B[:,2]
C[1,0] 又用 B[:,0]
```

中间隔了一段时间，`B[:,0]` 可能已经被 L2 cache 挤掉了。

## 7. `GROUP_M` 想解决什么

`GROUP_M` 的想法是：

```text
不要一整行一整行算。
先拿几行 M tile 组成一个小组。
在这个小组里，让相邻 program 尽量用同一列 B。
```

假设：

```text
GROUP_M = 2
```

那么顺序从普通的：

```text
C[0,0], C[0,1], C[0,2],
C[1,0], C[1,1], C[1,2],
C[2,0], C[2,1], C[2,2],
C[3,0], C[3,1], C[3,2]
```

先停一下。

这个普通顺序的意思是：

```text
先把第 0 行全部算完。
再把第 1 行全部算完。
再把第 2 行全部算完。
```

也就是横着走：

```text
先 C[0,0]，再 C[0,1]，再 C[0,2]
```

问题是：

```text
C[0,0] 用 B[:,0]
C[0,1] 用 B[:,1]
C[0,2] 用 B[:,2]
```

它们用的是不同的 B。

如果我们想复用 `B[:,0]`，更好的选择是让下面这些挨在一起：

```text
C[0,0] 用 B[:,0]
C[1,0] 用 B[:,0]
```

因为它们的列号 `n` 都是 0。

再看：

```text
C[0,1] 用 B[:,1]
C[1,1] 用 B[:,1]
```

它们的列号 `n` 都是 1，也能复用同一块 B。

所以 `GROUP_M = 2` 的意思就是：

```text
先把 m=0 和 m=1 这两行绑成一个小组。
在这个小组里，不要横着一行一行走。
而是同一列 n 先算两行，再换下一列。
```

变成：

```text
C[0,0], C[1,0],
C[0,1], C[1,1],
C[0,2], C[1,2],

C[2,0], C[3,0],
C[2,1], C[3,1],
C[2,2], C[3,2]
```

把它画成表更明显。

普通顺序是：

```text
        n=0     n=1     n=2
m=0      0       1       2
m=1      3       4       5
m=2      6       7       8
m=3      9       10      11
```

数字表示第几个 program 去算。

`GROUP_M = 2` 后变成：

```text
        n=0     n=1     n=2
m=0      0       2       4
m=1      1       3       5

m=2      6       8       10
m=3      7       9       11
```

你可以看到：

```text
program 0 算 C[0,0]
program 1 算 C[1,0]
```

这两个挨着，而且都用 `B[:,0]`。

然后：

```text
program 2 算 C[0,1]
program 3 算 C[1,1]
```

这两个也挨着，而且都用 `B[:,1]`。

所以这里不是数学变了。

数学结果完全一样。

只是安排工人干活的顺序变了：

```text
普通顺序: 先横着走
group 顺序: 每 GROUP_M 行一组，在组里面先竖着走
```

注意前两个：

```text
C[0,0] 需要 B[:,0]
C[1,0] 也需要 B[:,0]
```

它们挨着执行，所以 `B[:,0]` 更可能还在 L2 cache 里。

这就是 grouped ordering。

## 8. 一句话解释 group

普通顺序：

```text
先横着走。
```

group 顺序：

```text
每次拿 GROUP_M 行，先在这几行里面竖着走，再换下一列。
```

画出来：

```text
GROUP_M = 2

先算这两行:

        n=0     n=1     n=2
m=0     0       2       4
m=1     1       3       5

再算下面两行:

        n=0     n=1     n=2
m=2     6       8       10
m=3     7       9       11
```

这里数字是 `pid`。

你会看到：

```text
pid 0 和 pid 1 都是 n=0
pid 2 和 pid 3 都是 n=1
pid 4 和 pid 5 都是 n=2
```

所以相邻 program 会更容易复用同一块 B。

## 9. 逐行解释 group 公式

代码：

```python
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

local_pid = pid % num_pid_in_group
pid_m = first_pid_m + (local_pid % group_size_m)
pid_n = local_pid // group_size_m
```

我们用具体数字解释。

假设：

```text
num_pid_m = 5
num_pid_n = 3
GROUP_M = 2
```

也就是：

```text
C 有 5 行 tile
C 有 3 列 tile
每个 group 放 2 行 tile
```

### 9.1 一个 group 里有几个 program

代码：

```python
num_pid_in_group = GROUP_M * num_pid_n
```

代入：

```text
num_pid_in_group = 2 * 3 = 6
```

意思是：

```text
一个 group 里面有 6 个 program。
```

因为一个 group 有：

```text
2 行 tile * 3 列 tile = 6 个 C tile
```

### 9.2 当前 pid 属于哪个 group

代码：

```python
group_id = pid // num_pid_in_group
```

如果 `num_pid_in_group = 6`：

```text
pid 0..5   属于 group 0
pid 6..11  属于 group 1
pid 12..17 属于 group 2
```

例如：

```text
pid = 8
group_id = 8 // 6 = 1
```

所以 `pid = 8` 在第 1 个 group。

注意这里 group 从 0 开始数。

### 9.3 这个 group 从哪一行 m 开始

代码：

```python
first_pid_m = group_id * GROUP_M
```

如果：

```text
GROUP_M = 2
```

那么：

```text
group 0 从 m=0 开始
group 1 从 m=2 开始
group 2 从 m=4 开始
```

例如：

```text
pid = 8
group_id = 1
first_pid_m = 1 * 2 = 2
```

所以 `pid = 8` 所在的 group 管的是：

```text
m = 2 和 m = 3 这两行 tile
```

### 9.4 为什么有 `group_size_m`

代码：

```python
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
```

这个是为了处理最后一组不满的情况。

刚才假设：

```text
num_pid_m = 5
GROUP_M = 2
```

那么分组是：

```text
group 0: m=0, m=1
group 1: m=2, m=3
group 2: m=4
```

最后 `group 2` 只有一行，不是两行。

所以不能永远假设 group 里有 `GROUP_M` 行。

要用：

```text
group_size_m = 当前 group 真实有几行
```

例如：

```text
group 0: group_size_m = 2
group 1: group_size_m = 2
group 2: group_size_m = 1
```

### 9.5 `local_pid` 是什么

代码：

```python
local_pid = pid % num_pid_in_group
```

意思是：

```text
当前 pid 在这个 group 里面是第几个？
```

例如：

```text
pid = 8
num_pid_in_group = 6

local_pid = 8 % 6 = 2
```

所以：

```text
pid 8 是 group 1 里的第 2 个 program
```

看完整表：

```text
pid 6  -> local_pid 0
pid 7  -> local_pid 1
pid 8  -> local_pid 2
pid 9  -> local_pid 3
pid 10 -> local_pid 4
pid 11 -> local_pid 5
```

### 9.6 为什么 `pid_m` 用 `%`

代码：

```python
pid_m = first_pid_m + (local_pid % group_size_m)
```

这里的 `%` 是取余数。

它的作用是：

```text
让 m 在 group 里面循环。
```

假设：

```text
first_pid_m = 2
group_size_m = 2
```

那么：

```text
local_pid 0: local_pid % 2 = 0 -> pid_m = 2 + 0 = 2
local_pid 1: local_pid % 2 = 1 -> pid_m = 2 + 1 = 3
local_pid 2: local_pid % 2 = 0 -> pid_m = 2 + 0 = 2
local_pid 3: local_pid % 2 = 1 -> pid_m = 2 + 1 = 3
local_pid 4: local_pid % 2 = 0 -> pid_m = 2 + 0 = 2
local_pid 5: local_pid % 2 = 1 -> pid_m = 2 + 1 = 3
```

所以 `pid_m` 会这样跳：

```text
2, 3, 2, 3, 2, 3
```

也就是在 group 的两行之间来回切。

### 9.7 为什么 `pid_n` 用 `//`

代码：

```python
pid_n = local_pid // group_size_m
```

这里的 `//` 是整数除法。

它的作用是：

```text
每过 group_size_m 个 program，n 才加 1。
```

假设：

```text
group_size_m = 2
```

那么：

```text
local_pid 0: 0 // 2 = 0
local_pid 1: 1 // 2 = 0
local_pid 2: 2 // 2 = 1
local_pid 3: 3 // 2 = 1
local_pid 4: 4 // 2 = 2
local_pid 5: 5 // 2 = 2
```

所以 `pid_n` 是：

```text
0, 0, 1, 1, 2, 2
```

这就正好得到：

```text
local_pid 0 -> m=2, n=0
local_pid 1 -> m=3, n=0
local_pid 2 -> m=2, n=1
local_pid 3 -> m=3, n=1
local_pid 4 -> m=2, n=2
local_pid 5 -> m=3, n=2
```

这就是我们想要的顺序：

```text
C[2,0], C[3,0],
C[2,1], C[3,1],
C[2,2], C[3,2]
```

## 10. 完整 pid 映射例子

继续用：

```text
num_pid_m = 5
num_pid_n = 3
GROUP_M = 2
```

完整映射是：

```text
pid 0  -> C[0,0]
pid 1  -> C[1,0]
pid 2  -> C[0,1]
pid 3  -> C[1,1]
pid 4  -> C[0,2]
pid 5  -> C[1,2]

pid 6  -> C[2,0]
pid 7  -> C[3,0]
pid 8  -> C[2,1]
pid 9  -> C[3,1]
pid 10 -> C[2,2]
pid 11 -> C[3,2]

pid 12 -> C[4,0]
pid 13 -> C[4,1]
pid 14 -> C[4,2]
```

最后一组只有 `m=4` 一行，所以：

```text
pid 12 -> C[4,0]
pid 13 -> C[4,1]
pid 14 -> C[4,2]
```

## 11. 后面的 load index 为什么这么算

代码：

```python
a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
```

先看 A。

`A` 的形状是：

```text
A[M, K]
```

`A` 的 tile 形状是：

```text
[BLOCK_M, BLOCK_K]
```

所以：

```python
a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
```

意思是：

```text
从 A 的第 pid_m 个 M 块、第 k 个 K 块开始加载。
```

如果：

```text
pid_m = 2
BLOCK_M = 64
k = 3
BLOCK_K = 32
```

那么 A tile 的左上角是：

```text
row = 2 * 64 = 128
col = 3 * 32 = 96
```

也就是加载：

```text
A[128 : 128+64, 96 : 96+32]
```

再看 B。

`B` 的形状是：

```text
B[K, N]
```

`B` 的 tile 形状是：

```text
[BLOCK_K, BLOCK_N]
```

所以：

```python
b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
```

意思是：

```text
从 B 的第 k 个 K 块、第 pid_n 个 N 块开始加载。
```

如果：

```text
k = 3
BLOCK_K = 32
pid_n = 1
BLOCK_N = 128
```

那么 B tile 的左上角是：

```text
row = 3 * 32 = 96
col = 1 * 128 = 128
```

也就是加载：

```text
B[96 : 96+32, 128 : 128+128]
```

## 12. 为什么 A 用 `pid_m`，B 用 `pid_n`

因为：

```text
C[m,n] = A[m,:] @ B[:,n]
```

所以：

```text
C 的行 m 决定要用 A 的哪几行
C 的列 n 决定要用 B 的哪几列
```

对应到代码：

```text
pid_m 决定 A 的行块
pid_n 决定 B 的列块
```

所以：

```python
a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
```

用 `pid_m`。

```python
b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
```

用 `pid_n`。

## 13. `k` 是在干嘛

矩阵乘法要沿着 K 方向做累加。

想象成拼乐高：

```text
C[m,n] 的最终结果不是一下子出来的。
它是很多小结果加起来的。
```

每次循环：

```text
拿 A 的一段 K
拿 B 的同一段 K
做一次小乘法
加到 acc
```

代码：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
    b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])
    acc = tl.dot(a, b, acc)
```

`acc` 就是当前这个 C tile 的累计结果。

如果 `K` 被切成 4 块，那么它大概是：

```text
acc = 0
acc += A[m,k0] @ B[k0,n]
acc += A[m,k1] @ B[k1,n]
acc += A[m,k2] @ B[k2,n]
acc += A[m,k3] @ B[k3,n]
```

最后 `acc` 就是完整的 `C[m,n]` tile。

## 14. store index 为什么这么算

代码：

```python
c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(OUT_DTYPE))
```

意思是：

```text
把算好的 acc 存回 C 的第 pid_m 行 tile、第 pid_n 列 tile。
```

如果：

```text
pid_m = 2
pid_n = 1
BLOCK_M = 64
BLOCK_N = 128
```

那么存到：

```text
C[128 : 128+64, 128 : 128+128]
```

也就是 `C[2,1]` 这个 tile。

## 15. 用一句话串起来

这个 kernel 的流程是：

```text
1. Triton 给我一个一维 program 编号 pid。
2. 我用 grouped ordering 把 pid 转成二维 tile 坐标 pid_m, pid_n。
3. 这个 program 只负责算 C[pid_m, pid_n] 这一小块。
4. 沿着 K 方向循环加载 A tile 和 B tile。
5. 用 tl.dot 一点点累加到 acc。
6. 最后把 acc 存回 C 对应的位置。
```

## 16. 最容易卡住的点

### 16.1 group 不是为了正确性

不用 group 也能算对。

group 是为了更快。

它改变的是：

```text
program 访问 C tile 的顺序
```

不是改变数学公式。

### 16.2 L2 cache 不是你手动放进去的

代码里没有写：

```text
put B into L2 cache
```

GPU 会自动缓存最近读过的数据。

程序员能做的是：

```text
安排访问顺序，让刚读过的数据尽快再次被使用。
```

### 16.3 `%` 和 `//` 是把一维编号拆成二维坐标

这很像把学生按座位排成几列。

如果每列有 2 个座位：

```text
local_pid 0 -> 第 0 列，第 0 行
local_pid 1 -> 第 0 列，第 1 行
local_pid 2 -> 第 1 列，第 0 行
local_pid 3 -> 第 1 列，第 1 行
```

公式就是：

```text
行 = local_pid % 每列人数
列 = local_pid // 每列人数
```

在代码里：

```text
行 = local_pid % group_size_m
列 = local_pid // group_size_m
```

## 17. 最后再看原代码

原代码：

```python
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)

num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M

group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

local_pid = pid % num_pid_in_group
pid_m = first_pid_m + (local_pid % group_size_m)
pid_n = local_pid // group_size_m
```

现在可以翻译成中文：

```text
pid:
  当前工人的编号。

num_pid_m:
  C 在行方向被切成多少块。

num_pid_n:
  C 在列方向被切成多少块。

num_pid_in_group:
  一个 group 里面有多少个 C tile。

group_id:
  当前工人属于第几个 group。

first_pid_m:
  当前 group 从第几行 tile 开始。

group_size_m:
  当前 group 真实有几行 tile，最后一组可能不满。

local_pid:
  当前工人在 group 里面的局部编号。

pid_m:
  当前工人负责的 C tile 行号。

pid_n:
  当前工人负责的 C tile 列号。
```

## 18. 记忆口诀

可以这样记：

```text
pid 是工号。
pid_m / pid_n 是座位号。
group 是一排一排安排座位的方法。
L2 cache 是离工人很近的临时货架。
GROUP_M 是一次把几行工人绑在一起干活。
```

最终目的：

```text
让相邻工人尽量用同一块 B，
让 B 留在 L2 cache 里，
少跑去显存大仓库搬数据，
所以更快。
```
