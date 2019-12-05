
# 次の並びを得る

import copy

L = 4
N = 3
C = 3

A = [N] + [0] * (L - 1)


def next_list_impl(a, prev_pos, cnt):
  # 最後のが末端についていれば、
  # 1個前のものを1個進め、次のを同じ位置に立てる
  # そうでなければ最後のを1個進める

  # 最後のを探す
  if cnt == 0:
    for i in range(prev_pos - 1, -1, -1):
      if a[i] > 0:
        prev_pos, cnt = i, a[i]
        break

  pos = next_list_impl(a, prev_pos, cnt - 1) if prev_pos == L - 1 else (prev_pos + 1)
  a[prev_pos] -= 1
  a[pos] += 1
  return pos

def next_list(a):
  if a[-1] == N:
    return None
  b = copy.copy(a)
  next_list_impl(b, L, 0)
  return b


def next_phase_impl(a, prev_pos, cnt):
  if cnt == 0:
    for i in range(prev_pos - 1, -1, -1):
      if a[i] > 0:
        prev_pos, cnt = i, a[i]
        break

  pos = next_phase_impl(a, prev_pos, cnt - 1) if cnt == 1 else (prev_pos + 1)
  a[prev_pos] -= 1
  a[pos] += 1
  return pos

def next_phase(a):
  if a[N - 1] > 0:
    return None
  b = copy.copy(a)
  next_phase_impl(b, L, 0)
  return b


def next_dual_phase_impl(a, prev_pos, prev_idx, cnt):
  print(cnt)
  if cnt == 0:
    for ij in range(prev_pos * 2 + prev_idx - 1, -1, -1):
      i, j = ij // 2, ij % 2
      if a[j][i] > 0:
        prev_pos, prev_idx, cnt = i, j, a[j][i]
        break
  print(cnt, a[1 - prev_idx][prev_pos])
  if prev_idx == 1 and cnt + a[1 - prev_idx][prev_pos] == 1:
    pos, idx = next_dual_phase_impl(a, prev_pos, prev_idx, cnt - 1)
  elif prev_idx == 0:
    pos, idx = prev_pos, prev_idx + 1
  else:
    pos, idx = prev_pos + 1, 0

  a[prev_idx][prev_pos] -= 1
  a[idx][pos] += 1
  return pos, idx

def next_dual_phase(a):
  if a[1].count(1) == N:
    return None
  b = copy.copy(a)
  next_dual_phase_impl(b, L, 0, 0)
  return b


def next_multi_phase_impl(a, prev_pos, prev_idx, cnt):
  print(cnt)
  if cnt == 0:
    for ij in range(prev_pos * C + prev_idx - 1, -1, -1):
      i, j = ij // C, ij % C
      if a[j][i] > 0:
        prev_pos, prev_idx, cnt = i, j, a[j][i]
        break

  sum_cnt = cnt
  for i in range(0, C - 1):
    sum_cnt += a[i][prev_pos]

  if prev_idx == C - 1 and sum_cnt == 1:
    pos, idx = next_multi_phase_impl(a, prev_pos, prev_idx, cnt - 1)
  elif prev_idx < C - 1:
    pos, idx = prev_pos, prev_idx + 1
  else:
    pos, idx = prev_pos + 1, 0

  a[prev_idx][prev_pos] -= 1
  a[idx][pos] += 1
  return pos, idx

def next_multi_phase(a):
  if a[-1].count(1) == N:
    return None
  b = copy.copy(a)
  next_multi_phase_impl(b, L, 0, 0)
  return b


#a = [copy.copy(A), [0] * L]
a = [copy.copy(A)] + [[0] * L for _ in range(C - 1)]

while a is not None:
  print(a)
  #a = next_dual_phase(a)
  a = next_multi_phase(a)







