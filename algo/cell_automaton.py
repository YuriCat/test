
N = 31
T = 40
A = [0] * N

A[N // 2] = 1
for t in range(T):
    print(''.join(map(lambda x:('*' if x > 0 else '.'), A)))
    B = []
    for i in range(N):
        s = A[(i+N-1)%N] * 4 + A[i] * 2 + A[(i+1)%N]
        B.append(1 if (1 <= s and s <= 4) else 0)
    A = B
