# λ収益

a = [0.1, 0.2, 0.3, 0.4, 0.5]

b = a + [0.5] * 100

lmbda = 0.8

def lambda_return(l):
    if len(l) == 1:
        return [l[0]]
    else:
        ll = lambda_return(l[1:])
        return [l[0] * (1 - lmbda) + ll[0] * lmbda] + ll

print(lambda_return(a)[:10])
print(lambda_return(b)[:10])

# 即時報酬ありの場合

r = [0.1, 0, 0.2, 0.4]
gamma = 0.9

def compute_return(l):
    if len(l) == 1:
        return [l[0]]
    else:
        ll = compute_return(l[1:])
        return [l[0] + ll[0] * gamma] + ll

print(compute_return(r))
print(sum(compute_return(r)))

def lambda_return_with_gamma(l):
    if len(l) == 1:
        return [l[0]]
    else:
        ll = lambda_return_with_gamma(l[1:])
        ret = l[0] + ll[0] * gamma
        return [ret * (1 - lmbda) + ll[0] * lmbda] + ll

print(lambda_return_with_gamma(r))
print(lambda_return(compute_return(r)))

print(lambda_return_with_gamma(r + [0] * 100)[:5])
print(lambda_return(compute_return(r + [0] * 100))[:5])

