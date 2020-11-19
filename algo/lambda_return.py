# λ収益

a = [0.1, 0.2, 0.3, 0.4, 0.5]

b = a + [0.5] * 100

lmbda = 0.7

def lambda_return(l):
    if len(l) == 1:
        return [l[0]]
    else:
        ll = lambda_return(l[1:])
        return [l[0] * (1 - lmbda) + ll[0] * lmbda] + ll

print('terminal reward')
print(lambda_return(a)[:10])
print(lambda_return(b)[:10])

# 即時報酬ありの場合

r = [0.1, 0, 0.2, 0.4]
gamma = 0.9
bootstrap_value = 0.2 / (1 - gamma)

def compute_return(l):
    if len(l) == 1:
        return [l[0]]
    else:
        ll = compute_return(l[1:])
        return [l[0] + ll[0] * gamma] + ll

print('immediate reward')
print(r, bootstrap_value, '->')
print(compute_return(r))
print(sum(compute_return(r)))

def lambda_return_with_gamma(l, bv):
    if len(l) == 0:
        return [bv]
    else:
        ll = lambda_return_with_gamma(l[1:], bv)
        ret = l[0] + ll[0] * gamma
        return [ret * (1 - lmbda) + ll[0] * lmbda] + ll

print(lambda_return_with_gamma(r, bootstrap_value))
print(sum(lambda_return_with_gamma(r, bootstrap_value)))
print(lambda_return(compute_return(r)))

print(lambda_return_with_gamma(r + [0] * 100, bootstrap_value)[:5])
print(lambda_return(compute_return(r + [0] * 100))[:5])

