# https://slideship.com/users/@massa142/presentations/2018/09/56hjwyTVtpSefg2dnTfKew/
# https://qiita.com/tibigame/items/84bfd46df494e05c711e

import sympy
from sympy import diff, symbols, Symbol, Sum

sympy.init_printing(use_unicode=False, wrap_line=True)

(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) = sympy.symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')
Y = x + x + 1
print(Y)

# 微分
dYdx = diff(Y, x)
print(dYdx)

# 級数
Sx = Sum(n ** x, (n, 1, k))
print(Sx)

# 