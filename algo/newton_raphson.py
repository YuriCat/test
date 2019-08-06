# ニュートン・ラプソン法

def f(x):
  return (x ** 2) - 2
def f1(x):
  return 2 * x;

def newton(x0, epochs = 10):
  x = x0
  print(x0)
  for _ in range(epochs):
    x = x - f(x) / f1(x)
    print(x)

newton(2)
newton(-2)
newton(1)