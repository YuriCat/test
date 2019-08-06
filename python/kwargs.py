def func(*args, **kwargs):
  print(args)
  print(kwargs)

func(a=0, b=1)
func(1, a=0, b=1)
