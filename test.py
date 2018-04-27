from __future__ import division

class A():
	def __init__(self):
		print('calling A.__init__')

	def foo(self):
		print('calling A.foo')


a = None

def func1():
	print('calling func1')
	a.foo()


if a is None:
	a = A()
	
