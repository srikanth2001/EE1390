import numpy as np
import matplotlib.pyplot as plt

def dvec(A,B):
    return B-A

def nvec(A,B):
    return np.dot(omat,dvec(A,B))

def midpt(A,B):
    return (A+B)/2.0
   
def line_int(n1,p1,n2,p2):
    A = np.linalg.inv(np.vstack((n1,n2)))
    p = np.zeros((2, 1))
    p[0] = p1
    p[1] = p2
    return np.matmul(A,p)

omat = np.array([[0,-1],[1,0]])
   
A = np.array([2,3])
B = np.array([4,5])


C = midpt(A,B)


n1 = np.array([-1,4])
n2 = dvec(A,B)
p1 = -3
p2 = np.dot(n2,C)

O = line_int(n1,p1,n2,p2)
print(O)
a= O[0,0]
b= O[1,0]
D = np.array([a, b])


r = np.linalg.norm(D - A)
print(r)


t = np.linspace(0, 2*np.pi, 1000)
x = a + (r*np.cos(t))
y = b + (r*np.sin(t))

plt.plot(x, y)
plt.xlabel('$x$')
plt.ylabel('$y$')

x1 = np.linspace(O[0][0] - (r+1), O[0][0] + (r+1),100)
y1 = 0.25*x1 - 0.75
plt.plot(x1,y1, label = '$Given-line$')

plt.plot(2,3,'o')
plt.text(2.1, 3.1, 'A')
plt.plot(4,5,'o')
plt.text(4.1, 5.1, 'B')
plt.text(-0.43,- 2.7, 'radius = 5.84')

plt.plot(O[0,0],O[1,0],'o')
plt.text(D[0] + 0.1, D[1] + 0.1, '$Centre$')

plt.axis('equal')
plt.title('Plot of the circle')
plt.legend(loc = 'best')
plt.grid()
plt.show()
