import cvxopt;
import numpy as np;
import sys
import numpy.linalg as la


def polyKernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def Q_matrix_1(X):
	n_samples,n_dimensions = X.shape
	K = np.zeros((n_samples,n_samples))
	for i,x_i in enumerate(X):
		for j,x_j in enumerate(X):
			K[i,j] = polyKernel(x_i,x_j)
	return K

X=[]
y=[]

file =open('nonlinsep.txt','r')
for line in file:
	X.append(map(float,line.split(",")[:2]))
	y.append(float(line.split(",")[2]))


X = np.array(X)
y = np.array(y)

K = Q_matrix_1(X)

n_samples,n_dimensions = X.shape

p = cvxopt.matrix(np.outer(y,y)*K)

q = cvxopt.matrix(np.ones(n_samples)*-1)
A = cvxopt.matrix(y,(1,n_samples))
b = cvxopt.matrix(0.0)
G = cvxopt.matrix(np.diag(np.ones(n_samples)*-1))
h= cvxopt.matrix(np.zeros(n_samples))
lag_op = cvxopt.solvers.qp(p,q,G,h,A,b)
lag_alpha = np.ravel(lag_op['x'])


support_vectors = []
support_index = []
for i in range(100):
    if lag_alpha[i] > 0.0001:
        support_index.append(i)
        support_vectors.append(np.square(X[i]))


support_vectors = np.array(support_vectors)

support_weights = lag_alpha[support_index]
x_new = X[support_index]
y_new = y[support_index]


bias = 0.0
for (y_k, x_k) in zip(y_new, x_new):
    result = bias
    for z_i, x_i, y_i in zip(support_weights, x_new, y_new):
        result += z_i * y_i * polyKernel(x_i, x_k)
    bias += y_k - np.sign(result).item()

bias = bias/len(x_new)


print "Bias", bias

print "Equation of Curve =>", str(support_weights[0]*y_new[0])+"(1+ "+str(x_new[0][0])+" *x1+"+str(x_new[0][1])+" *x2)^3+"+str(support_weights[1]*y_new[1])+"(1+ "+str(x_new[1][0])+" *x1+"+str(x_new[1][1])+" *x2)^3+"+str(support_weights[2]*y_new[2])+"(1+ "+str(x_new[2][0])+" *x1+"+str(x_new[2][1])+" *x2)^3+"+str(support_weights[3]*y_new[3])+"(1+ "+str(x_new[3][0])+" *x1+"+str(x_new[3][1])+" *x2)^3+"+str(bias)+"=0"

