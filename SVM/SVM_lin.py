import numpy as np
import cvxopt.solvers


def linear_kernel(x_i,x_j):
	return np.dot(x_i,x_j)


def Q_matrix_1(X):
	n_samples,n_dimensions = X.shape
	K = np.zeros((n_samples,n_samples))
	for i,x_i in enumerate(X):
		for j,x_j in enumerate(X):
			K[i,j] = linear_kernel(x_i,x_j)
	return K

def langrange_multipliers(X,y):
	n_samples,n_dimensions = X.shape
	K = Q_matrix_1(X)

	#min 1/2 alpha.T*P*alpha +q.T alpha
	#s.t y.T*alpha = 0, aplha>=0

	q = cvxopt.matrix(-1*np.ones(n_samples))
	P = cvxopt.matrix(np.outer(y,y)*K)

	A = cvxopt.matrix(y,(1,n_samples))
	
	b = cvxopt.matrix(0.0)

	G = cvxopt.matrix(np.diag(-1*np.ones(n_samples)))
	h = cvxopt.matrix(np.zeros(n_samples))

	

	solution = cvxopt.solvers.qp(P,q,G,h,A,b)

	return np.ravel(solution['x'])

def compute_w(langrange_multipliers,X,y):
	return np.sum(langrange_multipliers[i]*y[i]*X[i] for i in range(len(y)))

def compute_b(w,X,y):
	return np.sum([y[i]-np.dot(w,X[i]) for i in range(len(X))])/len(X)
	
#def prediction(w,X,b):
	

X=[]
y=[]

file =open('linsep.txt','r')
for line in file:
	X.append(map(float,line.split(",")[:2]))
	y.append(float(line.split(",")[2]))


X = np.array(X)
y = np.array(y)

alpha = langrange_multipliers(X,y)

has_positive_value = alpha > 1e-8
support_multipliers = alpha[has_positive_value]
support_vectors_X = X[has_positive_value]
support_vectors_y = y[has_positive_value]

w = compute_w(alpha,X,y)
w_from_sv = compute_w(support_multipliers,support_vectors_X,support_vectors_y)

b = compute_b(w_from_sv,support_vectors_X,support_vectors_y)

print 'Weights = ', w

print "Bias", b

print "Equation of Curve =>", str(w[0])+"*(x1)+ "+str(w[1])+"*(x2) +"+str(b)+"=0"


