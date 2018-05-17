from mvn import Mvn
import numpy as np
from numpy.linalg import inv
import time
import pickle

class ADFIMM:

    def __init__(self, alpha, mean_base, pres_base, precision_fixed):
        #Dirichlet Hyperparameters
        self.alpha = alpha
        self.innovation = Mvn(mean_base, pres_base)

        #fixed variance hyperparameter
        self.precision_fixed = precision_fixed
        #self.precision_fixed = inv(np.eye(len(mean_base))*10000)
        self.covariance_fixed = inv(self.precision_fixed)

        #useful hyperparameters
        self.dim = len(mean_base)
        self.covar_type = "fixed"     
        self.convergence_threshold = 0.01
        

    def factorise_prior(self, data):
        n , d = data.shape
        
        #each factor has 
        q_i = [None] * n
        for i in range(n):
            q_i[i] = 1

        t_i = [None] * n
        for i in range(n):
            t_i[i] = Mvn(data[0], self.precision_fixed)

        t_ij = [[None for _ in range(n)] for _ in range(n)]              

        #ADF is a single iteration 
        z_t_i = [None] * n
        print("Begin factorisation of prior:")

        exp1i = [None] * n
        exp2i = [None] * n


        for i in range(1,n):
            t1 = time.time()
            #generate message i -> j
            z_i = 0
            exp1 = 0
            exp2 = 0
            for j in range(i+1):
                new_ti = 1
                if i == j:
                    p = Mvn(self.innovation.mean, 0.5*self.precision_fixed)
                    z_ii = p.pdf(data[i])

                    t = self.innovation * Mvn(data[i], self.precision_fixed)

                    z_i = z_i + self.alpha * z_ii
                    exp1 = exp1 + self.alpha * z_ii * t.mean
                    c = np.atleast_2d(t.mean)
                    exp2 = exp2 + self.alpha* z_ii * (inv(t.precision) + np.dot(c.T,c))

                else:
                    p = Mvn(data[j], 0.5*self.precision_fixed)
                    z_ij = p.pdf(data[i])

                    t = Mvn(data[j], self.precision_fixed) * Mvn(data[i], self.precision_fixed)
                    z_i = z_i + z_ij
                    exp1 = exp1 + z_ij*t.mean
                    c = np.atleast_2d(t.mean)
                    exp2 = exp2 + z_ij*(inv(t.precision) + np.dot(c.T,c))

            exp1i[i] = exp1
            exp2i[i] = exp2
            exp1 = exp1 / z_i
            exp2 = exp2 / z_i
            z_t_i[i] = z_i
            z_i /= (i + self.alpha)
            
            c = np.atleast_2d(exp1)
            m2 = np.dot(c.T,c)
            mean = exp1
            covar = exp2 - m2
            new_ti = new_ti * Mvn(mean, inv(covar))   

            t2 = time.time()
            print("Factorising i = "+str(i)+" complete. Time taken: "+str(t2-t1))

        t_i[i] = new_ti
        t_ij[0][0] = 1
        print("Beginning message calculation")
        for i in range(1,n):
            t3 = time.time()
            for j in range(i+1):
                mean = 0
                covar = 0
                if i == j:
                    p = Mvn(self.innovation.mean, inv(inv(self.precision_fixed) + inv(self.innovation.precision)))
                    z_ii = p.pdf(data[i])
                    t = self.innovation * Mvn(data[i], self.precision_fixed)
                    new_zi = (z_t_i[i] - self.alpha *z_ii)

                    mean = (exp1i[i] - self.alpha * z_ii * t.mean)/new_zi
                    c1 = np.atleast_2d(t.mean)
                    c2 = np.atleast_2d(mean)

                    new_exp_v = (exp2i[i] - (self.alpha * z_ii * (inv(t.precision) + np.dot(c1.T,c1))))/new_zi
                    covar = new_exp_v - np.dot(c2.T,c2)

                else:
                    p = Mvn(data[j], 0.5*self.precision_fixed)
                    z_ii = p.pdf(data[i])
                    t = Mvn(data[j], self.precision_fixed) * Mvn(data[i], self.precision_fixed)
                    new_zi = (z_t_i[i] - self.alpha *z_ii)

                    mean = (exp1i[i] - self.alpha * z_ii * t.mean)/new_zi
                    c1 = np.atleast_2d(t.mean)
                    c2 = np.atleast_2d(mean)

                    new_exp_v = (exp2i[i] - (self.alpha * z_ii * (inv(t.precision) + np.dot(c1.T,c1))))/new_zi
                    covar = new_exp_v - np.dot(c2.T,c2)
                    
                t_cav_ij = Mvn(mean, inv(covar)*0.999)
                t_ij[i][j] = t_i[i]/t_cav_ij
            t4 = time.time()
            print("Messages for i = " + str(i) + " calculated. Time taken: " + str(t4-t3))
        print("Message calculation completed")
        
        q = 1
        for i in range(n):
            q = q * t_i[i]


        print("Prior approximation initialisation complete")
        print("ADF Initial q is :",q.mean, inv(q.precision))
        return t_i, t_ij