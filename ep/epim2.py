from mvn import Mvn
import numpy as np
from numpy.linalg import inv


import time

class Epim:
        
    def __init__(self, alpha, mean_base, pres_base):
        #Dirichlet Hyperparameters
        self.alpha = alpha
        self.innovation = Mvn(mean_base, pres_base)

        #fixed variance hyperparameter
        self.precision_fixed = np.eye(len(mean_base))*0.00025
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

        


        q_ij = [[None for _ in range(n)] for _ in range(n)]

        q_i[0] = self.innovation
        q_ij[0][0] = self.innovation                    
        



        #ADF is a single iteration 
        print("Begin factorisation of prior:")
        for i in range(1,n):
            t1 = time.time()
            #for all j <= i
            #generate message i -> j
            new_qi = 1

            for j in range(i+1):
                #combine all parents of i
                #print("1",new_qi)
                q_parent = 1
                if i == j:
                    for k in range (i):
                        #all parent points to j
                        q_parent = q_parent*q_i[k]
                else:
                    for k in range (j) + range(j+1,i+1):
                        #ALL POINTS <i + i NOT INCLUDING J
                        #if q_i[k] != 1 and q_parent != 1:
                        #   print(q_parent.mean,inv(q_parent.precision)," X ", q_i[k].mean, inv(q_i[k].precision))
                        q_parent = q_parent*q_i[k]
                #         if q_i[k] != 1:
                #             print("q",q_i[k].mean, q_i[k].precision)
                # if q_parent != 1:
                #     print("parent",i,j, q_parent.mean, q_parent.precision)
                        #print(q_parent.mean,inv(q_parent.precision))
                #moment match the parent x true distribution
                #q_parent*prior
                #prior is the dirichlet process
                z_i = 0
                exp1 = 0
                exp2 = 0
                for k in range (j+1):
                    #print(k,j+1)
                    if i == k:
                        p = q_parent * self.innovation
                        #print(i,k,p.mean,inv(p.precision))

                        t = Mvn(q_parent.mean, inv(inv(q_parent.precision)+inv(self.precision_fixed)))
                        z_ii = t.pdf(self.innovation.mean) * self.alpha
                        z_i = z_i + z_ii
                        #print("z_ii",z_ii, t.pdf)
                        exp1 = exp1 + z_ii * p.mean

                        c = np.atleast_2d(p.mean)
                        exp2 = exp2 + z_ii * (inv(p.precision) + np.dot(c.T,c))
                    else:
                        p = Mvn(data[i], 2*self.precision_fixed) * q_parent
                        print(p.mean, p.precision)
                        z_ij = p.pdf(data[k])
                        if z_ij == 0:
                            z_ij = 1e-300
                        z_i = z_i + z_ij
                        #print("z_ji", z_ij, p.mean, p.precision, data[k])
                        sig = inv(2*self.precision_fixed)
                        mu = np.dot(sig, np.dot(self.precision_fixed,data[i])+np.dot(self.precision_fixed,data[k]))

                        c = np.atleast_2d(mu)
                        sig = sig + np.dot(c.T,c)

                        exp1 = exp1 + z_ij * mu
                        exp2 = exp2 + z_ij * sig
                #Its possible for the z_i to be so small < 1e-300 that it is effectively 0
                #if z_i == 0:
                 #   new_qi = q_ij[i][j-1]
                  #  continue

                exp1 = exp1 / z_i
                exp2 = exp2 / z_i
                #print(z_i,i - 1 + self.alpha)
                z_i /= (i - 1 + self.alpha)
                c = np.atleast_2d(exp1)
                m2 = np.dot(c.T,c)
                mean = exp1
                covar = exp2 - m2
                #print(mean,covar)
                print(mean,covar)
                q_ij[i][j] = Mvn(mean, inv(covar))
                #print(z_i, new_qi)
                new_qi = new_qi * q_ij[i][j]
                #print(new_qi, q_ij[i][j])
                #print(new_qi,new_qi.mean,new_qi.precision)
            q_i[i] = new_qi 
            t2 = time.time()
            print("Factorising i = "+str(i)+" complete. Time took: "+str(t2-t1))

        return q_i, q_ij

    def moment_match(self, i, q_i, f_ij):
        #r_j = []
        #First Moment Matching

        z_i = 0
        s_ij = [[None for _ in range(i+1)] for _ in range(i+1)]
        theta_ij = [None for _ in range(i+1)]
        theta_ij_hat = [None for _ in range(i+1)]
        
        exp1 = 0
        exp2 = 0
  
        #print(q_i[i],f_ij[i][i])
        #print(q_i[i].mean, q_i[i].precision, f_ij[i][i].mean, f_ij[i][i].precision)
        if q_i[i] == f_ij[i][i]:
            q = 1
            for factor in q_i:
                q = q*factor
            return q

        q_cav_ii = q_i[i] / f_ij[i][i]

        for j in range(i+1):
            r_ji = 0

            q_cav_ij = q_i[i] / f_ij[i][j]
            #q\i (theta_j)
            # print(i,j)
            if self.check_posdef(inv(q_cav_ij.precision)):
                flag = False
                continue

            if i == j:
                #Expectation for each factor
                t1 = self.innovation*q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(inv(self.innovation.precision) + inv(q_cav_ij.precision)))
            
                z_ij = t2.pdf(self.innovation.mean)
                z_i = z_i + z_ij
                exp1 += t1.mean * self.alpha * z_ij
                c = np.atleast_2d(t1.mean)
                exp2 += (inv(t1.precision) + np.dot(c.T,c)) * self.alpha * z_ij

                s_ij[j] = z_ij
                theta_ij[j] = t1.mean
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

            else:
                #Expectaion for each factor
                t1 = q_cav_ii * q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(inv(q_cav_ii.precision) + inv(q_cav_ij.precision)))
                #print(inv(t2.precision))
                
                z_ij = t2.pdf(q_cav_ii.mean)
                z_i = z_i + z_ij
                exp1 += t1.mean * z_ij
                c = np.atleast_2d(t1.mean)
                exp2 += (inv(t1.precision) + np.dot(c.T,c)) * z_ij

                s_ij[j] = z_ij
                theta_ij[j] = t1.mean
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

        den = (z_i*(i - 1 + self.alpha))
        if den == 0:
            q = 1
            for factor in q_i:
                q = q*factor
            return q

        mean = exp1 / den
        covar = exp2 /den
        c = np.atleast_2d(mean)
        covar = covar - np.dot(c.T,c)        

        for j in range(i):
            q_cav_ij = q_i[i] / f_ij[i][j]

            c1 = np.atleast_2d(theta_ij_hat[j])
            c2 = np.atleast_2d(theta_ij[j] - q_cav_ij.mean)

            r_ij = s_ij[j]/den
            inv_r_ij = 1 - r_ij
            q_i[j].mean = inv_r_ij*q_cav_ij.mean + r_ij*(theta_ij[j])
            q_i[j].precision = inv(inv_r_ij*inv(q_cav_ij.precision) + r_ij*(theta_ij[j] - np.dot(c1.T,c1)) + r_ij*inv_r_ij*np.dot(c2.T,c2))

        q_i[i].mean = mean
        q_i[i].precision = inv(covar)


        q = 1
        for factor in q_i:
            q = q*factor
        return q


    def fit(self, data):
        # for each x_{i} in data, assign each point as a part of a distribution with the point as a mean and a fixed var
        #Initialise EP
        #q_i = self.set_up(data)
        n, dim = data.shape        
        q_i, f_ij = self.factorise_prior(data)
        
        q = 1
        for factor in q_i:
            q = q*factor

        print("initial q approximation is ", q.mean, inv(q.precision))
                
        #Until all f_i converge
        i = 1
        hard_cap = 10000
        hard_cap_count = 0
        converge_count = 0
        converged = False
        iteration = 0
        print("Set up finished, beginning approximating until convergence")
        while True:
            if i == n:
                converge_count = 0
                i = 0
                iteration += 1
            if iteration % 10 == 0 and iteration > 0 :
                print("Iteration = " + str(iteration))
    
            #moment matching
            old_mean = q_i[i].mean
            q = self.moment_match(i, q_i, f_ij)
            f = 1
            for j in range(i+1):
                f = f*f_ij[i][j]
            q_i[i] = q/f
            
            dist = np.linalg.norm(old_mean-q_i[i].mean)

            if(dist < 0.001):
                converge_count +=1

            if converge_count == (n-1):
                print("Convergence found. EP is complete and now exiting")
                break
            i += 1            
        print("Found posterior has form")
        print("mu ",q.mean,"variance ", inv(q.precision))


    def check_posdef(self, matrix):
        return np.all(np.linalg.eigvals(matrix)>0)
    