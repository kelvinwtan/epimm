from mvn import Mvn
import numpy as np
from numpy.linalg import inv


import time

class Epim:
        
    def __init__(self, alpha, mean_base, pres_base):
        np.set_printoptions(precision=17)
        #Dirichlet Hyperparameters
        self.alpha = alpha
        self.innovation = Mvn(mean_base, pres_base)

        #fixed variance hyperparameter
        self.precision_fixed = np.eye(len(mean_base))*0.04
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
        q_i[0] = self.innovation

        t_i = [None] * n
        for i in range(n):
            t_i[i] = Mvn(data[0], self.precision_fixed)

        t_ij = [[None for _ in range(n)] for _ in range(n)]
        t_ij[0][0] = Mvn(self.innovation.mean_base - data[0], 2*self.innovation.precision)                    
        
        #ADF is a single iteration 
        print("Begin factorisation of prior:")
        for i in range(1,n):
            t1 = time.time()
            #generate message i -> j
            new_ti = 1
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
                        q_parent = q_parent*q_i[k]
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
                        exp1 = exp1 + z_ii * p.mean

                        c = np.atleast_2d(p.mean)
                        exp2 = exp2 + z_ii * (inv(p.precision) + np.dot(c.T,c))
                    else:
                        p = Mvn(data[i], 2*self.precision_fixed) * q_parent
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

                exp1 = exp1 / z_i
                exp2 = exp2 / z_i
                z_i /= (i - 1 + self.alpha)
                c = np.atleast_2d(exp1)
                m2 = np.dot(c.T,c)
                mean = exp1
                covar = exp2 - m2
                t_ij[i][j] = Mvn(mean, inv(covar))
                new_ti = new_ti * t_ij[i][j]
            
            t_i[i] = new_ti 
            print(t_i[i].mean, t_i[i].precision)
            t2 = time.time()
            print("Factorising i = "+str(i)+" complete. Time taken: "+str(t2-t1))
        print("Completed initialisation of all t_ij")
        print("Setting q_i")

        for i in range(n):
            q_i[i] = 1
            for j in range(i, n):
                q_i[i] = q_i[i] * t_ij[j][i]

        print("Prior initialisation complete")
        for i in range(n):
            print(i, q_i[i].mean, inv(q_i[i].precision))
        return q_i, t_ij

    def moment_match(self, i, q_cav_i, q_i, f_ij):
        #r_j = []
        #First Moment Matching

        z_i = 0
        r_ij = [None for _ in range(i+1)]
        theta_ij = [None for _ in range(i+1)]
        theta_ij_hat = [None for _ in range(i+1)]
        
        exp1 = 0
        exp2 = 0
  
        #print(q_i[i],f_ij[i][i])
        #print(q_i[i].mean, q_i[i].precision, f_ij[i][i].mean, f_ij[i][i].precision)
        #NOT SURE IF NEED THIS ANYMORE
        if q_i[i] == f_ij[i][i]:
            q = 1
            for factor in q_i:
                q = q*factor
            return q

        q_cav_ii = q_cav_i / f_ij[i][i]

        for j in range(i+1):
            r_ji = 0

            q_cav_ij = q_cav_i / f_ij[i][j]
            #q\i (theta_j)
            # print(i,j)
            if self.check_posdef(inv(q_cav_ij.precision)) == False:
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

                r_ij[j] = self.alpha * z_ij
                theta_ij[j] = t1.mean
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

            else:
                #Expectaion for each factor
                t1 = q_cav_ii * q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(inv(q_cav_ii.precision) + inv(q_cav_ij.precision)))
                print((q_cav_ij.weight * q_cav_ii.weight))
                z_ij = t2.pdf(q_cav_ii.mean)
                z_i = z_i + z_ij
                exp1 += t1.mean * z_ij
                c = np.atleast_2d(t1.mean)
                exp2 += (inv(t1.precision) + np.dot(c.T,c)) * z_ij

                r_ij[j] = z_ij
                theta_ij[j] = t1.mean
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

        den = (z_i*(i - 1 + self.alpha))

        if den == 0:
            q = 1
            for factor in q_i:
                q = q*factor
            return q

        for j in range(i+1):
            r_ij[j] = r_ij[j]/den
 

        mean = exp1 / den
        covar = exp2 /den
        c = np.atleast_2d(mean)
        covar = covar - np.dot(c.T,c)        

        for j in range(i):
            if theta_ij[j] is None:
                continue
            q_cav_ij = q_i[i] / f_ij[i][j]
            c1 = np.atleast_2d(theta_ij_hat[j])
            c2 = np.atleast_2d(theta_ij[j] - q_cav_ij.mean)

            r = r_ij[j]
            inv_r = 1 - r
            q_i[j].mean = inv_r*q_cav_ij.mean + r*(theta_ij[j])
            q_i[j].precision = inv(inv_r*inv(q_cav_ij.precision) + r*(theta_ij[j] - np.dot(c1.T,c1)) + r*inv_r*np.dot(c2.T,c2))

        q_i[i].mean = mean
        print(covar)
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
        f_i, f_ij = self.factorise_prior(data)
        
        q_i = [None for _ in range(n)]
        for i in range(n):
            q_i[i] = Mvn(data[i], self.precision_fixed)

        q = 1
        for f in q_i:
            q = q*f

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
            if iteration % 50 == 0:
                print("Iteration number: " + str(iteration))
            converge_count = 0
            for i in range(1,n):
                
                #deletion
                q_cav_i = q / f_i[i]

                #moment matching
                old_f = f_i[i]
                q = self.moment_match(i, q_cav_i, f_i, f_ij)
            
                #update
                a = 1
                b = 1
                for j in range(i+1):
                    a = a * q_i[j]
                    b = b * f_ij[i][j]
                f_i[i] = a/b
            
                #checking for convergence
                dist_mean = np.linalg.norm(old_f.mean-f_i[i].mean)
                dist_pres = np.linalg.norm(old_f.precision - f_i[i].precision)

                if dist_mean < self.convergence_threshold and dist_pres < self.convergence_threshold:
                    converge_count +=1    
            
            if converge_count == (n-1):
                print("Convergence found. EP is complete and now exiting")
                break            
            iteration += 1
        print("Found posterior has form")
        print("mu ",q.mean,"variance ", inv(q.precision))


    def check_posdef(self, matrix):
        return np.all(np.linalg.eigvals(matrix)>0)
    