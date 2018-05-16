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
        self.precision_fixed = inv(np.eye(len(mean_base))*10000)
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
                #combine all parents of i
                #print("1",new_qi)
                #print(k,j+1)
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

        #print(t_i[i].mean, inv(t_i[i].precision))

        t_ij[0][0] = 1
        print("Beginning message calculation")
        for i in range(1,n):
            t3 = time.time()
            for j in range(i+1):
                mean = 0
                covar = 0
                if i == j:
                    p = Mvn(self.innovation.mean, 0.5*self.precision_fixed)
                    z_ii = p.pdf(data[i])
                    t = self.innovation * Mvn(data[i], self.precision_fixed)
                    new_zi = (z_t_i[i] - self.alpha *z_ii)

                    mean = (exp1i[i] - self.alpha * z_ii * t.mean)/new_zi
                    #z * z_t_i[i] * t_i[i].mean * self.alpha - t.mean* self.alpha) / ((i - self.alpha)*new_zi) 
                    c1 = np.atleast_2d(t.mean)
                    c2 = np.atleast_2d(mean)

                    new_exp_v = (exp2i[i] - (self.alpha * z_ii * (inv(t.precision) + np.dot(c1.T,c1))))/new_zi
                    covar = new_exp_v - np.dot(c2.T,c2)
                    #covar = (z * z_t_i[i] * self.alpha * inv(t_i[i].precision) + np.dot(c2.T,c2) - self.alpha*(inv(t.precision) + np.dot(c1.T,c1))) / ((i - self.alpha) * new_zi)

                else:
                    p = Mvn(data[j], 0.5*self.precision_fixed)
                    z_ii = p.pdf(data[i])
                    t = Mvn(data[j], self.precision_fixed) * Mvn(data[i], self.precision_fixed)
                    new_zi = (z_t_i[i] - self.alpha *z_ii)

                    mean = (exp1i[i] - self.alpha * z_ii * t.mean)/new_zi
                    #z * z_t_i[i] * t_i[i].mean * self.alpha - t.mean* self.alpha) / ((i - self.alpha)*new_zi) 
                    c1 = np.atleast_2d(t.mean)
                    c2 = np.atleast_2d(mean)

                    new_exp_v = (exp2i[i] - (self.alpha * z_ii * (inv(t.precision) + np.dot(c1.T,c1))))/new_zi
                    covar = new_exp_v - np.dot(c2.T,c2)
                    
                t_cav_ij = Mvn(mean, inv(covar)*0.999)
                #print(t_i[i].mean, t_i[i].precision,mean, covar)
                t_ij[i][j] = t_i[i]/t_cav_ij
            t4 = time.time()
            print("Messages for i = " + str(i) + " calculated. Time taken: " + str(t4-t3))
        print("Message calculation completed")
        
        q = 1
        for i in range(n):
            q = q * t_i[i]


        print("Prior approximation initialisation complete")
        """
        for i in range(n):
            print(i, q_i[i].mean, inv(q_i[i].precision))
        """

        print("ADF Initial q is :",q.mean, inv(q.precision))
        return t_i, t_ij

    def moment_match(self, i, q_cav_i, q_i, f_ij):
        #r_j = []
        #First Moment Matching

        z_i = 0
        r_ij = [None for _ in range(i+1)]
        theta_ij = [None for _ in range(i+1)]
        theta_ij_hat = [None for _ in range(i+1)]
        
        exp1 = 0
        exp2 = 0

        q_cav_ii = q_i[i] / f_ij[i][i]
        # print(q_cav_i.mean, q_cav_ii.mean)
        for j in range(i+1):
            r_ji = 0

            q_cav_ij = q_i[i] / f_ij[i][j]
            #q\i (theta_j)
            # print(i,j)
            if self.check_posdef(inv(q_cav_ij.precision)) == False:
                return None

            if i == j:
                #Expectation for each factor
                t1 = self.innovation*q_cav_ii
                t2 = Mvn(q_cav_ii.mean, inv(inv(self.innovation.precision) + inv(q_cav_ii.precision)))
            
                z_ij = t2.pdf(self.innovation.mean)
                z_i = z_i + z_ij

                k = self.alpha / (i + self.alpha)

                r_ij[j] = k * z_ij

                theta_ij[j] = t1.mean
                c = np.atleast_2d(t1.mean)
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

            else:
                #Expectaion for each factor
                t1 = q_cav_ii * q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(inv(q_cav_ii.precision) + inv(q_cav_ij.precision)))
                # print("cav",q_cav_ij.mean)
                z_ij = t2.pdf(q_cav_ii.mean)
                # print("zij", z_ij)
                z_i = z_i + z_ij
                
                k = 1 / (i + self.alpha)

                r_ij[j] = k * z_ij
                theta_ij[j] = t1.mean
                c = np.atleast_2d(t1.mean)
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

        exp1 = 0
        exp2 = 0
        for j in range(i+1):
            r_ij[j] = r_ij[j]/z_i
            exp1 = exp1 + theta_ij[j] * r_ij[j]
            exp2 = exp2 + theta_ij_hat[j] * r_ij[j]

        mean = exp1
        c = np.atleast_2d(mean)
        covar = exp2 - np.dot(c.T,c)        

        for j in range(i):
            # if theta_ij[j] is None:
            #     continue
            q_cav_ij = q_i[i] / f_ij[i][j]
            c1 = np.atleast_2d(theta_ij[j])
            c2 = np.atleast_2d(theta_ij[j] - q_cav_ij.mean)

            r = r_ij[j]
            inv_r = 1 - r
            #q_i[j].mean = inv_r*q_cav_ij.mean + r*(theta_ij[j])
            #q_i[j].precision = inv(inv_r*inv(q_cav_ij.precision) + r*(theta_ij_hat[j] - np.dot(c1.T,c1)) + r*inv_r*np.dot(c2.T,c2))
            # print(j, q_i[j].mean, q_i[j].precision)
        q_i[i].mean = mean
        q_i[i].precision = inv(covar)
        #print(inv(q_cav_i.precision), inv(q_i[i].precision))
        # print("i",i, q_i[i].mean, q_i[i].precision)
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

        adf = 1
        for f in f_i:
            q = q*f
            adf = adf*f


        print("initial q approximation is ", q.mean, inv(q.precision))

        #Until all f_i converge
        i = 1
        converge_count = 0
        converged = False
        iteration = 0
        print("Set up finished, beginning approximating until convergence")
        while True:
            
            print("Iteration number: " + str(iteration))
            total_convergence = 0
            converge_count = 0
            for i in range(1,n):

                #deletion
                #print(q.mean, f_i[i].mean)
                q_cav_i = q / f_i[i]

                #moment matching
                oldm = f_i[i].mean
                oldp = f_i[i].precision
                new_q = self.moment_match(i, q_cav_i, f_i, f_ij)
                if new_q is None:
                    continue
                q = new_q
                #update
                a = 1
                b = 1
                for j in range(i+1):
                    a = a * q_i[j]
                    b = b * f_ij[i][j]
                f_i[i] = a/b           

                #checking for convergence
                dist_mean = np.linalg.norm(oldm-f_i[i].mean)
                dist_pres = np.linalg.norm(oldp - f_i[i].precision)
                
                total_convergence = dist_mean + dist_pres


                #print(new_q.mean, inv(new_q.precision), dist_mean)
                #print(dist_mean, f_i[i].mean, old_f.mean)
                if dist_mean < self.convergence_threshold and dist_pres < self.convergence_threshold:
                    converge_count = n
                    break
                    #once one converges, no update will be changed to factor. so all factors converge once one has
            print("EP converges when this reaches 0: -> " + str(total_convergence))            
            if converge_count == n:
                print("Convergence found. EP is complete and now exiting")
                break            
            iteration += 1
        print("Found posterior has form")
        print("ADF:", adf.mean, inv(adf.precision))
        print("EP:",q.mean, inv(q.precision))
    def check_posdef(self, matrix):
        return np.all(np.linalg.eigvals(matrix)>0)
    