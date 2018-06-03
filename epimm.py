from mvn import Mvn
import numpy as np
from numpy.linalg import inv
import time
import pickle
class EPIMM:
        
    def __init__(self, alpha, mean_base, pres_base, precision_fixed):
        #Dirichlet Hyperparameters
        self.alpha = alpha
        self.innovation = Mvn(mean_base, pres_base)

        #fixed variance hyperparameter
        self.precision_fixed = precision_fixed
        self.covariance_fixed = inv(self.precision_fixed)

        #useful hyperparameters
        self.dim = len(mean_base)
        self.covar_type = "fixed"     
        self.convergence_threshold = 0.01
        

    def factorise_prior(self, data):
        n , d = data.shape
        
        q_i = [None] * n
        for i in range(n):
            q_i[i] = Mvn(data[i],self.precision_fixed)

        t_i = [None] * n
        for i in range(n):
            t_i[i] = Mvn(data[0], self.precision_fixed)
        adf = [None] * n
        for i in range(n):
            adf[i] = Mvn(data[0], self.precision_fixed)

        t_ij = [[None for _ in range(n)] for _ in range(n)]              

        #ADF is a single iteration 
        z_t_i = [None] * n
        print("Begin factorisation of prior:")

        exp1i = [None] * n
        exp2i = [None] * n


        #generate message i -> j
        #Basically moment match each dirichlet prior and then subtract the expectation contribution of each j to cut down on computation
        for i in range(1,n):
            t1 = time.time()
            z_i = 0
            exp1 = 0
            exp2 = 0
            for j in range(i+1):
                new_ti = 1
                if i == j:
                    p = Mvn(self.innovation.mean, 0.5*self.precision_fixed)
                    z_ii = p.pdf(data[i])
                    t = self.innovation * Mvn(data[i], self.precision_fixed)
                    k = self.alpha/ (i + self.alpha)
                    z_i = z_i + self.alpha*z_ii
                    exp1 = exp1 + k * z_ii * t.mean

                    c = np.atleast_2d(t.mean)
                    exp2 = exp2 + k * z_ii * (inv(t.precision) + np.dot(c.T,c))

                else:
                    p = Mvn(data[j], 0.5*self.precision_fixed)
                    z_ij = p.pdf(data[i])
                    t = Mvn(data[j], self.precision_fixed) * Mvn(data[i], self.precision_fixed)
                    k = 1 / (i + self.alpha)
                    z_i = z_i + z_ij
                    exp1 = exp1 + k * z_ij * t.mean
                    c = np.atleast_2d(t.mean)
                    exp2 = exp2 + k * z_ij * (inv(t.precision) + np.dot(c.T,c))

            exp1i[i] = exp1
            exp2i[i] = exp2
            z_t_i[i] = z_i
            z_i = z_i/(i + self.alpha)
            exp1 = exp1 / z_i
            exp2 = exp2 / z_i
            
            c = np.atleast_2d(exp1)
            m2 = np.dot(c.T,c)
            mean = exp1
            covar = exp2 - m2
            new_ti = new_ti * Mvn(mean, inv(covar))   

            t2 = time.time()
            print("Factorising i = "+str(i)+" complete. Time taken: "+str(t2-t1))

            t_i[i] = new_ti
            adf[i] = new_ti * (Mvn(data[i], self.precision_fixed))

        t_ij[0][0] = self.innovation
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
                    k = self.alpha/ (i + self.alpha)
                    new_zi = (z_t_i[i] - self.alpha *z_ii) / i
                    exp1 = exp1i[i] - k * z_ii * t.mean
                              

                    c = np.atleast_2d(t.mean)
                    exp2 = exp2i[i] - k * z_ii * (inv(t.precision) + np.dot(c.T,c))
                    
                    exp1 = exp1 / new_zi
                    exp2 = exp2 / new_zi

                    mean = exp1
                    c = np.atleast_2d(mean)
                    covar = exp2 - np.dot(c.T,c)

                else:
                    p = Mvn(data[j], 0.5*self.precision_fixed)
                    z_ij = p.pdf(data[i])
                    t = Mvn(data[j], self.precision_fixed) * Mvn(data[i], self.precision_fixed)
                    k = self.alpha/ (i + self.alpha)
                    new_zi = (z_t_i[i] - self.alpha *z_ii) / (i -1 +self.alpha)
                    exp1 = exp1i[i] - k * z_ii * t.mean
                    c = np.atleast_2d(t.mean)
                    exp2 = exp2i[i] - k * z_ii * (inv(t.precision) + np.dot(c.T,c))
                    
                    exp1 = exp1 / new_zi
                    exp2 = exp2 / new_zi

                    mean = exp1
                    c = np.atleast_2d(mean)
                    covar = exp2 - np.dot(c.T,c)

                #A hack to avoid a a situation where we are dividing 2 gaussians with the same covariance
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
        return t_i, adf,t_ij

    def moment_match(self, i, q_i, f_ij):
        z_i = 0
        r_ij = [None for _ in range(i+1)]
        theta_ij = [None for _ in range(i+1)]
        theta_ij_hat = [None for _ in range(i+1)]

        exp1 = 0
        exp2 = 0
        q_cav = []
        for j in range(i+1):
            m = q_i[i]/f_ij[i][j]
            q_cav.append(Mvn(m.mean,m.precision))
           
        q_cav_ii = q_cav[i]
        for j in range(i+1):
            r_ji = 0
            q_cav_ij = q_cav[j]
            # If cavity has no pos def covariance, skip that iteration of i
            if self.check_posdef(q_cav_ij.precision) == False:
                return None, None
            if i == j:
                #Expectation for each factor
                t1 = self.innovation*q_cav_ii
                t2 = Mvn(q_cav_ii.mean, inv(inv(self.innovation.precision) + inv(q_cav_ii.precision)))
                z_ij = t2.pdf(self.innovation.mean)
                k = self.alpha / (i + self.alpha)
                z_i = z_i + k*z_ij
                r_ij[j] = k * z_ij
                theta_ij[j] = t1.mean
                c = np.atleast_2d(t1.mean)
                theta_ij_hat[j] = inv(t1.precision) + np.dot(c.T,c)

            else:
                #Expectaion for each factor
                t1 = q_cav_ii * q_cav_ij
                t2 = Mvn(q_cav_ii.mean, inv(inv(q_cav_ii.precision) + inv(q_cav_ij.precision)))
                
                z_ij = t2.pdf(q_cav_ij.mean)
                k = 1 / (i + self.alpha)
                z_i = z_i  + k*z_ij
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

        q_i[i].mean = mean
        q_i[i].precision = inv(covar)


        for j in range(i):
            q_cav_ij = q_i[i] / f_ij[i][j]
            c1 = np.atleast_2d(theta_ij[j])
            c2 = np.atleast_2d(theta_ij[j] - q_cav_ij.mean)
            r = r_ij[j]
            inv_r = 1 - r
            q_i[j].mean = inv_r*q_cav_ij.mean + r*theta_ij[j]
            q_i[j].precision = inv( inv_r*inv(q_cav_ij.precision) + r*(theta_ij_hat[j] - np.dot(c1.T,c1)) + r*inv_r*np.dot(c2.T,c2))


        for j in range(i+1):
            f_ij[i][j] = q_i[i]/q_cav[j]

        q = 1
        for factor in q_i:
            q = q*factor
        return q, r_ij[i]


    #Fits a set of feature vectors with an IMM-EP.
    def fit(self, data, t_i = None, t_ij = None):
        #Initialise EP
    
        n, dim = data.shape        
        f_i = None
        f_ij = None
        if t_i == None and t_ij == None:
            f_i,adf,f_ij = self.factorise_prior(data)
        else:
            f_i = t_i
            f_ij = t_ij
            print("Prior calculated factors given, will use instead")

        q_i = [None for _ in range(n)]
        for i in range(n):
            q_i[i] = Mvn(data[i], self.precision_fixed)

        q = 1
        for f in q_i:
            q = q*f

        print("initial q approximation is ", q.mean, inv(q.precision))

        #Until all f_i converge
        converge_count = 0
        converged = False
        iteration = 0
        print("Set up finished, beginning approximating until convergence")
        while True:
            
            print("Iteration number: " + str(iteration))
            total_convergence = 0
            converge_count = 0
            k_num = 0

            for i in range(1,n):
                #moment matching
                oldm = f_i[i].mean
                oldp = f_i[i].precision
                new_q, k = self.moment_match(i, q_i, f_ij)
                if new_q is None:
                    continue
                k_num = k_num + k
                q = new_q

                #update
                f_i[i] = 1
                for j in range(i+1):
                    f_i[i] = f_ij[i][j] * f_i[i]

                #checking for convergence
                dist_mean = np.linalg.norm(oldm - f_i[i].mean)
                dist_pres = np.linalg.norm(oldp - f_i[i].precision)
                
                total_convergence += dist_mean + dist_pres

                if dist_mean < self.convergence_threshold and dist_pres < self.convergence_threshold:
                    converge_count += 1
            print("EP converges when this reaches 0: -> " + str(total_convergence))            
            if converge_count == (n-1) or iteration == 20:
                print("Convergence found. EP is complete and now exiting")
                break      
            elif total_convergence <0.01:
                print("All factors are negative covariance. Exiting")
                break     
            print(total_convergence)

            iteration += 1
        print("Found posterior has form")
        #print("ADF:", adf.mean, inv(adf.precision))
        print("EP:",q.mean, inv(q.precision))
        
        return q_i, adf, k_num

    def check_posdef(self, matrix):
        return np.all(np.linalg.eigvals(matrix)>0)
