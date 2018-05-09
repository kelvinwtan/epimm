from mvn import Mvn
import numpy as np
from numpy.linalg import inv


import time

class Epim:
        
    def __init__(self, alpha, mean_base, pres_base):
        #Dirichlet Hyperparameters
        self.alpha = alpha
        self.mvn_base = Mvn(mean_base, pres_base)

        #fixed variance hyperparameter
        self.precision_fixed = np.eye(len(mean_base))
    
        #useful hyperparameters
        self.dim = len(mean_base)
        self.covar_type = "fixed"     
        self.convergence_threshold = 0.01
        
    def factorise_prior(self, data):
        #Data (Number of features x Dimension of features)

        n, d = data.shape
        m_ij = np.ndarray(shape=(n,n,d))
        v_ij = np.ndarray(shape=(n,n,d,d))
        s_ij = np.ndarray(shape=(n,n))

        qm = data
        qp = self.precision_fixed

        t_i = np.ones(n) 
        t_ik = [[None] * n] * n
        for i in range(n):
            t1 = time.time()
            if i == 0:
                m_ij[0][0] = data[0]
                v_ij[0][0] = self.precision_fixed
                s_ij[0][0] = 1
                continue
            for j in range(i+1):
                mu = 0
                sig = 0

                #N(tm, tp) is product of all the priors of the parents
                tm = 0
                tp = (j + 1)*qp
              
                if i == j:
                    for k in range (i):
                        tm = tm + qm[k]
                    tm = np.dot(inv(tp),qm[k])
                else:
                    tm = qm[i]
                    for k in range (j) + range(j+1,i):
                        tm = tm + qm[k]
                    tm = np.dot(inv(tp),tm)    
                z_i = 0

                #Marginals k < i
                #For each team, find the marginal of each (delta x parents)
                for k in range(i):
                    new_pres = tp + qp
                    new_mean = np.dot(inv(new_pres),(np.dot(tp,tm) + np.dot(qp,qm[k])))

                    temp = Mvn(tm, tp + qp)
                    s = temp.pdf(qm[k])
                    
                    z_i += s
                    mu += s*new_mean
                    sig += s*inv(new_pres)
                
                pp = self.mvn_base.precision + tp
                pm = np.dot(inv(pp), np.dot(self.mvn_base.precision, self.mvn_base.mean) + np.dot(tp,tm))

                temp = Mvn(tm, inv(tp) + self.mvn_base.precision)
                s = temp.pdf(self.mvn_base.mean)
                z_i += self.alpha * s 
  
                mu = (mu + self.alpha*s*pm) / z_i# / (z_i * (i - 1 + self.alpha))
                sig = (sig + self.alpha*s*inv(pp)) / z_i# / (z_i * (i- 1 + self.alpha))
                c = np.atleast_2d(mu)
                sig -= np.dot(c.T,c)
                #print(sig)
                z_i /= (i - 1 + self.alpha)
                s_ij[i][j] = z_i
                m_ij[i][j] = mu
                v_ij[i][j] = inv(sig)
            t2 = time.time()
            print("For i = "+str(i)+" time = "+ str(t2-t1))
        return m_ij, v_ij, s_ij

                
    def marginal(self, i, q_cavity_i):
        #Find Z_i
        if self.covar_type == "fixed":
        #Z_ii = \int_{\theta} p(\theta)q^{\i}(\theta_i = \theta)d\theta
            t = Mvn(q_cavity_i[i].mean, q_cavity_i[i].covar + self.mvn_base.covar)
            z_ii = t.pdf(self.mvn_base.mean)
            
            z_ji = []
            for j in range(i):
                #Z_ji = \int_{\theta} q^{\i}(\theta_j = \theta)q^{\i}(\theta_i = \theta)d\theta
                t = Mvn(q_cavity_i[i].mean, q_cavity_i[i].covar + q_cavity_i[j].covar)
                z_ji.append(t.pdf(q_cavity_i[j].mean))
               
            z_ji_total = 0
            for k in z_ji:
                z_ji_total += k
            z_i = self.alpha * z_ii / (i - 1 + self.alpha) + z_ji_total / (i - 1 + self.alpha)     
        else:
            raise NotImplementedError

        return z_i, z_ii, z_ji

    def check_posdef(self, covar):
        return np.all(np.linalg.eigvals(covar)>0)
    

    def moment_match(self, i, q_cav_i, f_ij, q_i):
      #  z_i, z_ii, z_ji = self.marginal(i, q_cavity_i)
        #r_j = []
        #First Moment Matching
        r = 0
        mean = 0
        precision = 0
        covar = 0
        z_i = 0
        z_ij = [[None for _ in range(i+1)] for _ in range(i+1)]
        theta_ji = [None for _ in range(i+1)]
        theta_ji_hat = [None for _ in range(i+1)]
        for j in range(i+1):
            k = 0
            r_ji = 0
            theta_ji = [None for _ in range(i+1)]
            theta_ji_hat = [None for _ in range(i+1)]

            q_cav_ij = q_cav_i / f_ij[i][j]
            if self.check_posdef(inv(q_cav_ij.precision)):
                continue

            if i == j:
                #Expectation for each factor
                t1 = self.mvn_base*q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(self.mvn_base.precision) + inv(q_cav_ij.precision))
            
                z_ij[i][j] = t2.pdf(self.mvn_base.mean)
                z_i = z_i + z_ij[i][i]
                theta_ji[i] = t1.mean
                theta_ji_hat[i] = inv(t1.precision)

            else:
                #Expectaion for each factor
                t1 = q_cav_ij * q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(q_cav_ij.precision) + inv(q_cav_ij[j][i].precision))
                
                z_ij[i][j] = t2.pdf(q_cav_ij.mean)
                z_i = z_i + z_ij[i][j]

                theta_ji[j] = t1.mean
                theta_ji_hat[j] = inv(t1.precision)

        for j in range(i+1):
            q_cav_ij = q_cav_i / f_ij[i][j]
            if self.check_posdef(inv(q_cav_ij.precision)):
                continue
            mean_new = z_ij[i][j] * theta_ji[j]
            covar_new = z_ij[i][j] * theta_ji_hat[j]
            if i == j:
                mean_new *= self.alpha
                covar_new *= self.alpha
            mean += mean_new
            covar += covar_new

        if mean == 0:
            return Mvn(q_i[i].mean, q_i[i].precision)

        den = (z_i*(i - 1 + self.alpha))
        q_i[i].mean = np.atleast_2d(mean / den)
        covar = covar / den
        q_i[i].precision = inv(covar - np.dot(mean.T, mean))

        for j in range(i+1):
            r_ij = z_ij[i][j]/den
            inv_r_ij = 1 - r_ij

            m1 = np.atleast_2d(theta_ji[i][j])
            m2 = np.atleast_2d(m1 - q_cav_ij.mean)
            
            q_i[j].mean = inv_r_ij*q_cav_ij.mean + r_ij*theta_ji[j]
            q_i[j].precision = inv(inv_r_ij*inv(q_cav_ij.precision) + r_ij*(theta_ji_hat - np.dot(m1.T,m1)) + r_ij*inv_r_ij*np.dot(m2.T,m2))

        return Mvn(q_i[i].mean, q_i[i].precision)    


    def fit(self, data):
        # for each x_{i} in data, assign each point as a part of a distribution with the point as a mean and a fixed var
        #Initialise EP
        #q_i = self.set_up(data)
        n, dim = data.shape        
        mm_ij, mp_ij, ms_ij = self.factorise_prior(data)


        q_i = [None for _ in range(n)]
        f_ij = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            q_i[i] = Mvn(data[0], self.precision_fixed)
            for j in range(i+1):
                print(mp_ij[i][j])
                f_ij[i][j]= Mvn(mm_ij[i][j], mp_ij[i][j], ms_ij[i][j])

        #q is the estimation of the total posterior distribution
        q = q_i[0]

        for i in range(1,n):
            q = q * q_i[i]

        f_i = [None for _ in range(n)]
        for i in range(n):
            m_i = 0
            p_i = 0
            s_i = 1
            for j in range(i+1):
                p_i += mp_ij[i][j]
                m_i += np.dot(mp_ij[i][j],mm_ij[i][j])
                s_i *= ms_ij[i][j]
            m_i = np.dot(inv(p_i), m_i)
            f_i[i] = Mvn(m_i,p_i,s_i)
        
        #Until all f_i converge
        i = 0
        iterations = 0
        converged = False
        print("Set up finished, beginning approximating until convergence")
        while True:

            #DELETION STEP
            q_cav_i = q/f_i[i] 
            """
            if np.all(np.linalg.eigvals(inv(q_cav_i.precision)) > 0):
                i+=1
                print("ALERT: non semidef pos covar detected")
                continue
            """
            #moment matching
            old_q = q
            q = self.moment_match(i, q_cav_i, f_ij, q_i)
                    

            f_i[i] = q / q_i[i]
            i += 1

            dist = np.linalg.norm(old_q,q)
            print(dist)
            if(dist < 0.01):
                break
            
        print("mean = ")
        print(q.mean)


        """
            break
            if i >= len(q_i):
                #print("interation")
                i = 0
                converged = True
            old_f_i = f_i[i]
         
            #DELETION STEP
            try:
                t  = q_i[i]/f_i[i]
            except ValueError:
                i+=1
                continue


            q_cavity_i[i] = t
            q_i = self.moment_match(i, q_cavity_i, q_i)

            try:
                f_i[i] = q_i[i]/q_cavity_i[i]
                print(f_i[i].covar)
            except ValueError:
                i+=1
                continue
            
            dist = np.linalg.norm(f_i[i].mean - old_f_i.mean)
            print(dist)
            if(dist < self.convergence_threshold):
                converged = True

            if converged == True and i == len(q_i):
                break
            i += 1
            iterations += 1
        print("Finished EP")
        """