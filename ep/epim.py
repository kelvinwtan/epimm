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
        s_ij = np.ones(shape=(n,n))

        qm = data
        qp = self.precision_fixed

        t_i = np.ones(n)


        ntrue = 0
        nfalse = 0



        #calculate message i->j 
        for i in range(n):
            t1 = time.time()
            #ii when i = 0 is just its own prior
            if i == 0:
                m_ij[0][0] = data[0]
                v_ij[0][0] = self.precision_fixed
                s_ij[0][0] = 1
                continue
            for j in range(i+1):
                mu = 0
                exp2 = 0

                #N(tm, tp) is product of all the priors of the parents
                tm = 0
                tp = (j + 1)*qp
                if i == j:
                    for k in range (i):
                        tm = tm + qm[k]
                else:
                    tm = qm[i]
                    for k in range (j) + range(j+1,i):
                        tm = tm + qm[k]
                tm = np.dot(inv(tp),np.dot(qp,tm))

                z_i = 0
                #Marginals k < i
                #For each team, find the marginal of each (delta x parents)

                for k in range(i):
                    new_pres = tp + qp
                    new_mean = np.dot(inv(new_pres),(np.dot(tp,tm) + np.dot(qp,qm[k])))
                    temp = Mvn(tm, inv(inv(tp) + inv(qp)))
                    z_ij = temp.pdf(qm[k])*s_ij[i][j]
                    z_i = z_i + z_ij
                    mu = mu + new_mean * z_ij
                    exp2 = exp2 + inv(new_pres) * z_ij
                
                new_pres = self.innovation.precision + tp
                new_mean = np.dot(inv(new_pres), np.dot(self.innovation.precision, self.innovation.mean) + np.dot(tp,tm))
                
                temp = Mvn(tm, inv(inv(tp) + inv(self.innovation.precision)))
                z_ii = temp.pdf(self.innovation.mean)*s_ij[i][j]
                z_i = (z_i + self.alpha * z_ii)#/ (i - 1 + self.alpha)
  
                mu = (mu + self.alpha*new_mean*z_ii) / z_i
                exp2 = (exp2 + self.alpha*inv(new_pres)*z_ii) / z_i
                z_i = z_i/(i - 1 + self.alpha)
  
                c = np.atleast_2d(mu)
                d = np.dot(c.T,c)
                exp2 = exp2 - d
#                print(exp2,d, exp2+d, self.check_posdef(exp2+d))

                s_ij[i][j] = 1/z_i
                m_ij[i][j] = mu
                v_ij[i][j] = inv(exp2)
                
                
                #print(mu, " vs " , sig)
              #  print("sig")
                if self.check_posdef(exp2) == True:
                    ntrue +=1
                else:
                    nfalse +=1
                #Mvn(mu,inv(sig))
            t2 = time.time()
            print("For i = "+str(i)+" time = "+ str(t2-t1))
        
        print(ntrue,nfalse)

        while True:
            for i in range (4):
                a = 2
        return m_ij, v_ij, s_ij

                
    def marginal(self, i, q_cavity_i):
        #Find Z_i
        if self.covar_type == "fixed":
        #Z_ii = \int_{\theta} p(\theta)q^{\i}(\theta_i = \theta)d\theta
            t = Mvn(q_cavity_i[i].mean, q_cavity_i[i].covar + self.innovation.covar)
            z_ii = t.pdf(self.innovation.mean)
            
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
        
        flag = True
        for j in range(i+1):
            k = 0
            r_ji = 0
            theta_ji = [None for _ in range(i+1)]
            theta_ji_hat = [None for _ in range(i+1)]

            q_cav_ij = q_cav_i / f_ij[i][j]
            if self.check_posdef(inv(q_cav_ij.precision)):
                flag = False
                continue

            if i == j:
                #Expectation for each factor
                t1 = self.innovation*q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(self.innovation.precision) + inv(q_cav_ij.precision))
            
                z_ij = t2.pdf(self.innovation.mean)
                z_i = z_i + z_ij
                mean += t1.mean * self.alpha * z_ij
                covar += inv(t1.precision) * self.alpha * z_ij

            else:
                #Expectaion for each factor
                t1 = q_cav_ij * q_cav_ij
                t2 = Mvn(q_cav_ij.mean, inv(q_cav_ij.precision) + inv(q_cav_ij[j][i].precision))
                
                z_ij = t2.pdf(q_cav_ij.mean)
                z_i = z_i + z_ij

                mean += t1.mean * z_ij
                covar += inv(t1.precision) * z_ij

        den = (z_i*(i - 1 + self.alpha))
        if den == 0:
            den = 1        
        return Mvn(mean/den, covar/den)    


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