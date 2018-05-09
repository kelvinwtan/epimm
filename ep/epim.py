from mvn import Mvn
import numpy as np
from numpy.linalg import inv


import time

class Epim:
        
    def __init__(self, alpha, mean_base, covar_base):
        #Dirichlet Hyperparameters
        self.alpha = alpha
        self.mvn_base = Mvn(mean_base, covar_base)

        #fixed variance hyperparameter
        self.variance_fixed = np.eye(len(mean_base))
    
        #useful hyperparameters
        self.dim = len(mean_base)
        self.covar_type = "fixed"     
        self.convergence_threshold = 0.01
    
    def set_up(self, data):
        #Check if data and hyperparameters are in the same dimension
        #assert(self.dim == data.shape[1])
        #init each q(\theta_i) = p(x_i|\theta_i)
        q_i = []
        for x in data:
           q_i.append(Mvn(x, self.variance_fixed))
        return q_i
        
    def factorise_prior(self, data):
        #Data (Number of features x Dimension of features)

        n, d = data.shape

        m_ij = np.ndarray(shape=(n,n,d))
        v_ij = np.ndarray(shape=(n,n,d,d))
        s_ij = np.ndarray(shape=(n,n))


        #mean of the 
        qm = data
        qv = np.empty((n,d,d))
        #precision matrix
        #qp = np.empty((n,d,d))
        

        #Precision matrix, initialise all as the same, since there is no update step. use same
        #precision for all points
        qp = inv(self.variance_fixed)
        
        #for i in range(n):
        #   qv[i] = self.variance_fixed 
        #   qp[i] = self.variance_fixed

        #factors of each term 
        t_i = np.ones(n)
        
        t_ik = [[None] * n] * n

        for i in range(n):
            t1 = time.time()
            if i == 0:
                m_ij[0][0] = data[0]
                v_ij[0][0] = self.mvn_base.covar
                s_ij[0][0] = 1
                continue
            for j in range(i+1):

                mu = 0
                sig = 0

                t = self.mvn_base
                tm = 0
                tv = 0

                if i == 0:
                    t = 1
                else:
                    tv = (j + 1)*qp
                    if i == j:
                        for k in range (i):
                            tm = tm + qm[k]
                        tm = np.dot(inv(tv),qm[k])
                    else:
                        tm = qm[i]
                        for k in range (j) + range(j+1,i):
                            tm = tm + qm[k]
            
                    tm = np.dot(inv(tv),tm)    
                    tp = inv(tv)
                """ 
                    if i == j:
                        for k in range (i):
                            tv = tv + qp[i]
                            tm = tm + np.dot(qp[k],qm[k])
                    else:
                        tv = qp[i]
                        tm = np.dot(qm[i],qp[i])
                        for k in range (j) + range(j+1,i):
                            tv = tv + qp[k]
                            tm = tm + np.dot(qp[k],qm[k])
                            print("THIS SHOULD BE AN EYE")
                            print(qp[k])
                            print("THIS SHOULD INC BY 1")
                            print(tv)
                    tm = np.dot(inv(tv),tm)    
                    tp = inv(tv)
                """
                z_i = 0
                #Marginals k < i
                for k in range(i):
                    new_sig = tv + qp
                    new_mu = np.dot(inv(new_sig),(np.dot(tv,tm) + np.dot(qp,qm[k])))
                    temp = Mvn(tm, inv(tv) + qv[k])
                    s = temp.pdf(qm[k])
                    z_i += s
#                   s= 1
                    mu += s*new_mu
                    sig += s*new_sig
                   # print(new_sig)
                #Marginal k == i
                #z_i /= i - 1 + self.alpha
                base_p = inv(self.mvn_base.covar)
                pp = base_p + tv 
                pm = np.dot(inv(pp) , np.dot(base_p,self.mvn_base.mean) + np.dot(tv,tm))
                temp = Mvn(tm, t.covar + self.mvn_base.covar)
                s = temp.pdf(self.mvn_base.mean)
                z_i += self.alpha * s 
                
                #z_i /= (i - 1 + self.alpha)
                #/ (i - 1 + self.alpha)

                mu = (mu + self.alpha*s*pm) / z_i# / (z_i * (i - 1 + self.alpha))
                #Mu is acceptable
                #print("MU")
                #print(mu)

                sig = (sig + self.alpha*s*pp) / z_i# / (z_i * (i- 1 + self.alpha))

                c = np.atleast_2d(mu)
                sig -= np.dot(c.T,c)
                #print(sig)
                z_i /= (i - 1 + self.alpha)
                s_ij[i][j] = z_i
                m_ij[i][j] = mu
#                print("sig")
 #               print(inv(sig))
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
    

    def moment_match(self, i, q_cavity_i, q_i):
        z_i, z_ii, z_ji = self.marginal(i, q_cavity_i)
        #r_j = []
        #First Moment Matching
        r = 0
        mean_i = 0
        covar_i = 0
        for j in range(i+1):
            k = 0
            r_ji = 0
            theta_ji = 0l
            theta_ji_hat = 0
            if i == j:
                #Expectation for each factor
                t1 = self.mvn_base*q_cavity_i[i]
                t2 = Mvn(q_cavity_i[j].mean, self.mvn_base.covar + q_cavity_i[j].covar)
                theta_ji = t1.mean * t2.pdf(self.mvn_base.mean)/z_ii
                theta_ji_hat = t1.covar * t2.pdf(self.mvn_base.mean)/z_ii
                #Innovation
                #dirichlet coefficient
                k = self.alpha / (i - 1 + self.alpha)
                r_ji = k*z_ii/z_i
                #print("i:"+str(i)+" r_ii:"+str(r_ji))
            else:
                #Expectaion for each factor
                t1 = q_cavity_i[i] * q_cavity_i[j]
                t2 = Mvn(q_cavity_i[i].mean, q_cavity_i[i].covar + q_cavity_i[j].covar)
                theta_ji = t1.mean * t2.pdf(q_cavity_i[j].mean)/z_ji[j]
                theta_ji_hat = t1.covar * t2.pdf(self.mvn_base.mean)/z_ji[j]
                #Responsibility
                #Dirichlet coefficient
                k = 1 / (i - 1 + self.alpha)
                r_ji = k*z_ji[j]/z_i

                c1 = np.atleast_2d(theta_ji)
                c1 = np.dot(c1.T,c1)
                c2 = np.atleast_2d(theta_ji - q_cavity_i[j].mean)
                c2 = np.dot(c2.T,c2)
                q_i[j].mean = (1 - r_ji)*q_cavity_i[j].mean + r_ji * theta_ji
                q_i[j].covar = (1 - r_ji)*q_cavity_i[j].covar + r_ji*(theta_ji_hat - c1) + r_ji*(1 - r_ji)*c2

                mean_i += r_ji * theta_ji
                covar_i += r_ji * theta_ji_hat
            r += r_ji

        q_i[i].mean = mean_i     
        c3 = np.atleast_2d(mean_i)
        q_i[i].covar = covar_i - np.dot(c3.T,c3)
        a = (np.linalg.eig(q_i[i].covar))
        #print(a[0])
        return q_i

    def fit(self, data):
        # for each x_{i} in data, assign each point as a part of a distribution with the point as a mean and a fixed var
        #Initialise EP
        #q_i = self.set_up(data)
        n, dim = data.shape

        q_i_m = data
        q_i_p = np.empty((dim,1))
        for i in range(num):
            q_i_p[i] = np.eye(dim)
        
        mm_ij, mp_ij, ms_ij = self.factorise_prior(data)

        f_i = []
        f_m_i = np.ndarray(shape=(n,d))
        f_p_i = np.ndarray(shape=(n,d,d))
        f_s_i = np.ndarray(shape=(n))
        for i in range(num):
            m_i = 0
            p_i = 0
            s_i = 1
            for j in range(i+1):
                p_i += mp_ij[i][j]
                m_i += np.dot(mp_ij[i][j],mm_ij[i][j])
                s_i *= ms_ij[i][j]
            v_i = inv(p_i)
            m_i = np.dot(v_i, m_i)
            
            f_m_i[i] = m_i
            f_p_i[i] = p_i
            f_s_i[i] = s
            
            p = Mvn(m_i,v_i)
            f_i.append(p)
        
        #q\i(\theta_j)
        qm_cavity_ij = np.ndarray(shape=(n,n,d))
        qp_cavity_ij = np.ndarray(shape=(n,n,d,d))
        qs_cavity_ij = np.ndarray(shape=(n,n))
        for i in range(n):
            for j in range(n):
                qp_cavity_ij[i][j] = f_p_i[j] - p_ij[i][j] 
                qm_cavity_ij[i][j] = np.dot(inv(qp_cavity_ij[i][j]), (np.dot(f_p_i[j],f_m_i[j]) - np.dot(mp_ij[i][j],mm_ij[i][j]))) 
                
                temp = (1/ms_ij[i][j]) * np.linalg.det(inv(qp_cavity_ij[i][j])) / np.linalg.det(inv(f_p_i[j]))
                expm = np.atleast_2d(f_m_i[j] - mm_ij[i][j])
                expv = inv(f_p_i[j]) - inv(qp_cavity_ij[i][j])
                qs_cavity_ij[i][j] = temp * np.exp(-0.5*np.matmul(np.matmul(expm.T,expv),expm))

        #Until all f_i converge
        i = 0
        iterations = 0
        converged = False
        print("Set up finished, beginning approximating until convergence")
        while True:

            #DELETION STEP
            q_p_i_approx = q_p_i[i] - f_p_i[i]
            if np.all(np.linalg.eigvals(q_p_i_approx) > 0):
                i+=1
                continue
            q_m_i_approx = np.dot(inv(q_p_i_approx), np.dot(q_m_i[i],q_p_i[i]) - np.dot(f_m_i[i],f_p_i[i]))

            #moment matching


            i += 1
            break




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