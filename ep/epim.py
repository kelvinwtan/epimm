from mvn import Mvn
import numpy as np


class Epim:
        
    def __init__(self, alpha, mean_base, covar_base, covar_fixed):
	
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
	assert(self.dim == data.shape[1])
    
        #init each q(\theta_i) = p(x_i|\theta_i)
	for x in data:
	    q_i.append(Mvn(x, self.variance_fixed))
        return q_i
		
    def factorise_prior(self, q_i):
        n = len(q_i)
        f_ij = [][]
        for i in range(n):
            for j in range (i+1):
                if i == j:
                     #find f_ij = \sigma_\\x_j p(x_i| pa(x_i)) \prod_non_j_parents q_i 
                    #sigma component
                
                    #parents not k...
                    q1 = q_i[0]
                    for k in range (1,i):
                        if k != j:
                            q1 *= q_i[k]
                    
                    q = self.mvn_base * q1
                    q.mean = q.mean * self.alpha
                    for k1 in range(i):
                        for k2 in range(k1):
                            #here q_i[k] is a parent to q_i[i]
                            q2 = Mvn(q_i[k1]-q_i[k2], q_i[k1].covar + q_i[k2].covar)
                            q3 = q1*q2
                            q.mean = q.mean + q3.mean
                            q.covar = q.covar + q3.covar
                    f_ij[i][j] = q
        

                else:
                    #find f_ij = \sigma_\\x_j p(x_i| pa(x_i)) \prod_non_j_parents q_i 
                    #sigma component
                
                    #parents not k...
                    q1 = q_i[i]
                    for k in range (i):
                        if k != j:
                            q1 *= q_i[k]
                  
                    q = self.mvn_base * q1
                    q.mean = q.mean * self.alpha
                    for k1 in range(i):
                        for k2 in range(k1):
                            q2 = Mvn(q_i[k1]-q_i[k2], q_i[k1].covar + q_i[k2].covar)
                            q3 = q1*q2
                            q.mean = q.mean + q3.mean
                            q.covar = q.covar + q3.covar
                    f_ij[i][j] = q
        return f_ij 


    def marginal(self, i, q_cavity_i):
       	#Find Z_i
        if self.covar_type == "fixed":
	    #Z_ii = \int_{\theta} p(\theta)q^{\i}(\theta_i = \theta)d\theta
            t1 = self.mvn_base*q_cavity_i[i]
            t2 = ((2*np.pi)^(self.dim/2))*np.sqrt(t1.covar)
            t3 = Mvn(q_cavity_i[i].mean, q_cavity_i[i].covar + p_0.sig)
            z_ii = t2 * t3.pdf(self.mvn_base)
                
            z_ji = []
            for j in range(i+1):
                #Z_ji = \int_{\theta} q^{\i}(\theta_j = \theta)q^{\i}(\theta_i = \theta)d\theta
                t1 = q_cavity_i[i]*q_cavity_i[j]
                t2 = ((2*np.pi)^(self.dim/2))*np.sqrt(t1.covar)
                t3 = Mvn(q_cavity_i[i].mean, q_cavity_i[i].covar + q_cavity_i[j].covar)
                z_ji.append(t2 * t3.pdf(q_cavity_i[j].mean))
               
            z_ji_total = 0
            for k in z_ji:
                z_ji_total += k
            z_i = self.alpha * z_ii / (i - 1 + self.alpha) + z_ji_total / (i - 1 + self.alpha) 	
        else:
            raise NotImplementedError

        return z_i, z_ii, z_ji
            
 
    def moment_match(self, i, q_cavity_i, q_i):
        z_i, z_ii, z_ji = marginal(i, q_cavity_i)
	#r_j = []

	#First Moment Matching
	r = 0
	mean_i = 0
        covar_i = 0
        for j in range(i+1):
	    k = 0
	    r_ji = 0
	    theta_ji = 0
            theta_ji_hat = 0
	    if i == j:
		#Expectation for each factor
		t1 = self.base*q_cavity_i[i]
                t2 = Mvn(q_cavity_i[j], self.mvn_base.covar + q_cavity_i.covar)
	        theta_ji = t1.mean * t2.pdf(self.mvn_base.mean)
		theta_ji_hat = t1.covar * t2.pdf(self.mvn_base.mean)
		#Innovation
		"""
		r_mean = q_cavity_i[i].mean - self.mvn_base.mean
		r_covar = q_cavity_i[i].covar + self.mvn_base.covar
		r_ji =  Mvn(r_mean, r_covar)
		"""
		#dirichlet coefficient
		k = self.alpha / (i - 1 + self.alpha)
                r_ji = k*z_ii/z_i
            else:
		#Expectaion for each factor
		t1 = q_cavity_i[i] * q_cavity_i[j]
		t2 = Mvn(q_cavity_i[i], q_cavity_i[i].covar + q_cavity_i[j].covar)
		theta_ji = (t1.mean * t2.pdf(q_cavity_i[j].mean))
                theta_ji_hat = t1.covar * t2.pdf(self.mvn_base.mean)
		#Responsibility
		"""
        	r_mean = q_cavity_i[i].mean - q_cavity_i[j].mean
		r_covar = q_cavity_i[i].covar + q_cavity_i[j].covar
		r_ji = Mvn(r_mean, r_covar)	
		"""
		#Dirichlet coefficient
		k = 1 / (i - 1 + self.alpha)
	        r_ji = k * z_ji[j]/z_i

		#Update first moment for other factors
		q_i[j].mean = (1 - r_ji)*q_cavity_i[j].mean + r_ji * theta_ji
                q_i[j].covar = (1 - r_ji)*q_cavity_i[j].covar + r_ji*(theta_ji_hat - theta_ji^2) + r_ji(1 - r_ji)(theta_ji - q_cavity_i[j].mean)^2
            mean_i += r_ji * theta_ji
	    covar_i += r_ji * theta_ji_hat
            #r_j.append(r_ji)
	    r += r_ji
	print("This should be 1: " + r) # should be equal to 1
	q_i[j].mean = mean_i 	
        q_i[j].covar = covar_i - mean_i^2
    return q_i

    def fit(self, data):
	# for each x_{i} in data, assign each point as a part of a distribution with the point as a mean and a fixed var

        #Initialise EP
        q_i = setup(data)
        f_ij = factorise_priors(data)
        f_i = []
        for i in range(len(q_i)):
            f = 1
            for j in range (i + 1):
                f *= f_ij[i][j]
            f_i.append(f)

        #Until all f_i converge
        i = 0
        converged = False
        q_cavity_i = []
        for 
        while True:
            if i > len(q_i):
                i = 0
                converged = True

            old_f_i = f_i[i]
            q_cavity_i[i] = q_i[i] / f_i[i]
            q_i = self.moment_match(i, q_cavity_i, q_i)
            f_i[i] = q_i[i]/q_cavity_i[i]
            
            if(np.linalg.norm(f_i[i].mean - old_f_i.mean) < self.convergence_threshold):
                converged = True

            if converged == True and i == len(q_i):
                break
            i += 1

        print("Finished EP")

