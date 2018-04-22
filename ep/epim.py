from mvn import Mvn
import numpy as np


class Epim:
        covariance = "fixed"
        q_i = []
        p_0 = Mvn(0,1)
        mu_0 = 0
        var_0 = 1
        dimension = 1
        
	def __init__(self, alpha):
		self.alpha = 1
		self.dimension = 0
                fixed_var = True

	def set_up(self, data, dimension):
                self.dimension = dimension
                
                #init gaussian prior. Not sure what best to set this to
		mu_0 = np.zeros(dimension)
                var_0 = np.eye(dimension)
                p_0 = Mvn(mu_0, var_0)
                
                #init each q(\theta_i) = p(x_i|\theta_i)
		for x in data:
			q_i.append(Mvn(x, np.eye(dimension))
		return q_i
		
	def marginal(self, i):
            #Find Z_i
            if covariance == "fixed":
	        #Z_ii = \int_{\theta} p(\theta)q^{\i}(\theta_i = \theta)d\theta
                t1 = p_0*q_cavity_i[i]
                t2 = ((2*np.pi)^(self.dimension/2))*np.sqrt(t1.sig)
                t3 = Mvn(q_cavity_i[i].mu, q_cavity_i[i].sig + p_0.sig)
                z_ii = t2 * t3.pdf(p_0.mu)
                
                z_ji = []
                for j in range(i):
                    for 
                    #Z_ji = \int_{\theta} q^{\i}(\theta_j = \theta)q^{\i}(\theta_i = \theta)d\theta
                    t1 = q_cavity_i[i]*q_cavity_i[j]
                    t2 = ((2*np*pi)^(self.dimension/2))*np.sqrt(t1.sig)
                    t3 = Mvn(q_cavity_i[i].mu, q_cavity_i[i].sig + q_cavity_i[j].sig)
                    z_ji.append(t2 * t3.pdf(q_cavity_i[j].mu))
               
                z_ji_total = 0
                for k in z_ji:
                    z_ji_total += k
                    
		z_i = self.alpha * z_ii / (i - 1 + self.alpha) + z_ji_total / (i - 1 + self.alpha) 	
            return z_i, z_ii, z_ji

        def moment_match(self, i):
           
            # first moment match
            z_i, z_ii, z_ji = marginal(i)

            t1 = p_0*q_cavity_i[i]
            theta_ji[i] = t1.mu/z_ii

            for j in range(i):
                t2 = q_cavity_i[i]*q_cavity_i[j]
                theta_ji[j] = t2.mu/z_ji[j]
                
            r_ji = [][]
            for j in range(i+1):
                if i == j:   
                    r_ji[i][j] = Mvn(q_cavity_i[i].mu-p_0.mu, q_cavity_i[i].sig + p_0.sig)
                else:
                    r_ji[i][j] = Mvn(q_cavity_i[i].mu-q_cavity_i[j].mu, q_cavity_i[i].sig + q_cavity_i[j].sig)
            
            mu_star = 0
            for j in range(i+1):
                k = 0
                if i == j:
                    k = self.alpha / (i-1+self.alpha)
                else
                    k = 1 / (i-1+self.alpha)
                mu_star += k * r_ji[i][j] * theta_ji[j]
                
            q_i[i].mu = mu_star
            for j in range(i):
                q_i[j].mu = (1 - r_ji[i][j])*q_cavity_i[j] + r_ji[i][j]*theta_ji[j]

            #second moment match
            theta_ji_2 = []

            theta_ji_2[i] = t1.sig/z_ii
            for j in range(i):
                t2 = q_cavity_i[i]*q_cavity_i[j]
                theta_ji_2[i] = t2.sig/z_ji[j]

            
            for j in range(i+1):
                q_i[j].sig = (1-r_ji[i][j])*q_cavity_i[j].sig + r_ji[i][j]*(theta_ji_2[j] - theta_ji[j]^2) + r_ji[i][j](1-r_ji[i][j])(theta_ji[j] - q_cavity_i[j].mu)^2)
            
            sig_star = 0
            for j in range(i+1):
                if i == j:
                    k = self.alpga / (i-1+self.alpha)
                else
                    k = 1 / (i-1+self.alpha)
                sig_star += k* r_ji[i][j] * theta_ji_2[j]
            sig_star -= q_i[i].mu^2
            q_i[i].sig = sig_star

	def train(self, data):
		dim = data.shape[1]
		
		# for each x_{i} in data, assign each point as a part of a distribution with the point as a mean and a fixed var

		#modelling each datapoint as a distribution
		for x_i in data:
 			q_i.append(Mvn(x_i,np.identity(dim)))


		#q(\theta) = \prod_i (q(\theta_i))
		#where q(\theta) approximates the distribution p(D|\theta)\prod_i(p(\theta_i|\theta_i=0:n))
		messages = [][]
		s_ij	 = [][]
		#for j<=i
		for i in range(n):
			for j in range(i+1):
			#TODO do prior messages properly
			messages[i][j].append(mvn(dpt_mvn[i]-dpt_mvn[j], np.eye(2, dtype=int))
		
		#Initialise EP
		q_i = self.set_up(data, dim)
		
		#q approximates the entire distribution
		q = 1
		for i in q_i:
			q *= i
		


		i = 0
		#Deletion
		#remove f_hat_i from q to get q\i
		hat_f_i = hat_f_ij[i][0]		
		q_cavity_i = 1
		for j in range(i+1):
			q_cavity_ij[i][j] = q_i[j]/hat_f_ij[i][j]
			
			a = inv(s_ij[i][j])*abs(q_cavity_ij[i][j].sig)/abs(q_i[j].sig)
			e = math.exp(-0.5*transpose(q_i[j].mu-q_cavity_ij[i][j].mu)*inv(q_i[j]-q_cavity[i][j])*(q_i[j].mu-q_cavity_ij[i][j]))
			
			q_cavity_i *= q_cavity_ij[i][j] * a * e
		
