from gmm import GMM

class BayesFactor:
    def __init__(self, threshold = 1)
        self.threshold = threshold

    def assess(self, x, gmm1, gmm2):
        """
        Outputs the Bayes Factor K given by
        K = P(D|M_1) / P(D|M_2)
        
        Where the posterior distributions are given as gaussian mixtures

        Parameters
        -----------------------------------------------------------------------
        x: value taken of pdf
        gmm1: gaussian mixture of model 1
        gmm2: gaussian mixture of model 2

        Returns
        -------
        bool: Accepted or Unaccepted
        float: score
        """
        print("Scoring data has {} dimensions", x.shape[0])
        num = gmm1.log_likelihood(x)
        den = gmm2.log_likelihood(x)

        score = num/den

        if score > threshold:
            return True, score
        else
            return False, score


        
