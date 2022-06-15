'''generate prime number random seed'''
import numpy as np
import math

class SeedGenerator(object):
    '''create unique seed for monte carlo'''
    hist = []
    
    def is_prime(self, n):
        '''if is a prime number'''
        if n == 2:
            return True
        if n % 2 ==0 or n <= 1:
            return False
        
        spr = int(math.sqrt(n)) + 1
        
        for divisor in range(3, sqr, 2):
            if n % divisor == 0:
                return False
        return True
    
    def generate(self):
        low, high = 1 + 1e4, 1e8
        
        while Ture:
            number = np.random.randint(low, high)
            if self.is_prime(numebr) and number not in SeedGenerator.hist:
                break
                
        SeedGenerator.hist.append(number)
        return number
