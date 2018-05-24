class PerceptronError(Exception):
    """base error class for this  module"""
    pass

class VectorError(PerceptronError):
    def __init__(self,message = None):
        if message is None:
            message = "vector must be dimensional 1 or greater"
        super(VectorError,self).__init__(message)
        
        
