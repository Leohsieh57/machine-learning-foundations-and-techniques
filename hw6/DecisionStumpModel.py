import numpy as np

def sign(x):
    return 2*(0 < x)-1

def h(x, s, theta):
    return s*sign(x-theta)

class DecisionStumpModel:
    def __init__(self, FeatList, LabelList):
        self.FeatList = list(FeatList)
        self.LabelList = list(LabelList)
        self.s = -1
        self.theta = -1
        self.optimized = False
    
    def Ein(self, s, theta):
        ErrorList = [label != h(feat, s, theta) for feat, label in zip(self.FeatList, self.LabelList)]
        return np.average(ErrorList)
    
    def GetOptParams(self):
        xSet = sorted(list(self.FeatList))
        ThetaDomain = [-1] + [0.5*(x1+x2) for x1, x2 in zip(xSet, xSet[1:]) if x1 != x2]
        EinParamList = [(self.Ein(s, theta), s, theta) for s in [-1,1] for theta in ThetaDomain]
        Ein, self.s, self.theta, = min(EinParamList)
        self.optimized = True
        return Ein

    def Predict(self, x):
        assert self.optimized
        return h(x, self.s, self.theta)