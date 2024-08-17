import random
import time
import json
import math
import numpy as np



def calc_prob(n, eps, delt):
    epsp = eps / (2 * np.log(math.e/delt))
    return math.exp(epsp * n)

class MultiSetDP:
    def __init__(self, seed):
        self.itms = []
        self.counts = {}
        self.sets = {}
        self.seed = seed
        self.output = []
    def addToSet(self, setNum, itmNum, itmCount):
        if setNum in self.sets:
            self.sets[setNum].update({itmNum: itmCount})
        else:
            self.sets.update({setNum: {itmNum: itmCount}})
    def addItem(self, itmNum, itmCount):
        if itmNum not in self.itms:
            self.itms.append(itmNum)
        self.counts.update({itmNum: itmCount})
    def cover(self, eps, delt):
        currng = np.random.default_rng(seed = self.seed)
        output = []
        sets = self.sets.copy()
        counts = self.counts.copy()
        for i in range(0, len(sets)):
            nsets = []
            goodness = []
            for key in sets:
                rset = sets[key]
                cgood = 0
                for itm in rset:
                    if itm in counts:
                        cgood = cgood + max(0, min(rset[itm], counts[itm]))
                goodness.append(cgood)
                nsets.append(key)
            probs = list(map(lambda x: calc_prob(x, eps, delt), goodness))
            sumarr = sum(probs)
            probs = list(map(lambda x: x / sumarr, probs))
            temp = currng.choice(nsets, p = probs, shuffle=False)
            temp = temp.item()
            output.append(temp)
            for key in sets[temp]:
                if key in counts:
                    counts[key] = counts[key] - sets[temp][key]
                    if counts[key] <= 0:
                        counts.pop(key, None)
            sets.pop(temp, None)
        self.output = output
        return output
    def optCov(self):
        output = []
        sets = self.sets.copy()
        counts = self.counts.copy()
        for i in range(0, len(sets)):
            curbest = 0
            curgood = 0
            for key in sets:
                cgood = 0
                for itm in sets[key]:
                    if itm in counts:
                        cgood = cgood + max(0, min(sets[key][itm], counts[itm]))
                if cgood >= curgood:
                    curgood = cgood
                    curbest = key
            if curgood <= 0:
                return output, counts
            output.append(curbest)
            temp = curbest
            for key in sets[temp]:
                if key in counts:
                    counts[key] = counts[key] - sets[temp][key]
                    if counts[key] <= 0:
                        counts.pop(key, None)
            sets.pop(temp, None)
            fini = True
            for i in counts:
                if counts[i] > 0:
                    fini = False
            if fini:
                return output, {}
        return output, counts
    def reseed(self, seed):
        self.seed = seed
    def realCover(self):
        realOut = []
        fcounts = self.counts.copy()
        counts = {}
        for itm in fcounts:
            if fcounts[itm] > 0:
                counts[itm] = fcounts[itm]
        output = self.output.copy()
        sets = self.sets.copy()
        for s in output:
            add = False
            for itm in sets[s]:
                if itm in counts:
                    if counts[itm] > 0 and sets[s][itm] > 0:
                        add = True
            if add:
                for itm in sets[s]:
                    if itm in counts:
                        counts[itm] = counts[itm] - sets[s][itm]
                        if counts[itm] <= 0:
                            counts.pop(itm, None)
                realOut.append(itm)
                if not bool(counts):
                    return realOut, {}
        return realOut, counts