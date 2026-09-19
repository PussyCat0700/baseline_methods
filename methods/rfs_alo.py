"""RFs-ALO

Reference and mechanism: Based on [Ibrahim et al. (2020)](https://doi.org/10.1109/TII.2019.2916566), this method combines random forests with ant-lion hyperparameter optimization.

Task adaptation: Replace I–V curve prediction with power forecasting. Restrict ALO to the three prescribed configurations on the tuning pool, then freeze the selected forest settings.

Pseudocode:
SELECT(pool,candidates):
    InitializeALO(fixed.population_size)
    for iteration in 1..fixed.alo_iterations:
        indices = ProjectToLegalCandidateIndices(ALO.positions)
        fitness = CachedMeanSiteValidationScores(indices,pool)
        ALO.update(fitness)
    EvaluateAnyUnvisitedCandidates(pool,candidates)
    return LowestMeanValidationScoreConfig()
TRAIN(T,V,c):
    X,Y = TabularFeatures(T.inputs),T.targets
    RF = StepRegressor(RandomForest(n_estimators=c.trees,max_leaf_nodes=c.leaves))
    return RF.fit(X,Y)
PREDICT(Wp,Wf,P): return Return480(RF(TabularFeatures(Wp,Wf,P)))
"""

import numpy as np
from common import BaseMethod, StepRegressor, tabular_features

class RFsALO(BaseMethod):
    @staticmethod
    def select(evaluate,candidates,fixed,seed):
        rng = np.random.default_rng(seed)
        n,iters = fixed['alo_population'],fixed['alo_iterations']
        positions = rng.uniform(0,len(candidates)-1,size=n)
        cache,trace = {},[]
        def score(index):
            index = int(index)
            if index not in cache:cache[index] = float(evaluate(candidates[index]))
            return cache[index]
        elite = 0
        for iteration in range(iters):
            indices = np.clip(np.rint(positions),0,len(candidates)-1).astype(int)
            losses = np.array([score(i) for i in indices])
            elite = min(cache,key=lambda k:(cache[k],candidates[k]['id']))
            quality = 1/(losses-losses.min()+1e-8); quality /= quality.sum()
            shrink = 1+100*(iteration+1)/iters
            new = []
            for _ in range(n):
                lion = positions[rng.choice(n,p=quality)]
                walks = []
                for center in (lion,float(elite)):
                    walk = np.cumsum(rng.choice([-1.,1.],size=iters+1))
                    unit = 2*(walk[iteration]-walk.min())/max(np.ptp(walk),1)-1
                    walks.append(center+unit*(len(candidates)-1)/shrink)
                new.append(np.clip(np.mean(walks),0,len(candidates)-1))
            positions = np.asarray(new)
            positions[0] = elite
            trace.append({'iteration':iteration+1,'candidate_ids':[candidates[i]['id'] for i in indices],
                          'best_candidate_id':candidates[elite]['id']})
        for i in range(len(candidates)):score(i)
        chosen = min(cache,key=lambda k:(cache[k],candidates[k]['id']))
        return candidates[chosen],trace

    def _fit(self, train, valid):
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(n_estimators=self.config['trees'],max_leaf_nodes=self.config['leaves'],
            max_features=self.fixed['max_features'],bootstrap=self.fixed['bootstrap'],random_state=self.seed,n_jobs=self.fixed['n_jobs'])
        self.forest = StepRegressor(forest).fit(tabular_features(train['Wp'],train['Wf'],train['P']),train['Y'])

    def _predict(self, Wp, Wf, P, time_index):
        return self.forest.predict(tabular_features(Wp,Wf,P))
