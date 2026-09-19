"""NARX-GA

Reference and mechanism: Based on [Hassan et al. (2021)](https://doi.org/10.1016/j.renene.2021.02.103), this method uses a genetic algorithm to fit a nonlinear autoregressive network with exogenous inputs.

Task adaptation: Use local power lags and available weather; recursively feed predictions back over the common horizon.

Pseudocode:
TRAIN(T,V,c):
    net = NARX(lags=c.feedback_lags,hidden=c.hidden_units)
    population = InitializeWeightVectors(net,fixed.population_size)
    for generation in 1..fixed.ga_generations:
        fitness = RecursiveTrainingError(population,T)
        population = SelectCrossoverMutate(population,fitness,fixed.ga_settings)
        PreserveElites(population)
        RecordValidationScores(population,V)
    return NetworkWithBestValidationWeights(net)
PREDICT(Wp,Wf,P):
    history,U = copy(P),Align15(Wf)
    for h in 1..480:
        Yhat[h] = net(history[-c.feedback_lags:],AvailableWeatherLags(Wp,U,h))
        history.append(Yhat[h])
    return Return480(Yhat)
"""

import numpy as np
from common import BaseMethod, align15, subset, task_score

class NARXGA(BaseMethod):
    def unpack(self,population):
        size=self.input_dim*self.config['hidden_units']; hidden=self.config['hidden_units']
        w=population[:,:size].reshape(-1,self.input_dim,hidden)
        b=population[:,size:size+hidden]
        out=population[:,size+hidden:size+2*hidden]
        bias=population[:,-1]
        return w,b,out,bias

    def simulate(self,population,Wp,Wf,P):
        w,b,out,bias=self.unpack(population)
        lag=self.config['feedback_lags']; exog=align15(Wf); old=align15(Wp)
        wl=self.fixed['weather_lags']
        extended=np.concatenate([old[:,-wl:],exog],1) if wl else exog
        population_size=len(population)
        history=np.zeros((population_size,len(P),lag+480),dtype=np.float32)
        history[:,:,:lag]=P[None,:,-lag:]
        for h in range(480):
            weather=np.concatenate([extended[:,h+wl-j] for j in range(wl+1)],-1)
            x=np.concatenate([history[:,:,h:h+lag],np.broadcast_to(weather,(population_size,)+weather.shape)],-1)
            hidden=np.tanh(np.einsum('pni,pih->pnh',x,w,optimize=True)+b[:,None,:])
            history[:,:,lag+h]=np.einsum('pnh,ph->pn',hidden,out,optimize=True)+bias[:,None]
        return history[:,:,lag:]

    def _fit(self,train,valid):
        rng=np.random.default_rng(self.seed)
        count=min(len(train['P']),self.fixed['max_fit_windows'])
        data=subset(train,np.sort(rng.choice(len(train['P']),count,replace=False)))
        hidden=self.config['hidden_units']
        self.input_dim=self.config['feedback_lags']+self.features*(self.fixed['weather_lags']+1)
        size=self.input_dim*hidden+2*hidden+1
        n=self.fixed['population_size']
        if n<2:raise ValueError('GA requires at least two individuals')
        population=rng.normal(0,.05,(n,size)).astype(np.float32)
        elite_count=max(1,int(n*self.fixed['elite_fraction']))
        if elite_count>=n:raise ValueError('GA elite_fraction must leave offspring slots')
        best_score=float('inf'); self.history=[]
        a,b=self.protocol['training_steps']
        for generation in range(self.fixed['ga_generations']):
            prediction=self.simulate(population,data['Wp'],data['Wf'],data['P'])
            fitness=((prediction[:,:,a:b]-data['Y'][None,:,a:b])**2).mean((1,2))
            vp=self.simulate(population,valid['Wp'],valid['Wf'],valid['P'])
            scores=[task_score(vp[k],valid['Y'],self.protocol) for k in range(n)]
            chosen=int(np.argmin(scores))
            if scores[chosen]<best_score:
                self.weights=population[chosen:chosen+1].copy();best_score=scores[chosen]
            self.history.append({'generation':generation+1,'training_best':float(np.min(fitness)),'validation_best':float(min(scores))})
            if generation+1==self.fixed['ga_generations']:break
            order=np.argsort(fitness,kind='stable');offspring=[population[i].copy() for i in order[:elite_count]]
            def parent():
                ids=rng.integers(0,n,2);return population[ids[np.argmin(fitness[ids])]]
            while len(offspring)<n:
                first,second=parent(),parent()
                child=first.copy()
                if rng.random()<self.fixed['crossover_rate']:
                    mask=rng.random(size)<.5;child[mask]=second[mask]
                mutate=rng.random(size)<self.fixed['mutation_rate']
                child[mutate]+=rng.normal(0,self.fixed['mutation_scale'],int(mutate.sum()))
                offspring.append(child)
            population=np.stack(offspring)

    def _predict(self,Wp,Wf,P,time_index):
        return self.simulate(self.weights,Wp,Wf,P)[0]
