


import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization, BicScore,ExhaustiveSearch


##########################################
# Step 1: We create a Bayesian Network
##########################################

# Structure: B -> A, C -> A, A -> D, A -> E
model = BayesianNetwork([("B", "A"), ("C", "A"), ("A", "D"), ("A", "E")])



# Distribution of B
d_B = TabularCPD(variable="B", variable_card=2,
                   values=[[0.6],  # P(B=0)
                           [0.4]]) # P(B=1)

# Distribution of C
d_C = TabularCPD(variable="C", variable_card=2,
                   values=[[0.8],  # P(C=0)
                           [0.2]]) # P(C=1)

# Conditional distribution of A given B and C.

d_A = TabularCPD(variable="A", variable_card=2,
                   values=[[0.9, 0.7, 0.6, 0.2],  # P(A=0 | B, C)
                           [0.1, 0.3, 0.4, 0.8]], # P(A=1 | B, C)
                   evidence=["B", "C"],
                   evidence_card=[2, 2])

# Conditional distribution of D given A

d_D = TabularCPD(variable="D", variable_card=2,
                   values=[[0.8, 0.3],
                           [0.2, 0.7]],
                   evidence=["A"],
                   evidence_card=[2])

# Conditional distribution of E given A

d_E = TabularCPD(variable="E", variable_card=2,
                   values=[[0.9, 0.4],
                           [0.1, 0.6]],
                   evidence=["A"],
                   evidence_card=[2])

#We check if the graph together with the distribution form a Bayesian Network
model.add_cpds(d_B, d_C, d_A, d_D, d_E)
assert model.check_model(), "Not a Bayesian Netword"

print("Step 1:We create a Bayesian network whose nodes have the following conditional distributions:")
print("distr of  B:")
print(d_B)
print("\ndistr of C:")
print(d_C)
print("\ncond distr of A:")
print(d_A)
print("\ncond distr of D:")
print(d_D)
print("\ncond distr of E:")
print(d_E)

##########################################
# Pas2: We generate N data points
##########################################
N=100000
sampler = BayesianModelSampling(model)
data = sampler.forward_sample(size=N)
print("\nStep 2: We generate N data points")
print(data.head())

##########################################
# Pas 3: "We "forget" the conditional distribution and we use the data and the graph structure to estimate them
##########################################

# we create a new model with the same structure but without conditional dstributions
estimated_model = BayesianNetwork(model.edges())

# we estimate the cond distr. using MLE
estimated_model.fit(data, estimator=MaximumLikelihoodEstimator)

print("\nStep 3: Estimated conditional distributions")
for cpd in estimated_model.get_cpds():
    print("Cond. distr of {}: \n{}".format(cpd.variable, cpd))

##########################################
# Pas4: Using censored data, we use Expectation Maximization to estimate the parameters of the model
##########################################


# to obtain censored data, we randomly set 20% of the entries of column E NaN
partial_data = data.copy()
partial_mask = np.random.rand(len(data)) < 0.2
partial_data.loc[partial_mask, "E"] = np.nan

print("\nStep 4: first 10 lines of the new dataset")
print(partial_data.head(10))


# We use EM to estimate the parameters of the model
# we create a dictionary with the possible values of the variables A,B,C,D,E
state_names = {
    "A": [0, 1],
    "B": [0, 1],
    "C": [0,1],
    "D": [0,1],
    "E": [0,1]
}

for col, cats in state_names.items():
    data[col] = pd.Categorical(data[col], categories=cats)

model_em = BayesianNetwork(model.edges())
model_em.fit(data, estimator=ExpectationMaximization, state_names=state_names, n_jobs=1)

print("\nStep 4: Estimated cond. distr. using EM")
for cpd in model_em.get_cpds():
    print("Cond. distr. of {}: \n{}".format(cpd.variable, cpd))



##########################################
# Step 5: we learn the graph structure using data
##########################################
#we use a score based method
#scoring function
bic = BicScore(data)
es = ExhaustiveSearch(data,scoring_method=bic)
learned_graph = es.estimate()
print("\nStep 5: we learn the graph structure using data")

print("\nEdges of the learned graph", learned_graph.edges())

learned_graph = BayesianNetwork(learned_graph.edges())

learned_graph.fit(data, estimator=MaximumLikelihoodEstimator)

for cpd in learned_graph.get_cpds():
    print(f"Cond. distr. of {cpd.variable}:")
    print(cpd)
    print()
