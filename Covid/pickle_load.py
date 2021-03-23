import pickle

s = pickle.load(open("mets_state_wise", "rb")) 

rmse = []
mape = []

for state in a: 
    rmse.append(a[state][0]) 
    mape.append(a[state][1])


