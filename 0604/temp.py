import pickle

a = [2,3,4]
n = 2

with open('a'+str(n)+'.pickle','wb') as f:
  pickle.dump(a,f)

loss = pickle.load(open("losses_ep0.pickle", "rb"))
print(loss)
