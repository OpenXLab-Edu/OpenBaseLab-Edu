
from BaseML import cls

model=cls('CART')
model.load_dataset('lenses.csv', type ='csv', x_column = [0,1,2,3,4],y_column=[5])
model.train()
model.save('mymodel.pkl')

y=model.inference([[1,  1,  1,  1,  1]])
m=cls('CART')
m.load('mymodel.pkl')
y=m.inference([[1,  1,  1,  1,  1]])
print(y)