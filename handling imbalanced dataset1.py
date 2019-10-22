import pandas as pd

from imblearn.over_sampling import RandomOverSampler


data = pd.read_csv("creditcard_csv.csv")
#print(data.head(10))

x = data.iloc[: , :30].values
y = data.iloc[:,30].values

print(x.shape)
print(y.shape)

#print(x)
#print(y)

count = data["Class"].value_counts()

print("Before sampling Fraud transaction is :  " , count[0])
print("Before sampling normal transaction is : " , count[1])


ros = RandomOverSampler()
x_res , y_res = ros.fit_sample(x,y)

print(x_res.shape)
print(y_res.shape)

fraud = 0
normal = 0

for i in y_res:
    if i=="'1'":
        fraud = fraud+1
    else:
        normal = normal+1

print("after sampling Fraud transaction is : " , fraud)
print("after sampling Normal transaction is : "  , normal)


