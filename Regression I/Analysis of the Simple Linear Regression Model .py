
import matplotlib.pyplot as plt
import numpy as np
import csv


y = price_data = []
x1 = bedrooms_data = [] 
x2 = bathrooms_data = [] 
x3 = sqft_living_data = [] 
x4 = sqft_lot_data = [] 
x5 = floors_data = []
x6 = waterfront_data = []
x7 = view_data = []
x8 = condition_data = []
x9 = grade_data = [] 
x10 = sqft_above_data = [] 
x11 = sqft_basement_data = [] 
x12 = yr_built_data = []
x13 = yr_renovated_data = []
x14 = lat_data = []
x15 = long_data = []
x16 = sqft_living15_data = []
x17 = sqft_lot15_data = []

with open("kc_house_data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        y.append(float(row['price']))
        x1.append(float(row['bedrooms']))
        x2.append(float(row['bathrooms']))
        x3.append(float(row['sqft_living']))
        x4.append(float(row['sqft_lot']))
        x5.append(float(row['floors']))
        x6.append(float(row['waterfront']))
        x7.append(float(row['condition']))
        x8.append(float(row['grade']))
        x9.append(float(row['sqft_above']))
        x10.append(float(row['sqft_basement']))
        x11.append(float(row['sqft_basement']))
        x12.append(2015 - float(row['yr_built']))
        x13.append(float(row['yr_renovated']))
        x14.append(float(row['lat']))
        x15.append(float(row['long']))
        x16.append(float(row['sqft_living15']))
        x17.append(float(row['sqft_lot15']))
        

independent_variables = [\
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17]
corrcoeff = 0
index = 0
for i in independent_variables:
    tmp = abs(np.corrcoef(y, i)[0, 1])
    if tmp > corrcoeff:
        corrcoeff = tmp
        index = independent_variables.index(i)+1
corrcoeff_result = (round(corrcoeff,2), index)


plt.scatter(x3, y, color="r", marker=".")
plt.title("scatter plot")
plt.xlabel("sqft_living (x3)")
plt.ylabel("price (y)")
plt.xlim(0,8000)
plt.ylim(0,5000000)
plt.show()


n = len(y)
xbar = np.mean(x3)
ybar = np.mean(y)
sumxiyi = sum(xi*yi for xi, yi in zip(x3, y))
sumxi2 = sum(xi**2 for xi in x3)
sxy = sumxiyi - n*xbar*ybar
sxx = sumxi2 - n*xbar**2
β1hat = sxy / sxx
β0hat = ybar - β1hat*xbar
regression_model_result = ("%.5f x %.5f" % (β1hat,β0hat))


plt.scatter(x3, y, color="r", marker=".")
plt.title("scatter plot & Linear regression model")
plt.xlabel("sqft_living (x3)")
plt.ylabel("price (y)")
plt.xlim(0,8000)
plt.ylim(0,5000000)
X = np.linspace(0, 8000, 100)
Y = β1hat*X + β0hat
plt.plot(X,Y)
plt.show()


sst = sumyi2 = sum(yi**2 for yi in y)
ssr = β1hat*sxx
sse = sst - ssr
mse = sse / n-2
seβ1hat = (mse/sxx) ** 0.5
tα2 = zα2 = 1.96
β1_confidence_interval = (β1hat-seβ1hat*tα2, β1hat+seβ1hat*tα2)
seβ0hat = (mse**0.5) * ((1/n)+(xbar**2/sxx))**0.5
β0_confidence_interval = (β0hat-seβ0hat*tα2, β0hat+seβ0hat*tα2)


T = (β1hat-0)/seβ1hat
t_assumption_result = abs(T) > tα2


from scipy.stats import f
dfr = 1
msr = ssr/dfr
F0 = msr/mse
Fα = f.ppf(q=0.05, dfn=1, dfd=n)
f_assumption_result = F0 > Fα


error = [y[i] - (β1hat*x3[i] + β0hat) for i in range(n)]
error.sort()
P = [(i-0.5)/n for i in range(1,n+1)]
plt.plot(error,P, "g.")
plt.xlabel("e(i)")
plt.ylabel("Pi")
plt.show()


yhat = [(β1hat*x3[i] + β0hat) for i in range(n)]
error = [y[i] - yhat[i] for i in range(n)]
plt.plot(yhat, error, "g.")
plt.plot(yhat, [0 for i in range(n)], color="r")
plt.xlabel("yihat")
plt.ylabel("ei")
plt.show()


from scipy.stats import boxcox
W, λ = boxcox(y)


Wbar = np.mean(W)
Wsumxiwi = sum(xi*wi for xi, wi in zip(x3, W))
Wsxy = Wsumxiwi - n*xbar*Wbar
Wβ1hat = Wsxy / sxx
Wβ0hat = Wbar - Wβ1hat*xbar
Wregression_model_result = ("%.10f x %.10f" % (Wβ1hat, Wβ0hat))
What = Wβ1hat*np.array(x3) + Wβ0hat
plt.scatter(x3, W)
plt.plot(x3, What, "r")
plt.xlabel("x3")
plt.ylabel("wi")
plt.show()


Werror = [W[i] - What[i] for i in range(n)]
plt.plot(What, Werror, "g.")
plt.plot(What, [0 for i in range(n)], color="r")
plt.xlabel("wihat")
plt.ylabel("wei")
plt.show()

        
            

