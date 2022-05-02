# Delaytron Codes
For running experiments for particular dataset say (cifar-10)
```
# python3 Delaytron.py.py --data 'fashion' --dim 100 --num_class 10
```
This will create a folder named cifar10 (contains datafiles for plotting error curves)
To plot results (uncommnet the name of dataset for which error rates are to be plotted)
````
python3 plot.py
````
This wil generate error curves as shown in the figure 1 of the paper. 
