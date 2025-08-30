Place your training data CSV file here (default path used by the app is data/train.csv).

Format:
- CSV (comma separated), numeric values
- Optional header row (will be skipped automatically)
- Last column is the binary target (0/1)
- All preceding columns are features

Example (6 features + target):

f1,f2,f3,f4,f5,f6,target
0.1,0.2,0.3,0.4,0.5,0.6,1
1.0,2.0,3.0,4.0,5.0,6.0,0
