import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os

def knapsack_dp(values, weights, capacity):
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1), dtype=int)
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

def knapsack_ip(values, weights, capacity):
    n = len(values)
    
    
    model = gp.Model("Knapsack")
    model.setParam('OutputFlag', 0)  
    
    
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
    
    
    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    
    model.optimize()
    
    
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return 0  

def read_knapsack_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[1].strip())
    capacity = int(lines[2].strip())
    
    values = []
    weights = []
    for i in range(4, 4+n):
        if i < len(lines):  
            v, w = map(int, lines[i].strip().split())
            values.append(v)
            weights.append(w)
    
    return values, weights, capacity


def run_experiment(file_path):
    values, weights, capacity = read_knapsack_data(file_path)
    
    start = time.time()
    dp_value = knapsack_dp(values, weights, capacity)
    dp_time = time.time() - start
    
    start = time.time()
    ip_value = knapsack_ip(values, weights, capacity)
    ip_time = time.time() - start

    return {
        "instance": os.path.basename(file_path),
        "num_items": len(values),
        "capacity": capacity,
        "dp_value": dp_value,
        "dp_time": dp_time,
        "ip_value": ip_value,
        "ip_time": ip_time,
    }

if __name__ == "__main__":
    data = "data_kplib/s004_edit.kp"
    
    results = run_experiment(data)
    
    print(f"\nResults for edited s004.kp:")
    print(f"Number of items: {results['num_items']}")
    print(f"Capacity: {results['capacity']}")
    print(f"Dynamic Programming: Value = {results['dp_value']}, Time = {results['dp_time']:.4f}s")
    print(f"Integer Programming: Value = {results['ip_value']}, Time = {results['ip_time']:.4f}s") 
    
    
    os.makedirs("results", exist_ok=True)  
    with open("results/IPvsDP.txt", "w") as f:
        f.write("Instance,Items,Capacity,DP Value,DP Time,IP Value,IP Time\n")
        f.write(f"{results['instance']},{results['num_items']},{results['capacity']},{results['dp_value']},{results['dp_time']:.4f}," 
               f"{results['ip_value']},{results['ip_time']:.4f}\n")
    
    print("\nResults saved to results/IPvsDP.txt")