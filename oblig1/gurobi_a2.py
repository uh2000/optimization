import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from tqdm import tqdm

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

def knapsack_greedy(values, weights, capacity):
    n = len(values)
    ratio = [(values[i] / weights[i], i) for i in range(n)]
    ratio.sort(reverse=True, key=lambda x: x[0])
    
    total_value = 0
    total_weight = 0
    for _, i in ratio:
        if total_weight + weights[i] <= capacity:
            total_weight += weights[i]
            total_value += values[i]
    
    return total_value

def read_knapsack_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[1].strip())
    capacity = int(lines[2].strip())
    
    values = []
    weights = []
    for i in range(4, n+2):
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
    
    start = time.time()
    greedy_value = knapsack_greedy(values, weights, capacity)
    greedy_time = time.time() - start
    
    return {
        "instance": os.path.basename(file_path),
        "num_items": len(values),
        "capacity": capacity,
        "dp_value": dp_value,
        "dp_time": dp_time,
        "ip_value": ip_value,
        "ip_time": ip_time,
        "greedy_value": greedy_value,
        "greedy_time": greedy_time,
    }

if __name__ == "__main__":
    data_dir = "data_kplib"
    instance_files = [f for f in os.listdir(data_dir) if f.endswith('.kp')]
    
    print(f"Found {len(instance_files)} knapsack instances")
    
    results = []
    for file_name in tqdm(instance_files[:5], desc="Processing instances"):
        file_path = os.path.join(data_dir, file_name)
        result = run_experiment(file_path)
        results.append(result)
        
        print(f"\nResults for {file_name}:")
        print(f"Number of items: {result['num_items']}")
        print(f"Capacity: {result['capacity']}")
        print(f"Dynamic Programming: Value = {result['dp_value']}, Time = {result['dp_time']:.4f}s")
        print(f"Integer Programming: Value = {result['ip_value']}, Time = {result['ip_time']:.4f}s") 
        print(f"Greedy Algorithm: Value = {result['greedy_value']}, Time = {result['greedy_time']:.4f}s")
        print(f"Greedy vs Optimal gap: {(1 - result['greedy_value']/result['dp_value'])*100:.2f}%")
    
    os.makedirs("results", exist_ok=True)
    with open("results/knapsack_results.txt", "w") as f:
        f.write("Instance,Items,Capacity,DP Value,DP Time,IP Value,IP Time,Greedy Value,Greedy Time,Greedy Gap\n")
        for r in results:
            gap = (1 - r['greedy_value']/r['dp_value'])*100 if r['dp_value'] > 0 else 0
            f.write(f"{r['instance']},{r['num_items']},{r['capacity']},{r['dp_value']},{r['dp_time']:.4f}," 
                   f"{r['ip_value']},{r['ip_time']:.4f},{r['greedy_value']},{r['greedy_time']:.4f},{gap:.2f}%\n")
    
    print("\nResults saved to knapsack_results.txt")