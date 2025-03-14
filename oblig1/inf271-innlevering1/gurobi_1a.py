import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import time
import os

def solve_minimum_dominating_set(V, E):
    

    model = gp.Model("Minimum_Dominating_Set")
    
    
    x = {i: model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in V}
    
    model.setObjective(gp.quicksum(x[i] for i in V), GRB.MINIMIZE)
    
    
    for i in V:
        neighbors = [j for j in V if (i, j) in E or (j, i) in E]
        model.addConstr(x[i] + gp.quicksum(x[j] for j in neighbors) >= 1, f"Coverage_{i}")
    

    start_time = time.time()
    model.setParam('OutputFlag', 1)  
    model.optimize()
    solve_time = time.time() - start_time
    

    if model.status == GRB.OPTIMAL:
        selected_nodes = [i for i in V if x[i].X > 0.5]  
        
        stats = {
            "objective_value": model.objVal,
            "status": "Optimal" if model.status == GRB.OPTIMAL else "Not optimal",
            "cpu_time": solve_time,
            "node_count": model.nodeCount,
            "mip_gap": model.mipGap
        }
        
        return selected_nodes, stats
    else:
        return [], {"status": f"Failed with status code {model.status}", "cpu_time": solve_time}

def read_instance(filename):
    """ Reads an OR-Lib instance file and extracts graph data. """
    with open(filename, "r") as file:
        lines = file.readlines()
        
    
    V_size, E_size = map(int, lines[0].split()[:2])
    V = list(range(1, V_size + 1))  
    E = set()
    
    
    for line in lines[1:]:
        i, j, _ = map(int, line.split())  
        E.add((i, j))
        E.add((j, i))  
    
    return V, E


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    for i in range(1, 6):
        print(f"\nSolving pmed{i}.txt...")
        V, E = read_instance(f"data_pmed/pmed{i}.txt")
        solution, stats = solve_minimum_dominating_set(V, E)
        
        print("Selected service unit locations:", solution)
        print(f"Number of service units: {len(solution)}")
        print(f"Objective value: {stats.get('objective_value', 'N/A')}")
        print(f"Status: {stats['status']}")
        print(f"CPU time: {stats['cpu_time']:.2f} seconds")
        print(f"MIP Gap: {stats.get('mip_gap', 'N/A')}")
        print(f"Node count: {stats.get('node_count', 'N/A')}")
        
        with open(f"results/pmed{i}_solution.txt", "w") as file:
            file.write(f"Selected service unit locations: {solution}\n")
            file.write(f"Number of service units: {len(solution)}\n")
            file.write(f"Total number of stations: {len(V)}\n")
            file.write(f"Objective value: {stats.get('objective_value', 'N/A')}\n")
            file.write(f"Solution status: {stats['status']}\n")
            file.write(f"Time (CPU seconds): {stats['cpu_time']:.2f}\n")
            if 'mip_gap' in stats:
                file.write(f"MIP Gap: {stats['mip_gap']}\n")
            if 'node_count' in stats:
                file.write(f"Node count: {stats['node_count']}\n")
            
            
            file.write("\nService units installed at stations:\n")
            for node in solution:
                file.write(f"{node}\n")
                
        print(f"Solution {i} written to file.")