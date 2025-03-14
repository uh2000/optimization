import time as time_module
import os as os_module
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict
import heapq

def read_pmed_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    parts = lines[0].split()
    n_nodes = int(parts[0])
    
    graph = defaultdict(dict)
    
    for i in range(1, len(lines)):
        parts = lines[i].split()
        if len(parts) >= 3:
            u, v, weight = int(parts[0]), int(parts[1]), int(parts[2])
            graph[u][v] = int(weight)
            graph[v][u] = int(weight)  
    
    return graph, n_nodes

def compute_shortest_paths(graph, n_nodes):

    dist = np.zeros((n_nodes+1, n_nodes+1), dtype=np.float32)
    dist.fill(float('inf'))
    
    for i in range(1, n_nodes+1):
        dist[i, i] = 0
    
    for u in range(1, n_nodes+1):
        for v, weight in graph[u].items():
            dist[u, v] = weight
    
    for k in range(1, n_nodes+1):
        for i in range(1, n_nodes+1):
            for j in range(1, n_nodes+1):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    return dist

def solve_min_diameter_problem(filename, p):
    start_time = time_module.time()
    graph, n_nodes = read_pmed_file(filename)
    distances = compute_shortest_paths(graph, n_nodes)
    
    model = gp.Model("p_median_min_diameter")
    
    model.setParam('OutputFlag', 1)
    model.setParam('LogToConsole', 1)
    model.setParam('DisplayInterval', 5)
    model.setParam('MIPGap', 0.2)
    model.setParam('TimeLimit', 3600)
    model.setParam('Threads', 0)
    model.setParam('Presolve', 2)
    model.setParam('Heuristics', 0.5)
    model.setParam('MIPFocus', 1)
    
    x = {}
    for j in range(1, n_nodes+1):
        x[j] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}")
    
    y = {}
    for i in range(1, n_nodes+1):
        for j in range(1, n_nodes+1):
            y[i,j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    
    max_diameter = model.addVar(name="max_diameter")
    
    model.addConstr(gp.quicksum(x[j] for j in range(1, n_nodes+1)) == p)
    
    for i in range(1, n_nodes+1):
        model.addConstr(gp.quicksum(y[i,j] for j in range(1, n_nodes+1)) == 1)
    
    for i in range(1, n_nodes+1):
        for j in range(1, n_nodes+1):
            model.addConstr(y[i,j] <= x[j])
    
    same_cluster = {}
    for i in range(1, n_nodes+1):
        for j in range(i+1, n_nodes+1):
            if distances[i,j] == float('inf'):
                continue
                
            same_cluster[i,j] = model.addVar(vtype=GRB.BINARY, name=f"same_{i}_{j}")
            
            model.addConstr(same_cluster[i,j] >= 
                            gp.quicksum(y[i,k] * y[j,k] for k in range(1, n_nodes+1)))
            
            model.addConstr(max_diameter >= distances[i,j] * same_cluster[i,j])
    
    model.setObjective(max_diameter, GRB.MINIMIZE)
    
    def progress_callback(model, where):
        if where == GRB.Callback.MIP:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            
            runtime = time_module.time() - start_time
            if runtime > model._last_report + 30:
                gap = abs(objbst - objbnd) / (1.0 + abs(objbst))
                print(f"         Progress update: Best objective = {objbst:.2f}, Bound = {objbnd:.2f}, Gap = {gap*100:.2f}%")
                model._last_report = runtime
    
    model._last_report = 0
    
    model.optimize(progress_callback)
    
    total_time = round(time_module.time() - start_time)
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.objVal
        
        service_units = [j for j in range(1, n_nodes+1) if x[j].x > 0.5]
        assignments = {}
        for i in range(1, n_nodes+1):
            for j in range(1, n_nodes+1):
                if y[i,j].x > 0.5:
                    assignments[i] = j
        
        clusters = defaultdict(list)
        for node, facility in assignments.items():
            clusters[facility].append(node)
        
        verified_max_diameter = 0
        for facility, nodes in clusters.items():
            for i in nodes:
                for j in nodes:
                    if distances[i,j] > verified_max_diameter:
                        verified_max_diameter = distances[i,j]
        
        base_filename = os_module.path.basename(filename)
        
        print(f"File: {base_filename}")
        print(f"Number of service units: {p}")
        print(f"Solution status: {'Optimal' if model.status == GRB.OPTIMAL else 'Time limit reached'}")
        print(f"Objective value: {objective_value}")
        print(f"Verified maximum diameter: {verified_max_diameter}")
        print(f"Runtime: {total_time} seconds")
        
        os_module.makedirs("results", exist_ok=True)
        
        base_filename = os_module.path.basename(filename)
        
        with open(f"results/{base_filename}_solution.txt", "w") as file:
            file.write(f"Instance: {filename}\n")
            file.write(f"Number of service units: {p}\n")
            file.write(f"Status: {'Optimal' if model.status == GRB.OPTIMAL else 'Time limit reached'}\n")
            file.write(f"Objective value: {objective_value}\n")
            file.write(f"Verified maximum diameter: {verified_max_diameter}\n")
            file.write(f"Runtime: {total_time} seconds\n\n")
            
            file.write(f"Service unit locations ({len(service_units)}):\n")
            file.write(", ".join(map(str, sorted(service_units))) + "\n\n")
            
            file.write("Cluster assignments:\n")
            for facility in sorted(clusters.keys()):
                nodes = sorted(clusters[facility])
                file.write(f"Facility {facility}: {len(nodes)} nodes - {', '.join(map(str, nodes))}\n")
        
        return objective_value, total_time
    else:
        print(f"Failed to find solution for {filename} with p={p}. Status: {model.status}")
        return None, total_time

def solve_all_pmed_instances():
    instances = {
        "pmed1.txt": 26,
        "pmed2.txt": 26,
        "pmed3.txt": 26,
        "pmed4.txt": 27,
        "pmed5.txt": 25,
    }
    
    results = {}
    total_instances = len(instances)
    current_instance = 1
    
    file_dir = "data_pmed/"
    
    for filename, p in instances.items():
        full_path = file_dir + filename
        print(f"\n===== Solving instance {current_instance}/{total_instances}: {full_path} with {p} service units =====")
        print(f"Step 1/3: Reading data from {full_path}...")
        
        step_start = time_module.time()
        graph, n_nodes = read_pmed_file(full_path)
        print(f"         Data loaded: {n_nodes} nodes, {sum(len(neighbors) for neighbors in graph.values())/2} edges")
        print(f"         Time: {time_module.time() - step_start:.2f} seconds")
        
        print(f"Step 2/3: Computing all-pairs shortest paths...")
        step_start = time_module.time()
        distances = compute_shortest_paths(graph, n_nodes)
        print(f"         Shortest paths computed")
        print(f"         Time: {time_module.time() - step_start:.2f} seconds")
        
        print(f"Step 3/3: Solving optimization model...")
        print(f"         This may take several minutes. Gurobi will display progress below:")
        print(f"         ------------------------------------------------------------")
        
        obj_val, runtime = solve_min_diameter_problem(full_path, p)
        results[filename] = (obj_val, runtime)
        
        print(f"         ------------------------------------------------------------")
        print(f"Instance {current_instance}/{total_instances} completed: {filename}")
        
        current_instance += 1
    
    print("\n" + "="*60)
    print("              SUMMARY OF RESULTS                ")
    print("="*60)
    print(f"{'Instance':<10} | {'P Value':<8} | {'Max Diameter':<12} | {'Time (s)':<8} | {'Status':<15}")
    print("-" * 60)
    
    total_time = 0
    
    with open("results/summary.txt", "w") as summary_file:
        summary_file.write("SUMMARY OF RESULTS\n")
        summary_file.write("=================\n\n")
        summary_file.write(f"{'Instance':<10} | {'P Value':<8} | {'Max Diameter':<12} | {'Time (s)':<8} | {'Status':<15}\n")
        summary_file.write("-" * 60 + "\n")
        
        for filename, (obj, time) in results.items():
            p = instances[filename]
            obj_str = f"{obj:.2f}" if obj is not None else "N/A"
            status = "Optimal" if obj is not None else "Failed"
            
            print(f"{filename:<10} | {p:<8} | {obj_str:<12} | {time:<8} | {status:<15}")
            
            summary_file.write(f"{filename:<10} | {p:<8} | {obj_str:<12} | {time:<8} | {status:<15}\n")
            
            total_time += time
        
        print("-" * 60)
        print(f"Total execution time: {total_time} seconds")
        print("="*60)
        
        summary_file.write("-" * 60 + "\n")
        summary_file.write(f"Total execution time: {total_time} seconds\n")
        summary_file.write("="*60 + "\n\n")
        summary_file.write("NOTE: The objective value represents the maximum diameter (largest distance\n")
        summary_file.write("      between any two nodes in the same cluster) across all clusters.\n")
    
    print("\nNOTE: The objective value represents the maximum diameter (largest distance")
    print("      between any two nodes in the same cluster) across all clusters.")
    print("\nDetailed results have been saved to the 'results' directory.")

if __name__ == "__main__":
    solve_all_pmed_instances()