import gurobipy as gp
from gurobipy import GRB

# Data
m = 3  # Number of crude materials (m)
k = [3.0, 2.0, 1.0]  # Unit purchase cost of each crude material (k)
n = 2  # Number of end products (n)
T = 3  # Number of market price segments (T)
p1 = [4.0, 3.5, 3.0]  # Unit market prices of end product 1 (p_{1k})
b1 = [10, 10, float('inf')]  # Max amount sold at each market price of end product 1 (b_{1k})
p2 = [2.0, 1.5, 1.0]  # Unit market prices of end product 2 (p_{2k})
b2 = [15, 15, float('inf')]  # Max amount sold at each market price of end product 2 (b_{2k})
p = [p1, p2]  # Market prices of end products
b = [b1, b2]  # Max amounts sold at each price segment for end products
a = [
    [0.7, 0.2],
    [0.4, 0.4],
    [0.3, 0.6]
]  # Yields (one row per crude material)
c_in = 110  # Input capacity
c_out = 90  # Output capacity

# Create a new model
model = gp.Model("LP")

# Add variables u, v, and z
u = model.addVars(m, lb=0, name="u")  # Supply of crude i element of m
v = model.addVars(n, T, lb=0, name="v")  # Production of product j element of n at price segment k
z = model.addVars(n, lb=0, name="z")  # Revenue obtained by selling product j

# Set the objective function
model.setObjective(
    gp.quicksum(z[j] for j in range(n)) - gp.quicksum(k[i] * u[i] for i in range(m)),
    GRB.MAXIMIZE
)

# Add constraints
model.addConstr(gp.quicksum(u[i] for i in range(m)) <= c_in, "input_capacity")
model.addConstr(gp.quicksum(v[j, t] for j in range(n) for t in range(T)) <= c_out, "output_capacity")

for j in range(n):
    model.addConstr(gp.quicksum(v[j, k] for k in range(T)) == gp.quicksum(a[i][j] * u[i] for i in range(m)), f"yield_constraint_{j}")
    model.addConstr(z[j] == gp.quicksum(p[j][k] * v[j, k] for k in range(T)), f"revenue_constraint_{j}")


for j in range(n):
    for t in range(T):
        model.addConstr(v[j, t] <= b[j][t], f"max_amount_sold_{j}_{t}")

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    u_values = model.getAttr('x', u)
    v_values = model.getAttr('x', v)
    z_values = model.getAttr('x', z)
    print("\nOptimal solution:")
    print("u:", [u_values[i] for i in range(m)])
    for j in range(n):
        print(f"v[{j}]:", [v_values[j, t] for t in range(T)] )
    print("z:", [z_values[j] for j in range(n)])
    
    # Print the optimal value of the objective function
    print("\nOptimal objective value:", model.objVal)
 