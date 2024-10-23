import gurobipy as gp
from gurobipy import GRB

# Data for the problem
num_crude_materials = 3 # Number of crude materials
unit_purchase_cost = [3, 2, 1] # Unit purchase cost of each crude material
num_end_products = 2 # Number of end products
unit_market_price = [4, 2] # Unit market price of each end product
yields = [ # Yield of each crude material for each end product
    [0.7, 0.2],
    [0.4, 0.4],
    [0.3, 0.6]
]
input_capacity = 11 # Input capacity
output_capacity = 9 # Output capacity

# Create the LP
model = gp.Model("LP")

# Variables u and v
u = model.addVars(num_crude_materials, lb=0, name="u")
v = model.addVars(num_end_products, lb=0, name="v")

# Objective function
model.setObjective(
    gp.quicksum(unit_market_price[j] * v[j] for j in range(num_end_products)) - 
    gp.quicksum(unit_purchase_cost[i] * u[i] for i in range(num_crude_materials)), 
    GRB.MAXIMIZE
)

# Constraints
input_constr = model.addConstr(gp.quicksum(u[i] for i in range(num_crude_materials)) <= input_capacity, "input_capacity")
output_constr = model.addConstr(gp.quicksum(v[j] for j in range(num_end_products)) <= output_capacity, "output_capacity")

yield_constrs = []
for j in range(num_end_products):
    yield_constrs.append(model.addConstr(v[j] - gp.quicksum(yields[i][j] * u[i] for i in range(num_crude_materials)) == 0, f"constraint_{j}"))

# Run the model
model.optimize()
print("Objective function optimal value:", model.ObjVal)


if model.status == GRB.OPTIMAL:
    u_values = model.getAttr('x', u)
    v_values = model.getAttr('x', v)
    print("\nOptimal solution:")
    print("u:", [u_values[i] for i in range(num_crude_materials)])
    print("v:", [v_values[j] for j in range(num_end_products)])
    
    # We us the dual values of the constraints
    # to get the highest bid on input and output capacity
    print("Highest bid on input capacity:", input_constr.getAttr("pi"))
    print("Highest bid on output capacity:", output_constr.getAttr("pi"))

else:
    print("No optimal solution found.")