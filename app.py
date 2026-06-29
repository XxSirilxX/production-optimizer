import streamlit as st
import pulp
import pandas as pd
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Production Planning Optimizer", layout="wide")

st.title("🇬🇧 Production Planning Optimizer")
st.caption("Operations Research-based Shift Scheduling & Inventory Control Panel")

# --- SIDEBAR: INTERACTIVE PARAMETERS ---
st.sidebar.header("🔧 SCM Parameters & Costs")

st.sidebar.subheader("Inventory & Safety Stock")
init_inv = st.sidebar.slider("Initial Inventory", 0, 10000, 3000, step=500)
final_inv = st.sidebar.slider("Required Final Inventory", 0, 10000, 2000, step=500)
holding_cost = st.sidebar.slider("Holding Cost / unit / mo (£)", 0.0, 10.0, 2.0, step=0.5)

st.sidebar.subheader("Shift Capacity & Costs")
normal_cap = st.sidebar.number_input("Normal Shift Max Capacity", value=5000)
normal_cost = st.sidebar.number_input("Normal Shift Cost (£)", value=100000)

extended_cap = st.sidebar.number_input("Extended Shift Max Capacity", value=7500)
extended_cost = st.sidebar.number_input("Extended Shift Cost (£)", value=180000)

min_prod = st.sidebar.number_input("Minimum Production (If Active)", value=2000)
switch_cost_val = st.sidebar.number_input("Normal ➔ Extended Switch Cost (£)", value=15000)

# --- MAIN PANEL: DEMAND PROFILER ---
st.subheader("📅 Supply Chain Demand Profiler")
st.write("Adjust monthly sales forecasts to compute shifts trade-offs instantly.")

col1, col2, col3, col4, col5, col6 = st.columns(6)
demands = []
with col1: demands.append(st.number_input("Month 1 Demand", value=6000))
with col2: demands.append(st.number_input("Month 2 Demand", value=6500))
with col3: demands.append(st.number_input("Month 3 Demand", value=7500))
with col4: demands.append(st.number_input("Month 4 Demand", value=7000))
with col5: demands.append(st.number_input("Month 5 Demand", value=6000))
with col6: demands.append(st.number_input("Month 6 Demand", value=6000))

# --- MILP SOLVER CORE ENGINE ---
def solve_production_planning():
    model = pulp.LpProblem("Production_Planning", pulp.LpMinimize)
    months = range(1, 7)
    
    # Decision Variables
    x = pulp.LpVariable.dicts("Prod_Qty", months, lowBound=0, cat='Continuous')
    y1 = pulp.LpVariable.dicts("Normal_Shift", months, cat='Binary')
    y2 = pulp.LpVariable.dicts("Extended_Shift", months, cat='Binary')
    I = pulp.LpVariable.dicts("Ending_Inv", months, lowBound=0, cat='Continuous')
    w = pulp.LpVariable.dicts("Switch_To_Extended", months, cat='Binary')
    
    # Objective Function
    model += pulp.lpSum([
        normal_cost * y1[t] + 
        extended_cost * y2[t] + 
        switch_cost_val * w[t] + 
        holding_cost * I[t] for t in months
    ])
    
    # Constraints
    for t in months:
        # Inventory Balance
        prev_inv = init_inv if t == 1 else I[t-1]
        model += prev_inv + x[t] - demands[t-1] == I[t]
        
        # Shift Selection Limit
        model += y1[t] + y2[t] <= 1
        
        # Production Boundaries
        model += x[t] >= min_prod * (y1[t] + y2[t])
        model += x[t] <= normal_cap * y1[t] + extended_cap * y2[t]
        
        # Transition Logic (Normal to Extended Switch)
        prev_y1 = 1 if t == 1 else y1[t-1]  # Problem states init stock came from normal shift
        model += w[t] >= prev_y1 + y2[t] - 1

    # Final Inventory target
    model += I[6] >= final_inv
    
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] == "Optimal":
        return model, x, y1, y2, I, w
    return None, None, None, None, None, None

# Run optimization
model, x, y1, y2, I, w = solve_production_planning()

if model and pulp.LpStatus[model.status] == "Optimal":
    
    # Prepare Data Matrix
    rows = []
    total_normal_cost = 0
    total_extended_cost = 0
    total_holding_cost = 0
    total_switch_cost = 0
    
    for t in range(1, 7):
        p_val = x[t].varValue
        s_type = "Normal" if y1[t].varValue == 1 else ("Extended" if y2[t].varValue == 1 else "Off")
        e_inv = I[t].varValue
        switched = "Yes" if w[t].varValue == 1 else "No"
        
        # Cost Breakdowns
        s_cost = normal_cost if s_type == "Normal" else (extended_cost if s_type == "Extended" else 0)
        h_cost = e_inv * holding_cost
        sw_cost = switch_cost_val if switched == "Yes" else 0
        p_cost = s_cost + h_cost + sw_cost
        
        total_normal_cost += s_cost if s_type == "Normal" else 0
        total_extended_cost += s_cost if s_type == "Extended" else 0
        total_holding_cost += h_cost
        total_switch_cost += sw_cost
        
        rows.append({
            "Period": f"Month {t}",
            "Shift Status": s_type,
            "Production Qty": int(p_val),
            "Demand": demands[t-1],
            "Ending Inv": int(e_inv),
            "Shift Cost": f"£{s_cost:,.2f}",
            "Holding Cost": f"£{h_cost:,.2f}",
            "Switch Cost": f"£{sw_cost:,.2f}",
            "Total Period Cost": f"£{p_cost:,.2f}"
        })
        
    df = pd.DataFrame(rows)
    total_min_cost = pulp.value(model.objective)
    
    # --- UI DISPLAY: METRIC CARDS ---
    st.write("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("TOTAL SCM COST", f"£{total_min_cost:,.2f}")
    m2.metric("NORMAL SHIFT COSTS", f"£{total_normal_cost:,.2f}")
    m3.metric("EXTENDED SHIFT COSTS", f"£{total_extended_cost:,.2f}")
    m4.metric("HOLDING COSTS", f"£{total_holding_cost:,.2f}")
    m5.metric("SWITCHING COSTS", f"£{total_switch_cost:,.2f}")
    
    # --- UI DISPLAY: CHART & TABLE ---
    st.write("---")
    tab1, tab2 = st.tabs(["📊 Optimization Curves", "📋 Optimal Supply Plan Matrix"])
    
    with tab1:
        fig = go.Figure()
        months_labels = [f"Month {t}" for t in range(1, 7)]
        fig.add_trace(go.Scatter(x=months_labels, y=demands, name='Demand Forecast', line=dict(color='red', width=3)))
        fig.add_trace(go.Bar(x=months_labels, y=[x[t].varValue for t in range(1, 7)], name='Optimal Production Plan', marker_color='royalblue'))
        fig.add_trace(go.Scatter(x=months_labels, y=[I[t].varValue for t in range(1, 7)], name='Ending Inventory Trajectory', line=dict(color='orange', dash='dash')))
        fig.update_layout(title="Production vs Demand Optimization Curves", xaxis_title="Timeline", yaxis_title="Units", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.error("❌ The current configuration is infeasible. Relax your constraints or check your inputs.")
