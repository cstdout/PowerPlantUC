import pandas as pd
import pyomo.environ as pyo
from milpoptimizer import solve_pyomo

if __name__ == "__main__":

    ntypes = 3
    nperiods = 5
    maxstart0 = 5

    generators = [12, 10, 5]
    period_hours = [6, 3, 6, 3, 6]
    demand = [15000, 30000, 25000, 40000, 27000]
    min_load = [850, 1250, 1500]
    max_load = [2000, 1750, 4000]
    base_cost = [1000, 2600, 3000]
    per_mw_cost = [2, 1.3, 3]
    startup_cost = [2000, 1000, 500]


    def numgen_rule(model, t, p):
        return model.ngen[t, p] <= generators[t]


    def min_output_rule(model, t, p):
        return model.genoutput[t, p] >= min_load[t] * model.ngen[t, p]


    def max_output_rule(model, t, p):
        return model.genoutput[t, p] <= max_load[t] * model.ngen[t, p]


    def meet_demand_rule(model, p):
        return sum(model.genoutput[t, p] for t in model.TYPES) >= demand[p]


    def reserve_rule(model, p):
        return sum(max_load[t] * model.ngen[t, p] for t in model.TYPES) >= 1.15 * demand[p]


    def startup0_rule(model, t):
        return model.ngen[t, 0] <= maxstart0 + model.nstart[t, 0]


    def startup_rule(model, t, p):
        if p == 0:
            return pyo.Constraint.Skip
        return model.ngen[t, p] <= (model.ngen[t, p - 1] + model.nstart[t, p])


    def objective_rule(model):
        active = 0
        for t in model.TYPES:
            for period in model.PERIODS:
                active += (base_cost[t] * period_hours[period] * model.ngen[t, period])

        per_mw = 0
        for t in model.TYPES:
            for period in model.PERIODS:
                per_mw += (per_mw_cost[t] * period_hours[period] * (model.genoutput[t, period] - min_load[t] * model.ngen[t, period]))

        startup_obj = 0
        for t in model.TYPES:
            for period in model.PERIODS:
                startup_obj += startup_cost[t] * model.nstart[t, period]

        return active + per_mw + startup_obj


    model = pyo.ConcreteModel(name="PowerGeneration")

    model.TYPES = pyo.RangeSet(0, ntypes - 1)
    model.PERIODS = pyo.RangeSet(0, nperiods - 1)

    model.ngen = pyo.Var(model.TYPES, model.PERIODS, domain=pyo.NonNegativeIntegers)
    model.nstart = pyo.Var(model.TYPES, model.PERIODS, domain=pyo.NonNegativeIntegers)
    model.genoutput = pyo.Var(model.TYPES, model.PERIODS, domain=pyo.NonNegativeReals)


    model.numgen = pyo.Constraint(model.TYPES, model.PERIODS, rule=numgen_rule)
    model.min_output = pyo.Constraint(model.TYPES, model.PERIODS, rule=min_output_rule)
    model.max_output = pyo.Constraint(model.TYPES, model.PERIODS, rule=max_output_rule)

    model.meet_demand = pyo.Constraint(model.PERIODS, rule=meet_demand_rule)
    model.reserve = pyo.Constraint(model.PERIODS, rule=reserve_rule)
    model.startup0 = pyo.Constraint(model.TYPES, rule=startup0_rule)

    model.startup = pyo.Constraint(model.TYPES, model.PERIODS, rule=startup_rule)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    res = solve_pyomo(model, time_limit=60, mip_gap=1e-4)
    print(res)                               

    rows = ["Type" + str(t) for t in range(ntypes)]
    units = pd.DataFrame(columns=range(nperiods), index=rows, data=0.0)
    for t in range(ntypes):
        for p in range(nperiods):
            units.loc["Type" + str(t), p] = model.ngen[t, p].x # ngen[t, p].value
    print(units)
