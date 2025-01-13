"""Some classes to make defining unit commitment/Powerflow in QUBO form 
and the read-out of binary solution data easier.
"""

import numpy as np
from dimod import BinaryQuadraticModel, ExactSolver
from typing import List
from hashlib import md5

class VariablesIndex:
    """A singleton class to keep track of the bit offsets for newly created 
    variables.
    """
    _instance = None
    _last_offset = 0
    _max_bit = 0
    all_number_index_offsets = {}
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    
    def register_number(self, var):
        if var not in self.all_number_index_offsets:
            self.all_number_index_offsets[var] = self._last_offset
            self._last_offset += var.num_bits
            self._max_bit = self._last_offset
    def reset_index(self):
        self.all_number_index_offsets = {}
        self._max_bit = 0
        self._last_offset = 0
        
# Function to convert QUBO matrix to dimod BQM
def qubo_to_bqm(Q):
    """
    Converts a QUBO matrix to a BinaryQuadraticModel.
    
    :param Q: QUBO matrix as a numpy array.
    :return: BinaryQuadraticModel object.
    """
    # Create a BinaryQuadraticModel
    bqm = BinaryQuadraticModel('BINARY')
    
    # Fill linear terms (diagonal of Q)
    for i in range(Q.shape[0]):
        bqm.add_variable(i, Q[i, i])
    
    # Fill quadratic terms (off-diagonal of Q)
    for i in range(Q.shape[0]):
        for j in range(i + 1, Q.shape[1]):
            if Q[i, j] != 0:
                bqm.add_interaction(i, j, Q[i, j])
    
    return bqm

class EncodedNumber:
    def __init__(self, name, max_value, num_bits, cost_per_unit=0, is_for_line = False):
        """
        Encodes a number as a binary vector.
        :param name: Name of the variable (for reference).
        :param max_value: Maximum value the number can take.
        :param num_bits: Number of binary bits used to encode the number.
        :param cost_per_unit: Cost per unit of production for this number.
        """
        self.name = name
        self._is_for_line = is_for_line
        self.max_value = max_value
        self.num_bits = num_bits
        ni = VariablesIndex()
        ni.register_number(self)
        _sc = self.max_value / (2**self.num_bits - 1)
        self.weights = np.array([2**i for i in range(num_bits)])* _sc  # Binary weights for each bit
        self.cost_per_unit = cost_per_unit        
        self.bit_from = ni.all_number_index_offsets[self]
        
    def set_bit_from(self, bit_from : int, no_renumber : bool = True):
        if self.bit_from is not None and no_renumber:
            raise Exception("Requested renumbering the bit range of the number - this is not allowed!")
        self.bit_from = bit_from
            
    def get_bit_inds(self):
        return np.arange(self.bit_from, self.bit_from + self.num_bits)
    def __neg__(self):
        ni = VariablesIndex()
        
        if self._is_for_line:
            cost_per_unit = 0 # to avoid double counting the flow cost
        else:
            cost_per_unit = self.cost_per_unit
        en = EncodedNumber('neg' + self.name, -self.max_value, self.num_bits, cost_per_unit)
        ni._max_bit -= en.num_bits
        ni._last_offset -= en.num_bits
        # to set it to the bit of the current variable.
        bit_from = ni.all_number_index_offsets[self]
        en.bit_from = bit_from
        ni.all_number_index_offsets[en] = bit_from 
        return en
    def __repr__(self):
        return f'{self.name}, {str(self.max_value)}, min_bit: {self.bit_from}'

class NumberSum:
    def __init__(self, encoded_numbers):
        """
        Represents a sum of encoded numbers, tracking bit offsets for easier handling.
        :param encoded_numbers: List of EncodedNumber objects.
        """
        self.encoded_numbers = encoded_numbers

    def get_global_bit_inds(self):
        """Returns the global bit indices for the encapsulated vars
        """
        return np.hstack([e.get_bit_inds() for e in self.encoded_numbers])

    def total_bits(self):
        """
        Computes the total number of bits across all EncodedNumbers.
        :return: Total number of bits.
        """
        return sum(num.num_bits for num in self.encoded_numbers)
    

    def create_qubo_matrix(self, target_sum, lambda_val=1, global_bit_range = True):
        """
        Constructs the QUBO matrix for the sum of the numbers, enforcing the constraint and considering costs.
        :param target_sum: The desired sum of all numbers.
        :param lambda_val: Penalty parameter.
        :param blobal_bit_range: Return to the global bit range (e.g., even if there is no 
            variable with bit index 50 and max is 50, the QUBO matrix will be of size 50x50)
        :return: QUBO matrix as a numpy array.
        """
        if global_bit_range:
            total_bits = VariablesIndex()._max_bit
        # total_bits = self.total_bits()
        Q = np.zeros((total_bits, total_bits))

        # Fill QUBO matrix for each number
        for num in self.encoded_numbers:
            indices = num.get_bit_inds()

            # Diagonal and off-diagonal terms within the number
            for i in range(num.num_bits):
                # Add cost term contribution
                Q[indices[i], indices[i]] += num.cost_per_unit * num.weights[i]
                
                # Add constraint penalty contribution
                Q[indices[i], indices[i]] += lambda_val * (num.weights[i] ** 2) - 2 * lambda_val * target_sum * num.weights[i]
                
                for j in range(i + 1, num.num_bits):
                    Q[indices[i], indices[j]] += 2 * lambda_val * num.weights[i] * num.weights[j]
                    Q[indices[j], indices[i]] += 2 * lambda_val * num.weights[i] * num.weights[j]

        # Cross-terms between different numbers
        for i, num1 in enumerate(self.encoded_numbers):
            for j, num2 in enumerate(self.encoded_numbers):
                if i >= j:
                    continue  # Avoid double-counting
                indices1 = num1.get_bit_inds()
                indices2 = num2.get_bit_inds()
                for k in range(num1.num_bits):
                    for l in range(num2.num_bits):
                        Q[indices1[k], indices2[l]] += 2 * lambda_val * num1.weights[k] * num2.weights[l]
                        Q[indices2[l], indices1[k]] += 2 * lambda_val * num1.weights[k] * num2.weights[l]

        return Q
    
class Plant:
    def __init__(self, plant_name = 'Plant', max_production = 10, cost_per_unit = 10, num_bits = 5):
        self.max_production = max_production
        self.cost_per_unit = cost_per_unit
        self.num_bits = num_bits
        self.name = plant_name
        self.number_repr = EncodedNumber(plant_name, max_value = max_production, num_bits=num_bits, cost_per_unit=self.cost_per_unit)
        self.production = None
        self.is_solved = False
    def set_solution(self, production):
        self.production = production
        self.is_solved = True
    def _compute_cost(self):
        if self.production is not None:
            return self.production * self.cost_per_unit
        else:
            return None
        
    def cost(self):
        if not self.is_solved:
            raise Exception("plant not solved! Cant give cost.")
        return self._compute_cost()
    def __repr__(self):
        return f'<Plant (name : {self.name}, prod: {self.production}, max_prod: {self.max_production}, cost: {self._compute_cost()})>'
    
class DemandBus:
    def __init__(self,   plants, demand, name = None):
        self.plants = plants
        if name is None:
            name = self.get_name_from_plants()
        self.name = name
        self.demand = demand
        self.plants_map  = {p.name : p for p in self.plants}
        self.number_sum = NumberSum([p.number_repr for p in self.plants])
        self.num_bits = sum([p.num_bits for p in self.plants])
        self.tot_production = None
        self.line_flows = None
        self.is_solved = False
        
    def get_name_from_plants(self):
        plants_name_agg = '.'.join([p.name for p in self.plants])
        return md5(plants_name_agg.encode()).hexdigest()
    
    def set_solution(self, tot_production, line_flows):
        self.tot_production = tot_production
        self.line_flows = line_flows
        self.is_solved = True
        
    def cost(self):
        if self.is_solved:
            tot_cost = 0
            for _,  p in self.plants_map.items():
                tot_cost += p.cost()
                
            return tot_cost
        raise Exception("Bus not solved - can't return cost.")
    def __repr__(self):
        if self.is_solved:
            return f'<Bus (name : {self.name} flows: {self.line_flows} production: {self.tot_production} demand: {self.demand} n_plants: {len(self.plants)})>'
    
class Line:

    def __init__(self, from_to : List[DemandBus], line_cost : float, line_capacity : float, num_bits = 5, name = None):
        self.from_to = from_to
        self.name = name
        self.line_cost = line_cost
        self.line_capacity = line_capacity 
        self.num_bits = num_bits
        self.number_repr = EncodedNumber(self.name, self.line_capacity, self.num_bits, cost_per_unit=self.line_cost, is_for_line=True)

    def get_from_to_and_num_bits(self):
        return self.from_to[0], self.from_to[1], self.num_bits
    
    def set_bit_offs(self,bit_offs):
        self.bit_offs = bit_offs
    def set_solution(self,transfer):
        self.transfer = transfer
    def __repr__(self):
        return f'<Line (name : {self.name}, transfer: {self.transfer}, line_capacity: {self.line_capacity}, line_cost: {self.line_cost})>'
    
class EnergyNetwork:
    def __init__(self, busses : List[DemandBus], lines : List[Line]):
        self.busses = busses # contains plants and a single demand (used directly in the constraint)
        self.lines = lines
        self._line_vars = set()
        for l in self.lines:
            self._line_vars.add(l.number_repr)
        self.node_bits_total = sum([b.num_bits for b in self.busses])
        self.line_bits_total = sum([l.num_bits for l in self.lines])
        self.is_compiled = False

    def compile(self):
        """if needed, add/subtract the line variables from the subs per-bus.
        Then the QUBO computation is simply computing the QUBO for the sums 
        and adding it. 
        """
        if not self.is_compiled:
            for l in self.lines:
                bus_from, bus_to = l.from_to
                n = l.number_repr
                if n not in bus_from.number_sum.encoded_numbers:
                    bus_from.number_sum.encoded_numbers.append(-n)
                    bus_to.number_sum.encoded_numbers.append(n)
        self.is_compiled = True
    
    def create_qubo_matrix(self, lambda_penalty_val = 100):
        """
        Constructs the QUBO matrix for the energy network.
        :param demands: List of energy demands for each node.
        :param lambda_bus_energy: Penalty parameter for energy conservation.
        :param lambda_transfer: Penalty parameter for transfer costs.
        :return: QUBO matrix as a numpy array.
        """
        self.compile()
        Q= None
        for b in self.busses:
            q_partial = b.number_sum.create_qubo_matrix(b.demand, lambda_penalty_val)
            if Q is None:
                Q = q_partial
                continue
            Q += q_partial
        return Q
    
    def set_solution(self, decoded_solution):
        bus_flows = {}
        bus_prod = {}
        for l in self.lines:
            l_val = decoded_solution[l.name]
            bus_from, bus_to = l.from_to
            if bus_from not in bus_flows:
                bus_flows[bus_from] = -l_val
            else:
                bus_flows[bus_from] -= l_val
            if bus_to not in bus_flows:
                bus_flows[bus_to] = l_val
            else:
                bus_flows[bus_to] += l_val
                


        for l in self.lines:
            l.set_solution(decoded_solution[l.name])
            
        for b in self.busses:
            _total_prod_bus = 0
            for p in b.plants:
                p.set_solution(decoded_solution[p.name])
                _total_prod_bus += decoded_solution[p.name]
            bus_prod[b] = _total_prod_bus
            b.set_solution(bus_prod[b], bus_flows[b])
class NetworkExactSolver:
    """Formulated the QUBO problem and solves it by 
    exhaustive search.
    """
    def __init__(self, nw : EnergyNetwork):
        self.nw = nw
        
    def get_qubo(self, lambda_val = 10):
        return self.nw.create_qubo_matrix(lambda_penalty_val=lambda_val)

    def solve(self, lambda_val = 10, qubo_solver = None):
        qubo_matrix = self.get_qubo(lambda_val)
        bqm = qubo_to_bqm(qubo_matrix)
        if qubo_solver is None:
            solver = ExactSolver()
            
        sampleset = solver.sample(bqm)
            
        # Extract the best solution
        best_solution = sampleset.first.sample  # Binary solution
        best_energy = sampleset.first.energy  # Energy of the best solution
        
        ni = VariablesIndex()
        
        # Decode the solution
        decoded_solution = {}
        for bus in self.nw.busses:
            number_sum_obj = bus.number_sum
            encoded_numbers = number_sum_obj.encoded_numbers
            for num in encoded_numbers:
                # if num not in self.nw._line_vars:
                bits_start = ni.all_number_index_offsets[num]
                bits_end = bits_start + num.num_bits
                bits = [best_solution[i] for i in range(bits_start, bits_end)]
            
                decoded_value = sum(b * w for b, w in zip(bits, num.weights))
                decoded_solution[num.name] = decoded_value
        
        self.nw.set_solution(decoded_solution)
        return decoded_solution
    