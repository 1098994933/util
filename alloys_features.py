import os
import pandas as pd
import numpy as np
from itertools import combinations
import re

from Config import Config

project_dataset_path = Config['project_dataset_path']
element_data = pd.read_excel(os.path.join(project_dataset_path, "element_data.xlsx"), index_col=0)


def find_elements(string):
    """
    split formula to element and its ratio
    """
    pattern = r'([A-Z][a-z]?)(\d\.*\d*)'
    elements = re.findall(pattern, string)

    element_dict = {}
    for element, proportion in elements:
        if proportion == '':
            proportion = '1'
        element_dict[element] = float(proportion)
    return element_dict


class AlloyFeature(object):
    def __init__(self, ci, ei):
        self.ele_data = element_data
        self.ci = ci
        self.ei = ei  # element name
        self.ri = np.array([self.ele_data.loc[e, "Atomic radius"] for e in self.ei])
        self.r_hat = np.sum(self.ci * self.ri)

        self.feature_names = ["Atomic size difference",
                              "Atomic packing effect",
                              "Cohesion energy",
                              "Shear modulus",
                              "Lattice distortion energy",
                              "Difference in Shear modulus",
                              "Melting temperature",
                              "Entropy of mixing",
                              "Mixing enthalpy",
                              "Solid solution phase forming ability",
                              "Phase formation coefficient", "VEC",
                              "Energy term", "Local size mismatch in n-element alloys",
                              "Local modulus mismatch",
                              "Peierls-Nabarro stress coefficient",
                              "Work function"
                              ]

        # self.feature_values = [self.delat_r, self.gama]

    def get_features(self):
        self.get_delat_r()
        self.get_gama()
        self.get_ec()
        self.get_g()
        self.get_delta_g()
        self.get_tm()
        self.get_entropy_of_mixing()
        self.get_mixing_enthalpy()
        self.get_fai()
        self.get_energy_term()
        self.get_dr()
        self.get_f()
        self.get_work_function()
        return [self.delat_r, self.gama,
                self.ec, self.g,
                self.u, self.delat_g,
                self.tm, self.entropy_of_mixing,
                self.get_mixing_enthalpy(),
                self.fai, self.pfc,
                self.vec, self.energy_term,
                self.dr, self.dg, self.f, self.w]

    def get_delat_r(self):
        self.delat_r = np.sum(self.ci * (1 - self.ri / self.r_hat) ** 2) ** 0.5
        return self.delat_r

    def get_gama(self):
        """Atomic packing effect in n-element alloys.
        rS and rL are the atomic radius of the smallest-size and largest-size atoms"""
        self.rs = min(self.ri)
        self.rl = max(self.ri)
        a = (self.rs + self.r_hat) ** 2
        b = self.r_hat ** 2
        self.gama_1 = 1 - ((a - b) / a) ** 0.5
        a = (self.rl + self.r_hat) ** 2
        b = self.r_hat ** 2
        self.gama_2 = 1 - ((a - b) / a) ** 0.5
        self.gama = self.gama_1 / self.gama_2
        return self.gama

    def get_ec(self):
        """Cohesion energy"""
        self.eci = np.array([self.ele_data.loc[e, "Cohesive_Energy(kJ/mol)"] for e in self.ei])
        self.ec = np.sum(self.ci * self.eci)
        # Valence electron concentration
        self.veci = np.array([self.ele_data.loc[e, "VEC"] for e in self.ei])
        self.vec = np.sum(self.ci * self.veci)
        return self.ec

    def get_g(self):
        """
        Rigidity_Modulus is Shear modulus
        """
        self.gi = np.array([self.ele_data.loc[e, "Rigidity_Modulus"] for e in self.ei])
        self.g = np.sum(self.ci * self.gi)
        # Young's modulus
        self.e = np.sum(self.ci * np.array([self.ele_data.loc[e, "Elastic_Modulus"] for e in self.ei]))
        # u Lattice distortion energy
        self.u = 0.5 * self.e * self.delat_r
        return self.g

    def get_delta_g(self):
        """
        """
        self.delat_g = np.sum(self.ci * (1 - self.gi / self.g) ** 2) ** 0.5
        return self.delat_g

    def get_tm(self):
        """Tm"""
        self.tmi = np.array([self.ele_data.loc[e, "Cohesive_Energy(kJ/mol)"] for e in self.ei])
        self.tm = np.sum(self.ci * self.tmi)
        return self.tm

    def get_entropy_of_mixing(self):
        """Entropy of mixing"""
        self.entropy_of_mixing = - 8.314 * np.sum(self.ci * np.log(self.ci))
        return self.entropy_of_mixing

    def get_mixing_enthalpy(self):
        """Mixing enthalpy"""
        combins = [c for c in combinations(list(range(len(self.ci))), 2)]
        self.mixing_enthalpy = 0
        for (i, j) in combins:
            # print(self.ci[i], self.ci[j], self.ri[i], self.ri[j])
            self.mixing_enthalpy = 4 * self.ci[i] * self.ci[j] * self.ele_data.loc[self.ei[i], self.ei[j]]
        return self.mixing_enthalpy

    def get_phase_formation_coefficient(self):
        self.phase_formation_coefficient = self.entropy_of_mixing / (self.delat_r ** 2)

    def get_work_function(self):
        self.wi = np.array([self.ele_data.loc[e, "Elemental work function"] for e in self.ei])
        self.w = np.sum(self.ci * self.wi)
        return self.w

    def get_dr(self):
        # Local size mismatch in n-element alloys
        combins = [c for c in combinations(list(range(len(self.ci))), 2)]
        self.dr = 0
        for (i, j) in combins:
            # (self.ci[i], self.ci[j], self.ri[i], self.ri[j])
            self.dr = self.dr + self.ci[i] * self.ci[j] * abs(self.ri[i] - self.ri[j])
        # Local modulus mismatch
        combins = [c for c in combinations(list(range(len(self.ci))), 2)]
        self.dg = 0
        for (i, j) in combins:
            self.dg = self.dg + self.ci[i] * self.ci[j] * abs(self.gi[i] - self.gi[j])
        return self.dr
        # print(combins)

    def get_energy_term(self):
        """Energy term"""
        self.energy_term = self.g * self.delat_r * (1 + self.u) * (1 - self.u)
        return self.energy_term

    def get_f(self):
        # Peierls-Nabarro stress coefficient
        self.f = 2 * self.g / (1 - self.u)

    def get_fai(self):
        # Solid solution phase forming ability
        if self.mixing_enthalpy != 0:
            self.fai = self.tm * self.entropy_of_mixing / abs(self.mixing_enthalpy)
        else:
            self.fai = 0
        # Phase formation coefficient
        self.pfc = self.entropy_of_mixing / (self.delat_r ** 2)


if __name__ == '__main__':
    ci = np.array([0.5, 0.4, 0.1])
    ei = ["Al", "Cu", 'Zn']
    af = AlloyFeature(ci, ei)
    print(af.ele_data)
    print(af.get_delat_r())
    print(af.get_features())
    dataset = pd.read_csv("../data/formula.csv")
    print(dataset)
    s = dataset['formula'][1]
    input_string = s
    elements = find_elements(input_string)
    print(elements)
    ci = np.array(list(elements.values())) / np.sum(np.array(list(elements.values())))
    ei = np.array(list(elements.keys()))

    af = AlloyFeature(ci, ei)
    print(af.get_features())

    features = []
    for formula in dataset['formula']:
        elements = find_elements(formula)
        ci = np.array(list(elements.values())) / np.sum(np.array(list(elements.values())))
        ei = np.array(list(elements.keys()))
        af = AlloyFeature(ci, ei)
        features.append(af.get_features())
    df = pd.DataFrame(features, columns=af.feature_names)
    print(df)
    df.to_csv("alloy_features.csv", index=False)
    # print(re.split("[A-Za-z]+.*?", s))
    # print(re.split("\w+", s))
    # ri = for i in af.ele_data
