"""
knowledge-aware feature calculation
"""
import os
import pandas as pd
import numpy as np
from itertools import combinations
import re
from util.Config import Config

project_dataset_path = Config['project_dataset_path']
element_data = pd.read_excel(os.path.join(project_dataset_path, "element_data.xlsx"), index_col=0)


def find_elements(string):
    """
    parse formula to element and its ratio as dict
    """
    pattern = r'([A-Z][a-z]?)(\d\.*\d*)'
    elements = re.findall(pattern, string)

    element_dict = {}
    for element, proportion in elements:
        if proportion == '':
            proportion = '1'
        element_dict[element] = float(proportion)
    return element_dict


def formula_to_features(formula_list):
    """
    :param dataset:
    :return:
    """
    features = []
    for formula in formula_list:
        elements = find_elements(formula)
        # calculate ci and ei
        ci = np.array(list(elements.values())) / np.sum(np.array(list(elements.values())))
        ei = np.array(list(elements.keys()))
        af = AlloyFeature(ci, ei)
        features.append(af.get_features())
    df = pd.DataFrame(features, columns=af.feature_names)
    return df


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
                              "Phase formation coefficient",
                              "VEC",
                              "Energy term",
                              "Local size mismatch in n-element alloys",
                              "Local modulus mismatch",
                              "Peierls-Nabarro stress coefficient",
                              "Work function",
                              "Local atomic distortion from one single atom",
                              "Local atomic distortion from a group of adjacent atoms",
                              "Pauling electronegativity",
                              "Allen electronegativity",
                              "Number of mobile electrons",
                              ]
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
        self.get_local_atomic_distortion()
        self.get_local_atomic_distortion2()
        self.get_pauling_electronegativity()
        self.get_allen_electronegativity()
        self.get_ed()
        return [self.delat_r, self.gama,
                self.ec, self.g,
                self.u, self.delat_g,
                self.tm, self.entropy_of_mixing,
                self.get_mixing_enthalpy(),
                self.fai, self.pfc,
                self.vec, self.energy_term,
                self.dr, self.dg, self.f, self.w,
                self.alpha1, self.alpha2, self.pe, self.ae, self.ed]

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
        self.tmi = np.array([self.ele_data.loc[e, "Elemental melting temperature"] for e in self.ei])
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
            if self.ei[i] != self.ei[j]:
                try:
                    mix = self.ele_data.loc[self.ei[i], self.ei[j]]
                except:
                    try:
                        mix = self.ele_data.loc[self.ei[j], self.ei[i]]
                    except:
                        mix = 0
                self.mixing_enthalpy = self.mixing_enthalpy + 4 * self.ci[i] * self.ci[j] * mix
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
            self.dr = self.dr + self.ci[i] * self.ci[j] * abs(self.ri[i] - self.ri[j])
        # Local modulus mismatch
        combins = [c for c in combinations(list(range(len(self.ci))), 2)]
        self.dg = 0
        for (i, j) in combins:
            self.dg = self.dg + self.ci[i] * self.ci[j] * abs(self.gi[i] - self.gi[j])
        return self.dr

    def get_energy_term(self):
        """Energy term"""
        self.energy_term = self.g * self.delat_r * (1 + self.u) * (1 - self.u)
        return self.energy_term

    def get_f(self):
        """
        Peierls-Nabarro stress coefficient
        """
        self.f = 2 * self.g / (1 - self.u)

    def get_fai(self):
        # Solid solution phase forming ability
        if self.mixing_enthalpy != 0:
            # the units of mixing_enthalpy is kJ and entropy_of_mixing is J
            self.fai = self.tm * self.entropy_of_mixing / abs(self.mixing_enthalpy) / 1000
        else:
            self.fai = 0
        # Phase formation coefficient
        self.pfc = self.entropy_of_mixing / (max(self.delat_r ** 2, 0.0001))

    def get_local_atomic_distortion(self):
        """
        from one single atom alone
        """
        self.alpha1 = np.sum(self.ci * abs(self.ri - self.r_hat) / self.r_hat)
        return self.alpha1

    def get_local_atomic_distortion2(self):
        combins = [c for c in combinations(list(range(len(self.ci))), 2)]
        self.alpha2 = 0
        for (i, j) in combins:
            self.alpha2 = self.dg + self.ci[i] * self.ci[j] * abs(self.ri[i] + self.ri[j] - 2 * self.r_hat) / (
                    2 * self.r_hat)
        return self.alpha2

    def get_pauling_electronegativity(self):
        """Pauling electronegativity"""
        self.pei = np.array([self.ele_data.loc[e, "Electronegativity Pauling"] for e in self.ei])
        self.pe_hat = np.sum(self.ci * self.pei)
        self.pe = (np.sum(self.ci * (self.pei - self.pe_hat) ** 2)) ** 0.5
        return self.pe

    def get_allen_electronegativity(self):
        self.aei = np.array([self.ele_data.loc[e, "Electronegativity Allred"] for e in self.ei])
        self.ae_hat = np.sum(self.ci * self.aei)
        self.ae = np.sum(self.ci * abs(1 - self.aei / self.ae_hat) ** 2)
        return self.ae

    def get_ed(self):
        """ Elemental electron density"""
        self.edi = np.array([self.ele_data.loc[e, "Elemental electron density"] for e in self.ei])
        self.ed = np.sum(self.ci * self.edi)
        return self.ed


if __name__ == '__main__':
    # test case 1
    # ci = np.array([0.5, 0.4, 0.1])
    # ei = ["Al", "Cu", 'Zn']
    # af = AlloyFeature(ci, ei)
    # # test case 2
    # dataset = pd.read_csv("../data/formula.csv")
    # s = dataset['formula'][1]
    # input_string = s
    # elements = find_elements(input_string)
    # ci = np.array(list(elements.values())) / np.sum(np.array(list(elements.values())))
    # ei = np.array(list(elements.keys()))
    # af = AlloyFeature(ci, ei)

    # knowledge-aware feature calculation for training dataset
    dataset = pd.read_csv("../data/formula.csv")
    df1 = formula_to_features(dataset['formula'])
    print(df1)
    df1.to_csv("../data/alloy_features.csv", index=False)

    # designed alloys knowledge-aware feature calculation
    # designed_dataset = pd.read_csv("../data/formula_design.csv")
    # designed_feature = formula_to_features(designed_dataset['formula'])
    # f_name = "alloy"
    # designed_feature.to_csv(f"../data/formula_design_{f_name}_features.csv", index=False)
