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
    pattern = r'([A-Z][a-z]?)(\d*\.*\d*)'
    elements = re.findall(pattern, string)

    element_dict = {}
    for element, proportion in elements:
        if proportion == '':
            proportion = '1'
        element_dict[element] = float(proportion)
    return element_dict


def formula_to_features(formula_list):
    """
    :param formula_list:
    ci is the ratio of element_i (ei)
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


def normalize_element_dict(element_dict):
    """
    normalize the ratio to make the sum to 1
    :param element_dict:
    :return:
    """
    normalized_element_dict = {}
    ci = np.array(list(element_dict.values())) / np.sum(np.array(list(element_dict.values())))
    ei = np.array(list(element_dict.keys()))
    for index in range(len(ei)):
        normalized_element_dict[ei[index]] = ci[index]
    return normalized_element_dict


def formula_to_ratio_dataset(dataset):
    """
    :param dataset: contain the formula column with alloys formula
    :return:
    """
    df_ratio = pd.DataFrame()
    if not "formula" in list(dataset.columns):
        raise AssertionError(f"Error: no column named formula is not in the dataset !")
    for i, formula in enumerate(dataset.formula):
        element_dict = find_elements(formula)
        normalized = normalize_element_dict(element_dict)
        # print(normalized)
        df_formula = pd.DataFrame(data=normalized, index=[i])
        df_ratio = pd.concat([df_ratio, df_formula])
    df_ratio = df_ratio.fillna(0).reset_index(drop=True)
    dataset_reindex = dataset.reset_index(drop=True)
    df_all = pd.concat([dataset_reindex, df_ratio], axis=1)
    df_all.columns = [str(col) for col in df_all.columns]  # 防止np.str_导致报错
    element_columns = list(df_ratio.columns)
    return df_all, element_columns


class AlloyFeature(object):
    def __init__(self, ci, ei):
        self.element_data = element_data
        self.element_fractions = ci  # 比例
        self.element_symbols = ei  # 元素
        self.atomic_radii = np.array([self.element_data.loc[e, "Atomic radius"] for e in self.element_symbols])
        self.average_atomic_radius = np.sum(self.element_fractions * self.atomic_radii)  # 加权平均原子半径

        # 物理常数
        self.planck_constant = 6.62607015e-34  # 普朗克常数 (J·s)
        self.boltzmann_constant = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.avogadro_constant = 6.02214076e23  # 阿伏伽德罗常数 (1/mol)

        self.feature_names = ["Atomic size difference",
                              "Atomic packing effect",
                              "Cohesion energy",
                              "Shear modulus",  # 剪切模量
                              "Yang's modulus",
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
                              "Estimated atomic volume per atom",
                              "Estimated atomic number density",
                              "Debye wavevector (proxy)",
                              "Mean sound velocity (proxy)",
                              "Debye temperature (proxy)",
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
        self.get_debye_related_features()
        return [self.atomic_size_mismatch, self.atomic_packing_effect,
                self.cohesive_energy, self.shear_modulus, self.youngs_modulus,
                self.lattice_distortion_energy, self.shear_modulus_mismatch,
                self.melting_temperature, self.entropy_of_mixing,
                self.mixing_enthalpy,
                self.solid_solution_forming_ability, self.phase_formation_coefficient,
                self.valence_electron_concentration, self.energy_term,
                self.local_size_mismatch, self.local_modulus_mismatch, self.peierls_nabarro_coefficient,
                self.work_function,
                self.local_atomic_distortion_single, self.local_atomic_distortion_group,
                self.pauling_electronegativity_deviation, self.allen_electronegativity, self.electron_density,
                self.estimated_atomic_volume, self.estimated_number_density, self.debye_wavevector_proxy,
                self.mean_sound_velocity_proxy, self.debye_temperature_proxy]

    def get_delat_r(self):
        self.atomic_size_mismatch = np.sum(
            self.element_fractions * (1 - self.atomic_radii / self.average_atomic_radius) ** 2) ** 0.5
        return self.atomic_size_mismatch

    def get_gama(self):
        """Atomic packing effect in n-element alloys.
        rS and rL are the atomic radius of the smallest-size and largest-size atoms"""
        self.min_atomic_radius = min(self.atomic_radii)
        self.max_atomic_radius = max(self.atomic_radii)
        a = (self.min_atomic_radius + self.average_atomic_radius) ** 2
        b = self.average_atomic_radius ** 2
        self.atomic_packing_effect_small = 1 - ((a - b) / a) ** 0.5
        a = (self.max_atomic_radius + self.average_atomic_radius) ** 2
        b = self.average_atomic_radius ** 2
        self.atomic_packing_effect_large = 1 - ((a - b) / a) ** 0.5
        self.atomic_packing_effect = self.atomic_packing_effect_small / self.atomic_packing_effect_large
        return self.atomic_packing_effect

    def get_ec(self):
        """Cohesion energy"""
        self.cohesive_energies = np.array(
            [self.element_data.loc[e, "Cohesive_Energy(kJ/mol)"] for e in self.element_symbols])
        self.cohesive_energy = np.sum(self.element_fractions * self.cohesive_energies)
        # Valence electron concentration
        self.element_valence_electron_concentrations = np.array(
            [self.element_data.loc[e, "VEC"] for e in self.element_symbols])
        self.valence_electron_concentration = np.sum(
            self.element_fractions * self.element_valence_electron_concentrations)
        return self.cohesive_energy

    def get_g(self):
        """
        Rigidity_Modulus is Shear modulus
        """
        self.element_shear_moduli = np.array(
            [self.element_data.loc[e, "Rigidity_Modulus"] for e in self.element_symbols])
        self.shear_modulus = np.sum(self.element_fractions * self.element_shear_moduli)
        # Young's modulus
        self.youngs_modulus = np.sum(self.element_fractions * np.array(
            [self.element_data.loc[e, "Elastic_Modulus"] for e in self.element_symbols]))
        # u Lattice distortion energy
        self.lattice_distortion_energy = 0.5 * self.youngs_modulus * self.atomic_size_mismatch
        return self.shear_modulus, self.youngs_modulus

    def get_delta_g(self):
        """
        """
        self.shear_modulus_mismatch = np.sum(
            self.element_fractions * (1 - self.element_shear_moduli / self.shear_modulus) ** 2) ** 0.5
        return self.shear_modulus_mismatch

    def get_tm(self):
        """Tm"""
        self.element_melting_temperatures = np.array(
            [self.element_data.loc[e, "Elemental melting temperature"] for e in self.element_symbols])
        self.melting_temperature = np.sum(self.element_fractions * self.element_melting_temperatures)
        return self.melting_temperature

    def get_entropy_of_mixing(self):
        """Entropy of mixing"""
        self.entropy_of_mixing = - 8.314 * np.sum(self.element_fractions * np.log(self.element_fractions))
        return self.entropy_of_mixing

    def get_mixing_enthalpy(self):
        """Mixing enthalpy"""
        pair_indices = [c for c in combinations(list(range(len(self.element_fractions))), 2)]
        self.mixing_enthalpy = 0
        for (i, j) in pair_indices:
            if self.element_symbols[i] != self.element_symbols[j]:
                try:
                    mixing_value = self.element_data.loc[self.element_symbols[i], self.element_symbols[j]]
                except:
                    try:
                        mixing_value = self.element_data.loc[self.element_symbols[j], self.element_symbols[i]]
                    except:
                        mixing_value = 0
                self.mixing_enthalpy = self.mixing_enthalpy + 4 * self.element_fractions[i] * self.element_fractions[
                    j] * mixing_value
        return self.mixing_enthalpy

    def get_phase_formation_coefficient(self):
        self.phase_formation_coefficient = self.entropy_of_mixing / (self.atomic_size_mismatch ** 2)

    def get_work_function(self):
        self.element_work_functions = np.array(
            [self.element_data.loc[e, "Elemental work function"] for e in self.element_symbols])
        self.work_function = np.sum(self.element_fractions * self.element_work_functions)
        return self.work_function

    def get_dr(self):
        # Local size mismatch in n-element alloys
        pair_indices = [c for c in combinations(list(range(len(self.element_fractions))), 2)]
        self.local_size_mismatch = 0
        for (i, j) in pair_indices:
            self.local_size_mismatch = self.local_size_mismatch + self.element_fractions[i] * self.element_fractions[
                j] * abs(self.atomic_radii[i] - self.atomic_radii[j])
        # Local modulus mismatch
        pair_indices = [c for c in combinations(list(range(len(self.element_fractions))), 2)]
        self.local_modulus_mismatch = 0
        for (i, j) in pair_indices:
            self.local_modulus_mismatch = self.local_modulus_mismatch + self.element_fractions[i] * \
                                          self.element_fractions[j] * abs(
                self.element_shear_moduli[i] - self.element_shear_moduli[j])
        return self.local_size_mismatch

    def get_energy_term(self):
        """Energy term"""
        self.energy_term = self.shear_modulus * self.atomic_size_mismatch * (1 + self.lattice_distortion_energy) * (
                    1 - self.lattice_distortion_energy)
        return self.energy_term

    def get_f(self):
        """
        Peierls-Nabarro stress coefficient
        """
        self.peierls_nabarro_coefficient = 2 * self.shear_modulus / (1 - self.lattice_distortion_energy)

    def get_fai(self):
        # Solid solution phase forming ability
        if self.mixing_enthalpy != 0:
            # the units of mixing_enthalpy is kJ and entropy_of_mixing is J
            self.solid_solution_forming_ability = self.melting_temperature * self.entropy_of_mixing / abs(
                self.mixing_enthalpy) / 1000
        else:
            self.solid_solution_forming_ability = 0
        # Phase formation coefficient
        self.phase_formation_coefficient = self.entropy_of_mixing / (max(self.atomic_size_mismatch ** 2, 0.0001))

    def get_local_atomic_distortion(self):
        """
        from one single atom alone
        """
        self.local_atomic_distortion_single = np.sum(
            self.element_fractions * abs(self.atomic_radii - self.average_atomic_radius) / self.average_atomic_radius)
        return self.local_atomic_distortion_single

    def get_local_atomic_distortion2(self):
        pair_indices = [c for c in combinations(list(range(len(self.element_fractions))), 2)]
        self.local_atomic_distortion_group = 0
        for (i, j) in pair_indices:
            self.local_atomic_distortion_group = self.local_modulus_mismatch + self.element_fractions[i] * \
                                                 self.element_fractions[j] * abs(
                self.atomic_radii[i] + self.atomic_radii[j] - 2 * self.average_atomic_radius) / (
                                                         2 * self.average_atomic_radius)
        return self.local_atomic_distortion_group

    def get_pauling_electronegativity(self):
        """Pauling electronegativity"""
        self.element_pauling_electronegativities = np.array(
            [self.element_data.loc[e, "Electronegativity Pauling"] for e in self.element_symbols])
        self.average_pauling_electronegativity = np.sum(
            self.element_fractions * self.element_pauling_electronegativities)
        self.pauling_electronegativity_deviation = (np.sum(self.element_fractions * (
                    self.element_pauling_electronegativities - self.average_pauling_electronegativity) ** 2)) ** 0.5
        return self.pauling_electronegativity_deviation

    def get_allen_electronegativity(self):
        self.element_allen_electronegativities = np.array(
            [self.element_data.loc[e, "Electronegativity Allred"] for e in self.element_symbols])
        self.average_allen_electronegativity = np.sum(self.element_fractions * self.element_allen_electronegativities)
        self.allen_electronegativity = np.sum(self.element_fractions * abs(
            1 - self.element_allen_electronegativities / self.average_allen_electronegativity) ** 2)
        return self.allen_electronegativity

    def get_ed(self):
        """ Elemental electron density"""
        self.element_electron_densities = np.array(
            [self.element_data.loc[e, "Elemental electron density"] for e in self.element_symbols])
        self.electron_density = np.sum(self.element_fractions * self.element_electron_densities)
        return self.electron_density

    def get_debye_related_features(self):
        """
        Estimate Debye-related features using only constants, atomic radii, and already-computed properties.
        This yields proxy features for ranking/correlation purposes.
        """
        # Estimated atomic volume per atom from average atomic radius (sphere approximation)
        self.estimated_atomic_volume = (4.0 / 3.0) * np.pi * (self.average_atomic_radius ** 3)
        # Estimated atomic number density (atoms per m^3 in relative units)
        self.estimated_number_density = 1.0 / max(self.estimated_atomic_volume, 1e-30)
        # Debye wavevector proxy k_D = (6*pi^2*n)^(1/3)
        self.debye_wavevector_proxy = (6.0 * (np.pi ** 2) * self.estimated_number_density) ** (1.0 / 3.0)
        # Mean sound velocity proxy using Young's modulus and electron density as a mass-density surrogate
        self.mean_sound_velocity_proxy = np.sqrt(max(self.youngs_modulus, 0.0) / max(self.electron_density, 1e-30))
        # Debye temperature proxy
        self.debye_temperature_proxy = (self.planck_constant / self.boltzmann_constant) * self.debye_wavevector_proxy * self.mean_sound_velocity_proxy


if __name__ == '__main__':
    # d = find_elements("ZrCu")
    # print(d)
    # d = find_elements("AlCrFeNiMo0.5")
    # print(d)
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
    dataset = pd.read_csv("../data/formula_Ga-In-Sn-Bi.csv")
    df1 = formula_to_features(dataset['formula'])
    print(df1)
    df1.to_csv("../data/alloy_features.csv", index=False)

    # designed alloys knowledge-aware feature calculation
    # designed_dataset = pd.read_csv("../data/formula_design.csv")
    # designed_feature = formula_to_features(designed_dataset['formula'])
    # f_name = "alloy"
    # designed_feature.to_csv(f"../data/formula_design_{f_name}_features.csv", index=False)
