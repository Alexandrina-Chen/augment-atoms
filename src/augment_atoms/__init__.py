"""
augment-atoms: Dataset augmentation for atomistic machine learning

This module implements a GPU-accelerated rattle-relax-repeat procedure for
augmenting datasets of atomic configurations using a PES model.

Enhanced with energy deviation filtering to control the energy range of
generated child structures relative to their parents.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.io import read, write

__version__ = "0.2.0"
__all__ = [
    "AugmentConfig",
    "EnergyDeviationConfig",
    "augment_atoms",
    "augment_single_structure",
    "main",
]

logger = logging.getLogger(__name__)

# Boltzmann constant in eV/K
kB = 8.617333262e-5


# =============================================================================
# Energy Deviation Filter
# =============================================================================

@dataclass
class EnergyDeviationConfig:
    """
    Configuration for energy deviation filtering of child structures.
    
    The filter accepts a child structure if its energy satisfies:
        lower_bound <= (E_child - E_parent) <= upper_bound
    
    Parameters
    ----------
    mode : str
        One of 'absolute', 'per_atom', or 'relative'
        - 'absolute': Direct energy difference in configured units (e.g., eV)
        - 'per_atom': Energy difference per atom (e.g., eV/atom)
        - 'relative': Fractional deviation from parent energy (e.g., 0.1 = 10%)
    lower_bound : float
        Lower bound for energy deviation (can be negative for lower energies)
    upper_bound : float
        Upper bound for energy deviation (typically positive)
    
    Examples
    --------
    # Accept children within ±0.5 eV of parent (absolute)
    >>> config = EnergyDeviationConfig(mode='absolute', lower_bound=-0.5, upper_bound=0.5)
    
    # Accept children within ±0.1 eV/atom of parent
    >>> config = EnergyDeviationConfig(mode='per_atom', lower_bound=-0.1, upper_bound=0.1)
    
    # Accept children within -10% to +50% of parent energy
    >>> config = EnergyDeviationConfig(mode='relative', lower_bound=-0.10, upper_bound=0.50)
    """
    mode: str = 'per_atom'
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    
    def __post_init__(self):
        valid_modes = ('absolute', 'per_atom', 'relative')
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"lower_bound ({self.lower_bound}) cannot be greater than "
                f"upper_bound ({self.upper_bound})"
            )


class EnergyDeviationFilter:
    """
    Filter child structures based on their energy deviation from parent.
    
    This filter is applied after relaxation to reject structures with
    energies that deviate too much from their parent.
    """
    
    def __init__(self, config: EnergyDeviationConfig):
        self.config = config
        self._stats = {
            'accepted': 0,
            'rejected': 0,
            'too_low': 0,
            'too_high': 0
        }
    
    def compute_deviation(
        self, 
        parent_energy: float, 
        child_energy: float, 
        n_atoms: int
    ) -> float:
        """Compute energy deviation based on configured mode."""
        delta_E = child_energy - parent_energy
        
        if self.config.mode == 'absolute':
            return delta_E
        elif self.config.mode == 'per_atom':
            return delta_E / n_atoms
        elif self.config.mode == 'relative':
            if abs(parent_energy) < 1e-10:
                return np.sign(delta_E) * np.inf if delta_E != 0 else 0.0
            return delta_E / abs(parent_energy)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
    
    def accept(
        self, 
        parent_energy: float, 
        child_energy: float, 
        n_atoms: int
    ) -> bool:
        """
        Determine whether to accept a child structure.
        
        Returns True if the energy deviation is within bounds.
        """
        deviation = self.compute_deviation(parent_energy, child_energy, n_atoms)
        
        if deviation < self.config.lower_bound:
            self._stats['rejected'] += 1
            self._stats['too_low'] += 1
            return False
        
        if deviation > self.config.upper_bound:
            self._stats['rejected'] += 1
            self._stats['too_high'] += 1
            return False
        
        self._stats['accepted'] += 1
        return True
    
    def accept_atoms(self, parent: Atoms, child: Atoms) -> bool:
        """Convenience method that works directly with ASE Atoms objects."""
        return self.accept(
            parent.get_potential_energy(),
            child.get_potential_energy(),
            len(child)
        )
    
    def get_statistics(self) -> dict:
        """Get acceptance/rejection statistics."""
        total = self._stats['accepted'] + self._stats['rejected']
        return {
            **self._stats,
            'total': total,
            'acceptance_rate': self._stats['accepted'] / total if total > 0 else 0.0
        }
    
    def reset_statistics(self):
        """Reset all counters."""
        for key in self._stats:
            self._stats[key] = 0


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class AugmentConfig:
    """
    Configuration for the augmentation procedure.
    
    Parameters
    ----------
    n_per_structure : int
        Number of augmentations per starting structure
    T : float
        Temperature in Kelvin for Boltzmann weighting
    beta : float
        Explore-vs-exploit trade-off (0 = explore, 1 = exploit)
    sigma_range : Tuple[float, float]
        Range of standard deviations for rattling (in Å)
    seed : int
        Random seed for reproducibility
    cell_sigma : Optional[float]
        Standard deviation for cell perturbation (in Å), None for no perturbation
    units : str
        Energy units (default: 'eV')
    max_force : float
        Maximum force magnitude to relax to (in energy/Å)
    min_separation : float
        Minimum allowed separation between atoms (in Å)
    max_relax_steps : int
        Maximum number of relaxation steps per iteration
    similarity_threshold : float
        Threshold for considering structures too similar (in Å)
    energy_deviation : Optional[EnergyDeviationConfig]
        Configuration for energy deviation filtering (None to disable)
    max_filter_attempts : int
        Maximum attempts to generate an acceptable child when filtering
    """
    n_per_structure: int = 10
    T: float = 300.0
    beta: float = 0.5
    sigma_range: Tuple[float, float] = (0.01, 0.1)
    seed: int = 42
    cell_sigma: Optional[float] = None
    units: str = 'eV'
    max_force: float = 30.0
    min_separation: float = 0.5
    max_relax_steps: int = 20
    similarity_threshold: float = 0.1
    energy_deviation: Optional[EnergyDeviationConfig] = None
    max_filter_attempts: int = 10
    
    def __post_init__(self):
        if len(self.sigma_range) != 2:
            raise ValueError("sigma_range must be a tuple of (min, max)")
        if self.sigma_range[0] > self.sigma_range[1]:
            raise ValueError("sigma_range[0] must be <= sigma_range[1]")
        if self.beta < 0 or self.beta > 1:
            raise ValueError("beta must be in [0, 1]")


# =============================================================================
# Core Functions
# =============================================================================

def get_minimum_distance(atoms: Atoms) -> float:
    """Get the minimum distance between any pair of atoms."""
    positions = atoms.get_positions()
    n = len(atoms)
    if n < 2:
        return np.inf
    
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist


def get_rmsd(atoms1: Atoms, atoms2: Atoms) -> float:
    """Compute RMSD between two structures (assuming same ordering)."""
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    return np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))


def select_parent(
    tree: List[Atoms],
    generations: List[int],
    config: AugmentConfig,
    rng: np.random.Generator
) -> Tuple[Atoms, int]:
    """
    Select a parent structure from the tree.
    
    Probability is given by:
        P_i = beta * (exp(-E_i / kT) / Z) + (1-beta) * (G_i / sum(G))
    
    where E_i is energy, G_i is generation, T is temperature.
    """
    n = len(tree)
    energies = np.array([atoms.get_potential_energy() for atoms in tree])
    gens = np.array(generations, dtype=float)
    
    # Boltzmann weights (exploit term)
    kT = kB * config.T
    boltz = np.exp(-energies / kT)
    boltz /= boltz.sum()
    
    # Generation weights (explore term)
    gen_weights = gens / gens.sum() if gens.sum() > 0 else np.ones(n) / n
    
    # Combined probability
    probs = config.beta * boltz + (1 - config.beta) * gen_weights
    probs /= probs.sum()  # Normalize
    
    idx = rng.choice(n, p=probs)
    return tree[idx], idx


def rattle(
    parent: Atoms,
    seed_cell: np.ndarray,
    config: AugmentConfig,
    rng: np.random.Generator
) -> Atoms:
    """
    Rattle atomic positions and optionally the unit cell.
    
    Transformation:
        R' = [(A + I) × R] + B
        C' = (A + I) × C_0
    
    where A is a random matrix, B is random displacements.
    """
    child = parent.copy()
    n_atoms = len(child)
    
    # Sample sigma values
    sigma_A = rng.uniform(*config.sigma_range)
    sigma_B = rng.uniform(0, config.cell_sigma) if config.cell_sigma else 0.0
    
    positions = child.get_positions()
    
    if config.cell_sigma is not None and any(child.pbc):
        # Periodic system: apply cell perturbation
        A = rng.normal(0, sigma_A, size=(3, 3))
        transform = A + np.eye(3)
        
        # Transform positions
        new_positions = positions @ transform.T
        
        # Add random displacements
        B = rng.normal(0, sigma_B, size=(n_atoms, 3))
        new_positions += B
        
        # Transform cell
        new_cell = seed_cell @ transform.T
        child.set_cell(new_cell, scale_atoms=False)
        child.set_positions(new_positions)
    else:
        # Isolated system: only rattle positions
        B = rng.normal(0, sigma_A, size=(n_atoms, 3))
        child.set_positions(positions + B)
    
    return child


def relax(
    child: Atoms,
    parent_energy: float,
    config: AugmentConfig,
    rng: np.random.Generator
) -> Optional[Atoms]:
    """
    Relax the child structure using a Robbins-Monro inspired scheme.
    
    Update rule at step x:
        R' = R + (sigma_B / x) * (F / ||F||)
    
    Returns None if relaxation fails or structure is invalid.
    """
    sigma_B = rng.uniform(*config.sigma_range)
    
    for step in range(1, config.max_relax_steps + 1):
        try:
            forces = child.get_forces()
            energy = child.get_potential_energy()
        except Exception as e:
            logger.debug(f"Calculator failed at step {step}: {e}")
            return None
        
        # Check maximum force
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        # Normalize forces and update positions
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms > 1e-10, force_norms, 1.0)
        force_dirs = forces / force_norms
        
        step_size = sigma_B / step
        new_positions = child.get_positions() + step_size * force_dirs
        child.set_positions(new_positions)
        
        # Early stopping with probability based on energy change
        if max_force < config.max_force:
            delta_E = energy - parent_energy
            kT = kB * config.T
            stop_prob = min(0.25, np.exp(-delta_E / kT) if delta_E > 0 else 0.25)
            if rng.random() < stop_prob:
                break
    
    # Check minimum separation
    if get_minimum_distance(child) < config.min_separation:
        return None
    
    return child


def is_too_similar(
    child: Atoms,
    tree: List[Atoms],
    config: AugmentConfig
) -> bool:
    """Check if child is too similar to any structure in the tree."""
    for existing in tree:
        if get_rmsd(child, existing) < config.similarity_threshold:
            return True
    return False


def augment_single_structure(
    seed: Atoms,
    calculator: Calculator,
    config: AugmentConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Atoms]:
    """
    Augment a single seed structure to create a family tree.
    
    Parameters
    ----------
    seed : Atoms
        The starting structure
    calculator : Calculator
        ASE calculator for energy/force evaluation
    config : AugmentConfig
        Augmentation configuration
    progress_callback : Callable, optional
        Called with (current, total) for progress updates
    
    Returns
    -------
    List[Atoms]
        The complete family tree including the seed
    """
    rng = np.random.default_rng(config.seed)
    
    # Initialize
    seed = seed.copy()
    seed.calc = calculator
    seed_energy = seed.get_potential_energy()
    seed_cell = seed.get_cell().array.copy()
    
    # Store energy and generation in atoms.info
    seed.info['energy'] = seed_energy
    seed.info['generation'] = 0
    
    tree = [seed]
    generations = [0]
    
    # Initialize energy filter if configured
    energy_filter = None
    if config.energy_deviation is not None:
        energy_filter = EnergyDeviationFilter(config.energy_deviation)
    
    # Generate children
    for i in range(config.n_per_structure):
        if progress_callback:
            progress_callback(i + 1, config.n_per_structure)
        
        # Try to find an acceptable child
        max_attempts = config.max_filter_attempts if energy_filter else 1
        child_accepted = False
        
        for attempt in range(max_attempts):
            # Select parent
            parent, parent_idx = select_parent(tree, generations, config, rng)
            parent_energy = parent.get_potential_energy()
            parent_gen = generations[parent_idx]
            
            # Rattle
            child = rattle(parent, seed_cell, config, rng)
            child.calc = calculator
            
            # Relax
            child = relax(child, parent_energy, config, rng)
            if child is None:
                continue
            
            # Check similarity
            if is_too_similar(child, tree, config):
                continue
            
            # Check energy deviation filter
            if energy_filter is not None:
                if not energy_filter.accept_atoms(parent, child):
                    continue
            
            # Child accepted!
            child_energy = child.get_potential_energy()
            child.info['energy'] = child_energy
            child.info['generation'] = parent_gen + 1
            child.info['parent_index'] = parent_idx
            
            tree.append(child)
            generations.append(parent_gen + 1)
            child_accepted = True
            break
        
        if not child_accepted:
            logger.warning(
                f"Could not generate acceptable child {i+1} after "
                f"{max_attempts} attempts"
            )
    
    # Log filter statistics
    if energy_filter is not None:
        stats = energy_filter.get_statistics()
        logger.info(
            f"Energy filter: accepted {stats['accepted']}, "
            f"rejected {stats['rejected']} "
            f"(too_low: {stats['too_low']}, too_high: {stats['too_high']})"
        )
    
    return tree


def augment_atoms(
    input_structures: List[Atoms],
    calculator: Calculator,
    config: AugmentConfig,
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None
) -> List[Atoms]:
    """
    Augment multiple structures.
    
    Parameters
    ----------
    input_structures : List[Atoms]
        List of seed structures
    calculator : Calculator
        ASE calculator for energy/force evaluation
    config : AugmentConfig
        Augmentation configuration
    progress_callback : Callable, optional
        Called with (struct_idx, n_structs, child_idx, n_children)
    
    Returns
    -------
    List[Atoms]
        All augmented structures (including seeds)
    """
    all_structures = []
    n_structures = len(input_structures)
    
    for struct_idx, seed in enumerate(input_structures):
        logger.info(f"Augmenting structure {struct_idx + 1}/{n_structures}")
        
        # Create per-structure progress callback
        if progress_callback:
            def struct_progress(child_idx, n_children):
                progress_callback(struct_idx, n_structures, child_idx, n_children)
        else:
            struct_progress = None
        
        # Use different seed for each structure
        struct_config = AugmentConfig(
            **{**config.__dict__, 'seed': config.seed + struct_idx}
        )
        
        tree = augment_single_structure(
            seed, calculator, struct_config, struct_progress
        )
        all_structures.extend(tree)
    
    return all_structures


# =============================================================================
# Calculator Loading
# =============================================================================

def lennard_jones() -> Calculator:
    """Create a Lennard-Jones calculator."""
    return LennardJones()


def graph_pes_calculator(path: str) -> Calculator:
    """
    Load a graph-pes model as an ASE calculator.
    
    Requires the graph-pes package to be installed.
    """
    try:
        from graph_pes.interfaces.ase import GraphPESCalculator
        return GraphPESCalculator(path)
    except ImportError:
        raise ImportError(
            "graph-pes is required for graph_pes_calculator. "
            "Install with: pip install graph-pes"
        )


def load_calculator(config: dict) -> Calculator:
    """
    Load a calculator from configuration.
    
    Supports:
    - +lennard_jones()
    - +graph_pes_calculator: {path: ...}
    - +module.function()
    """
    calc_config = config.get('calculator', '+lennard_jones()')
    
    if isinstance(calc_config, str):
        if calc_config == '+lennard_jones()':
            return lennard_jones()
        elif calc_config.startswith('+') and calc_config.endswith('()'):
            # Custom function call: +module.function()
            func_path = calc_config[1:-2]  # Remove + and ()
            module_path, func_name = func_path.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func()
    elif isinstance(calc_config, dict):
        if '+graph_pes_calculator' in calc_config:
            path = calc_config['+graph_pes_calculator']['path']
            return graph_pes_calculator(path)
    
    raise ValueError(f"Unknown calculator configuration: {calc_config}")


# =============================================================================
# Configuration Loading
# =============================================================================

def load_energy_deviation_config(config_dict: dict) -> Optional[EnergyDeviationConfig]:
    """Parse energy deviation configuration from YAML."""
    if 'energy_deviation' not in config_dict:
        return None
    
    ed = config_dict['energy_deviation']
    
    # Shorthand: [lower, upper] assumes per_atom mode
    if isinstance(ed, (list, tuple)) and len(ed) == 2:
        return EnergyDeviationConfig(
            mode='per_atom',
            lower_bound=ed[0],
            upper_bound=ed[1]
        )
    
    return EnergyDeviationConfig(
        mode=ed.get('mode', 'per_atom'),
        lower_bound=ed.get('lower_bound', -np.inf),
        upper_bound=ed.get('upper_bound', np.inf)
    )


def load_config(yaml_path: str) -> Tuple[AugmentConfig, dict]:
    """
    Load configuration from a YAML file.
    
    Returns
    -------
    Tuple[AugmentConfig, dict]
        The augmentation config and the full YAML dict
    """
    with open(yaml_path) as f:
        full_config = yaml.safe_load(f)
    
    cfg = full_config.get('config', {})
    
    # Parse energy deviation
    energy_deviation = load_energy_deviation_config(cfg)
    
    # Build AugmentConfig
    augment_config = AugmentConfig(
        n_per_structure=cfg.get('n_per_structure', 10),
        T=cfg.get('T', 300.0),
        beta=cfg.get('beta', 0.5),
        sigma_range=tuple(cfg.get('sigma_range', [0.01, 0.1])),
        seed=cfg.get('seed', 42),
        cell_sigma=cfg.get('cell_sigma', None),
        units=cfg.get('units', 'eV'),
        max_force=cfg.get('max_force', 30.0),
        min_separation=cfg.get('min_separation', 0.5),
        max_relax_steps=cfg.get('max_relax_steps', 20),
        similarity_threshold=cfg.get('similarity_threshold', 0.1),
        energy_deviation=energy_deviation,
        max_filter_attempts=cfg.get('max_filter_attempts', 10)
    )
    
    return augment_config, full_config


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command line entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) != 2:
        print("Usage: augment-atoms config.yaml")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    
    # Load configuration
    logger.info(f"Loading configuration from {yaml_path}")
    config, full_config = load_config(yaml_path)
    
    # Load input structures
    input_path = full_config['data']['input']
    output_path = full_config['data']['output']
    
    logger.info(f"Loading structures from {input_path}")
    structures = read(input_path, index=':')
    if isinstance(structures, Atoms):
        structures = [structures]
    logger.info(f"Loaded {len(structures)} structures")
    
    # Load calculator
    logger.info("Loading calculator")
    calculator = load_calculator(full_config.get('model', {}))
    
    # Log energy filter configuration
    if config.energy_deviation is not None:
        ed = config.energy_deviation
        logger.info(
            f"Energy filter enabled: mode={ed.mode}, "
            f"bounds=[{ed.lower_bound}, {ed.upper_bound}]"
        )
    
    # Run augmentation
    logger.info("Starting augmentation")
    augmented = augment_atoms(structures, calculator, config)
    
    logger.info(f"Generated {len(augmented)} total structures")
    
    # Save results
    logger.info(f"Saving to {output_path}")
    write(output_path, augmented)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()