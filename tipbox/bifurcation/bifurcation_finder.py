"""
Bifurcation detection using AUTO-07P for proper detection of fold, hopf, AND transcritical bifurcations.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
import os
import sys
import re
import io
from contextlib import redirect_stdout, nullcontext

from .polynomial_model import PolynomialModel
from .auto.convert_bifdata import convert_bifdata

# Setup AUTO-07P (required for bifurcation detection)
AUTO_DIR = os.environ.get('AUTO_DIR')
if not AUTO_DIR:
    raise EnvironmentError(
        "AUTO_DIR environment variable not set. "
        "Please install AUTO-07P and set AUTO_DIR. See docs/INSTALL_AUTO.md"
    )

auto_python = os.path.join(AUTO_DIR, 'python')
if auto_python not in sys.path:
    sys.path.insert(0, auto_python)

try:
    import auto
except ImportError as e:
    raise ImportError(
        f"AUTO-07P Python interface not found at {auto_python}. "
        f"Please install AUTO-07P. See docs/INSTALL_AUTO.md. Error: {e}"
    )


@dataclass
class BifurcationPoint:
    """Information about a detected bifurcation or null parameter"""
    bif_type: str  # 'fold' (LP), 'hopf' (HB), 'transcritical' (BP), or 'null' (NA)
    bif_param_name: str  # which parameter (e.g., 'a1', 'b3')
    bif_param_index: int  # index in parameter array (0-19) - use to get from model
    bif_value: float  # parameter value at bifurcation (0 for null)


def find_all_bifurcations(
    model: PolynomialModel,
    param_range: Tuple[float, float] = (-5.0, 5.0),
    verbose: bool = True,
) -> List[BifurcationPoint]:
    """
    Find bifurcations using AUTO-07P for all nonzero parameters.

    Following Bury et al. (2021) PNAS paper approach.
    Returns BifurcationPoint for every nonzero parameter:
    - bif_type='fold'/'hopf'/'transcritical' if bifurcation found
    - bif_type='null' if no bifurcation in param_range

    Parameters
    ----------
    model : PolynomialModel
    param_range : tuple
        Parameter range for continuation. Default (-5, 5) as in PNAS paper.
    verbose : bool
        If True (default), print AUTO-07P output. Set to False to suppress output.

    Returns
    -------
    list of BifurcationPoint objects for all nonzero parameters

    Raises
    ------
    ImportError if AUTO-07P not installed (checked at module import)
    """
    # Create temporary directory for AUTO files
    temp_dir = Path(tempfile.mkdtemp(prefix='auto_'))
    original_dir = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        
        # Create AUTO files
        _create_fortran_file()
        _create_constants_file(param_range)
        
        # Prepare parameters for AUTO
        par_labels = [f'a{i}' for i in range(1, 11)] + [f'b{i}' for i in range(1, 11)]
        pars_array = np.concatenate([np.array(model.pars_a), np.array(model.pars_b)])
        par_indices = list(range(1, 11)) + list(range(12, 22))
        p = dict(zip(par_indices, pars_array))
        u = np.array(model.equi)
        
        # Parameter labels
        nonzero_idx = np.where(np.abs(pars_array) > 1e-6)[0]
        nonzero_params = [par_labels[i] for i in nonzero_idx]
        
        results = []

        # Context manager for suppressing AUTO-07P stdout when verbose=False
        # Only wraps auto.run() - the main source of verbose output
        devnull = io.StringIO()
        output_ctx = nullcontext() if verbose else redirect_stdout(devnull)

        # Run continuation for each nonzero parameter
        for par in nonzero_params:
            try:
                model_auto = auto.load('model')
                model_auto = auto.load(u, PAR=p, ICP=par, UZSTOP={par: list(param_range)})
                with output_ctx:
                    auto_result = auto.run(model_auto)
                auto.save(auto_result, f'out{par}')
                auto.clean()

                # Parse results (PNAS approach)
                out = convert_bifdata(f'b.out{par}')

                # Map AUTO type codes to our names
                type_map = {'LP': 'fold', 'HB': 'hopf', 'BP': 'transcritical', 'NA': 'null'}

                # Create BifurcationPoint object
                bif = BifurcationPoint(
                    bif_type=type_map.get(out['type'], 'null'),
                    bif_param_name=out['bif_param'],
                    bif_param_index=par_labels.index(par),
                    bif_value=float(out['value'])
                )
                results.append(bif)

            except Exception:
                # Silently continue if parameter fails
                try:
                    auto.clean()
                except:
                    pass

        return results
        
    finally:
        # Return to original directory and cleanup
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_template_path(filename: str) -> Path:
    """Get path to AUTO template file"""
    module_dir = Path(__file__).parent
    template_path = module_dir / 'auto' / filename
    
    if not template_path.exists():
        raise FileNotFoundError(f"AUTO template not found: {template_path}")
    
    return template_path


def _create_fortran_file():
    """Copy model.f90 template from auto/ directory"""
    template = _get_template_path('model.f90')
    shutil.copy(template, 'model.f90')


def _create_constants_file(param_range: Tuple[float, float]):
    """Create c.model from template with updated parameter range"""
    template = _get_template_path('c.model')
    
    with open(template, 'r') as f:
        contents = f.read()
    
    # Update parameter range
    rl0, rl1 = param_range
    contents = re.sub(r'RL0=\s*[-+]?\d+\.?\d*', f'RL0= {rl0}', contents)
    contents = re.sub(r'RL1=\s*[-+]?\d+\.?\d*', f'RL1= {rl1}', contents)
    
    with open('c.model', 'w') as f:
        f.write(contents)


def select_bifurcation(
    bifurcations: List[BifurcationPoint],
    bifurcation_type: str
) -> Optional[BifurcationPoint]:
    """
    Select a bifurcation from the list based on type preference.
    
    Parameters
    ----------
    bifurcations : list of BifurcationPoint
        All detected bifurcations
    bifurcation_type : str
        Supported type: 'fold', 'hopf', 'transcritical'
    
    Returns
    -------
    BifurcationPoint or None
        First matching bifurcation, or None if no match found
    """
    if not bifurcations:
        return None
    
    # Return first bifurcation of requested type
    for bif in bifurcations:
        if bif.bif_type == bifurcation_type:
            return bif
    
    return None
