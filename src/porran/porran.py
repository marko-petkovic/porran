"""High-level API for graph-based substitutions and defect generation."""

import os
from time import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from numpy import ndarray
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

from .create_structure import create_dmof, create_zeo, create_defect_mof
from .get_zeolite import get_zeolite
from .graph_creation import radius_graph, zeo_graph, mof_graph
from .mask_method import (
    mask_all,
    mask_array,
    mask_box,
    mask_combination,
    mask_h_on_c,
    mask_species,
    mask_zeo,
)
from .replacement_algorithms import (
    multi_clusters,
    chains, 
    clusters, 
    maximize_entropy, 
    random, 
    random_lowenstein,
    lowenstein,
)
from .utils import is_atom, normalize_supercell, write_cif


class PORRAN:
    """Main user-facing interface for structure loading and generation workflows."""

    def __init__(
        self,
        cif_path: Optional[str] = None,
        graph_method: Optional[Union[str, Callable]] = None,
        mask_method: Optional[Union[List[str], ndarray, str]] = None,
        seed: Optional[int] = None,
        download_path: Optional[str] = "downloads",
        *args,
        **kwargs,
    ):
        
        self.cif_path = None
        self.download_path = download_path
        self.mask_method_input = None
        self.graph_args = ()
        self.graph_kwargs = {}

        if cif_path is not None:
            self.init_structure(cif_path, graph_method, mask_method, download_path=download_path, *args, **kwargs)
        if seed is not None:
            self.set_seed(seed)

    def _store_generation_inputs(self, mask_method, args, kwargs):
        """Persist the current mask and graph arguments for later reuse."""
        self.mask_method_input = mask_method
        self.graph_args = args
        self.graph_kwargs = dict(kwargs)

    def init_structure(
        self,
        cif_path: str,
        graph_method: Optional[Union[str, Callable]],
        mask_method: Optional[Union[List[str], ndarray, str]] = None,
        check_cif: bool = False, site_tolerance: float = 1e-3,
        download_path: Optional[str] = "downloads",
        *args,
        **kwargs,
    ):
        """
        Initialize the structure and the method to build the graph

        Parameters
        ----------
        cif_path : str
            Path to the cif file
        graph_method : Optional[Union[str, Callable]]
            Method to build the graph. If str, it can be 'zeolite' or 'radius'
        mask_method : Optional[Union[List[str], ndarray, str]]
            Method to select atoms to include in the graph.
            To directly select atoms, its possible to provide an np.array with the indices of the atoms to include set to 1
            To select atoms by species, provide a list of species to include
        check_cif : bool, optional
            Check the cif file for errors, default is False
        site_tolerance : float, optional
            Tolerance for site matching, default is 1e-3
        download_path: Optional[str] = "downloads"
            Path to download MOF nodes and linkers files
    
        Returns
        -------
        None
        """
        # name is the name of the cif file
        self.name = os.path.splitext(os.path.basename(cif_path))[0]
        self.cif_path = cif_path
        self.download_path = download_path
        self._store_generation_inputs(mask_method, args, kwargs)
        self.structure = self._read_structure(cif_path, check_cif)
        self.graph_method = self._get_graph_method(graph_method)
        self.mask_method = self._get_mask_method(mask_method)
        self.mask = self.mask_method(self.structure, mask_method, *args, **kwargs) # type: ignore
        self.structure_graph = self.graph_method(self.structure, mask=self.mask, download_path=download_path, cif_path=cif_path, *args, **kwargs) # type: ignore

    def from_IZA_code(
        self,
        zeolite_code: str,
        graph_method: Optional[Union[str, Callable]] = None,
        mask_method: Optional[Union[List[str], ndarray, str]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the structure from an IZA code

        Parameters
        ----------
        zeolite_code : str
            IZA code of the zeolite
        graph_method : Optional[Union[str, Callable]] = None,
            Method to build the graph. If str, it can be 'zeolite' or 'radius'
        mask_method : Optional[Union[List[str], ndarray, str]] = None,
            Method to select atoms to include in the graph.
            To directly select atoms, its possible to provide an np.array with the indices of the atoms to include set to 1
            To select atoms by species, provide a list of species to include

        Returns
        -------
        None
        """
        self.name = zeolite_code
        self.cif_path = None
        self._store_generation_inputs(mask_method, args, kwargs)
        self.structure = get_zeolite(zeolite_code)
        self.graph_method = self._get_graph_method(graph_method)
        self.mask_method = self._get_mask_method(mask_method)
        self.mask = self.mask_method(self.structure, mask_method, *args, **kwargs) # type: ignore
        self.structure_graph = self.graph_method(self.structure, mask=self.mask, *args, **kwargs) # type: ignore

    def change_graph_method(
        self,
        graph_method: Optional[Union[str, Callable]] = None,
        mask_method: Optional[Union[List[str], ndarray, str]] = None,
        *args,
        **kwargs,
    ):
        """
        Change the method to build the graph

        Parameters
        ----------
        graph_method : Optional[Union[str, Callable]] = None,
            Method to build the graph. If str, it can be 'zeolite' or 'radius'
        mask_method : Optional[Union[List[str], ndarray, str]]
            Method to select atoms to include in the graph.
            To directly select atoms, its possible to provide an np.array with the indices of the atoms to include set to 1
            To select atoms by species, provide a list of species to include
        Returns
        -------
        None
        """
        self.graph_method = self._get_graph_method(graph_method)
        self.mask_method = self._get_mask_method(mask_method)
        self._store_generation_inputs(mask_method, args, kwargs)
        self.mask = self.mask_method(self.structure, mask_method, *args, **kwargs) # type: ignore
        self.structure_graph = self.graph_method(self.structure, mask=self.mask, *args, **kwargs) # type: ignore

    def generate_structures(
        self,
        n_structures: int,
        replace_algo: Union[str, Callable],
        create_algo: Union[str, Callable],
        n_subs: int,
        max_tries: int = 100,
        post_algo: Optional[Callable] = None,
        write: bool = False, overwrite_ok = False,
        writepath: Optional[str] = "structures",
        verbose: bool = True,
        print_error : bool = False,
        struc_name: Optional[str] = None,
        custom_charges: Optional[Dict[str, float]] = None,
        modify_O_connected_to_Al: bool = False,
        modify_O_connected_to_Al_Al: bool = False,
        supercell: Union[int, List[int], tuple] = (1, 1, 1),
        *args,
        **kwargs,
    ) -> List[Structure]:
        """
        Generate structures by replacing nodes in the graph

        Parameters
        ----------
        n_structures : int
            Number of structures to generate
        replace_algo : Union[str, Callable]
            Algorithm to select nodes to replace. If str, it can be 'random', 'random_lowenstein', 'lowenstein', 'clusters', 'multi_clusters','chains' or 'maximize_entropy'
        create_algo : Union[str, Callable]
            Algorithm to create the new structure. If str, it can be 'zeolite'
        n_subs : int
            Number of nodes to replace
        max_tries : int, optional
            Maximum number of tries to replace nodes, default is 100
        post_algo : Callable, optional
            Post processing algorithm to apply to the new structure
        write : bool, optional
            Write the structures to a file, default is False
        writepath : str, optional
            Path to write the structures to, default is None
            If writepath is not specified, a folder named 'structures' will be created in the current directory
        verbose : bool, optional
            Whether to provide information about the generation process, default is True
        print_error : bool, optional
            Whether to print errors when a structure cannot be generated, default is False
        struc_name : str, optional
            Custom name for the structure file. If not provided, the name will be the name of the replacement algorithm
        custom_charges : Dict[str, float], optional
            Custom charges for the atoms in the structure. The keys should be the species strings and the values should be the charges. If not provided, all charges will be set to 0.
        modify_O_connected_to_Al : bool, optional
            Whether to modify the O atoms connected to Al atoms in the structure (O -> Oa), default is False
        modify_O_connected_to_Al_Al : bool, optional
            Whether to modify the O atoms connected to Al atoms in the structure (O -> Oaa), default is False
            If modify_O_connected_to_Al is False, this parameter will be ignored
        supercell : Union[int, List[int], tuple], optional
            Supercell expansion to apply before placing defects. Defaults to (1, 1, 1).

        Returns
        -------
        List[Structure]
            List of generated structures
        """
        if not modify_O_connected_to_Al:
            modify_O_connected_to_Al_Al = False

        writepath = self._prepare_writepath(write, writepath, overwrite_ok)

        self.replace_algo = self._get_replace_algo(replace_algo)
        self.create_algo = self._get_create_algo(create_algo)
        self.post_algo = post_algo

        supercell = normalize_supercell(supercell)

        structure_for_creation, mask_for_creation, graph_for_replacement = (
            self._prepare_generation_context(supercell, *args, **kwargs)
        )

        structures = []

        total_failed = 0
        failed = 0
        written_count = 0

        start = time()
        for i in range(n_structures):

            sub_array, iteration_failed = self._try_replace_until_success(
                graph_for_replacement,
                n_subs,
                max_tries,
                print_error,
                *args,
                **kwargs,
            )
            total_failed += iteration_failed

            if sub_array is None:
                failed += 1
                continue

            new_structure = self.create_algo(
                structure_for_creation,
                mask_for_creation,
                sub_array,
                modify_O_connected_to_Al=modify_O_connected_to_Al,
                modify_O_connected_to_Al_Al=modify_O_connected_to_Al_Al,
                download_path=self.download_path,
                supercell=supercell,
                *args,
                **kwargs,
            ) # type: ignore
            if self.post_algo is not None:
                new_structure = self.post_algo(new_structure, *args, **kwargs)
            structures.extend(new_structure)
            if write:
                written_count = self._write_generated_structures(
                    new_structure,
                    writepath,
                    written_count,
                    struc_name,
                    custom_charges,
                    *args,
                    **kwargs,
                )

        end = time()
        if verbose:
            print(
                f"Successfully generated {n_structures - failed} structures in {end - start:.3f} seconds"
            )
            print(f"Failed to generate {failed} structures")
            print(f"Failed to generate new structures {total_failed} times")
        return structures

    def _prepare_writepath(
        self,
        write: bool,
        writepath: Optional[str],
        overwrite_ok: bool,
    ) -> Optional[str]:
        """Validate and optionally create the output directory for generated structures."""
        if not write:
            return writepath

        if writepath is None:
            writepath = "structures"

        if not os.path.exists(writepath):
            os.makedirs(writepath)
            return writepath

        if os.listdir(writepath) and not overwrite_ok:
            raise ValueError(
                f"Path {writepath} already contains files. Please provide an empty or non-existing path or set write to False."
            )

        return writepath

    def _prepare_generation_context(self, supercell, *args, **kwargs):
        """Build the structure, mask, and graph used for replacement and creation."""
        if supercell == (1, 1, 1):
            return self.structure, self.mask, self.structure_graph

        structure_for_creation = self.structure.copy()
        structure_for_creation.make_supercell(supercell)

        mask_for_creation = self.mask_method(
            structure_for_creation,
            self.mask_method_input,
            *args,
            **kwargs,
        ) # type: ignore

        graph_kwargs = dict(self.graph_kwargs)
        graph_kwargs.pop("supercell", None)
        if self.graph_method == mof_graph:
            if self.cif_path is None:
                raise ValueError("cif_path is required for mof graph generation")
            graph_for_replacement = self.graph_method(
                self.structure,
                mask=self.mask,
                download_path=self.download_path,
                cif_path=self.cif_path,
                supercell=supercell,
                *self.graph_args,
                **graph_kwargs,
            ) # type: ignore
        else:
            graph_for_replacement = self.graph_method(
                structure_for_creation,
                mask=mask_for_creation,
                *self.graph_args,
                **graph_kwargs,
            ) # type: ignore

        return structure_for_creation, mask_for_creation, graph_for_replacement

    def _try_replace_until_success(
        self,
        graph,
        n_subs: int,
        max_tries: int,
        print_error: bool,
        *args,
        **kwargs,
    ):
        """Retry node replacement up to ``max_tries`` times and count failures."""
        failed_attempts = 0
        for _ in range(max_tries):
            try:
                return self._replace(graph, n_subs, *args, **kwargs), failed_attempts
            except Exception as exc:
                failed_attempts += 1
                if print_error:
                    print(f"Failed to generate new structure: {exc}")

        return None, failed_attempts

    def _write_generated_structures(
        self,
        structures: List[Structure],
        writepath: Optional[str],
        start_index: int,
        struc_name: Optional[str],
        custom_charges: Optional[Dict[str, float]],
        *args,
        **kwargs,
    ) -> int:
        """Write generated structures and return the next available output index."""
        for index, structure in enumerate(structures, start=start_index):
            self._write_structure(
                structure,
                writepath,
                index,
                struc_name,
                custom_charges,
                *args,
                **kwargs,
            )

        return start_index + len(structures)

    def _get_mask_method(self, mask_method: Optional[Union[List[str], ndarray, str]]):
        """Resolve a mask-method specifier into a callable mask function."""
        if mask_method is None:
            return mask_all
        elif isinstance(mask_method, str):
            if mask_method == "zeolite":
                return mask_zeo
            elif mask_method == "h_on_c":
                return mask_h_on_c
            else:
                raise ValueError(f"Unknown mask method: {mask_method}")
        elif isinstance(mask_method, list):
            # if all elements of the list are atoms, return mask_species
            if all([type(msk) == str for msk in mask_method]) and all(
                [is_atom(species) for species in mask_method]
            ):
                return mask_species
            # otherwise, create a combination of the masks
            else:
                masks = [
                    self._get_mask_method(msk_method) for msk_method in mask_method
                ]
                return mask_combination(masks)
        elif isinstance(mask_method, np.ndarray):
            if len(mask_method.shape) == 1:
                return mask_array
            elif mask_method.shape == (3, 2):
                return mask_box
            else:
                raise ValueError("Mask array must be 1D or have shape (3,2)")
        else:
            raise ValueError("Unknown mask method")

    def _get_replace_algo(self, replace_algo: Union[str, Callable]):
        """Resolve a replacement algorithm name or pass through a callable."""
        if isinstance(replace_algo, str):
            if replace_algo == "random":
                return random
            elif replace_algo == "random_lowenstein":
                return random_lowenstein
            elif replace_algo == "lowenstein":
                return lowenstein
            elif replace_algo == "clusters":
                return clusters
            elif replace_algo == "multi_clusters":
                return multi_clusters
            elif replace_algo == "chains":
                return chains
            elif replace_algo == "maximize_entropy":
                return maximize_entropy
            else:
                raise ValueError(f"Unknown replace algorithm: {replace_algo}")
        else:
            return replace_algo

    def _get_create_algo(self, create_algo: Union[str, Callable]):
        """Resolve a structure-creation algorithm name or pass through a callable."""
        if isinstance(create_algo, str):
            if create_algo == "zeolite":
                return create_zeo
            if create_algo == "dmof":
                return create_dmof
            if create_algo == "defect_mof":
                return create_defect_mof
            else:
                raise ValueError(f"Unknown create algorithm: {create_algo}")
        else:
            return create_algo

    def _write_structure(
        self, structure: Structure, 
        writepath: Optional[str] = None, 
        i: int = 0,
        struc_name: Optional[str] = None,
        custom_charges: Optional[Dict[str, float]] = None,
         *args, **kwargs
    ):
        """
        Write a structure to a file

        Parameters
        ----------
        structure : Structure
            Structure to write
        writepath : str, optional
            Path to write the structure to, default is None
        i : int
            Index of the structure, default is 0
        struc_name : str, optional
            Custom name for the structure file. If not provided, the name will be the name of the replacement algorithm
        custom_charges : Dict[str, float], optional
            Custom charges for the atoms in the structure. The keys should be the species strings and the values should be the charges. If not provided, all charges will be set to 0.

        Returns
        -------
        None
        """
        if writepath is None:
            writepath = "structures"

        if struc_name is None:
            struc_name = self.replace_algo.__name__


        write_cif(
            structure,
            filename=os.path.join(writepath, f"{self.name}_{struc_name}_{i}.cif"),
            custom_charges=custom_charges,
        )
        
    def _replace(self, graph, n_subs: int, *args, **kwargs):
        """
        Replace n_subs nodes in the graph

        Parameters
        ----------
        n_subs : int
            Number of nodes to replace

        Returns
        -------
        np.array
            Array of selected nodes to replace
        """
        sub_array = self.replace_algo(graph, n_subs, *args, **kwargs)
        return sub_array

    def _get_graph_method(self, graph_method: Optional[Union[str, Callable]] = None):
        """Resolve a graph-construction method name or pass through a callable."""
        if graph_method is None:
            raise ValueError("graph_method must be provided before initializing a structure")
        if isinstance(graph_method, str):
            if graph_method == "zeolite":
                return zeo_graph
            elif graph_method == "radius":
                return radius_graph
            elif graph_method == "mof":
                return mof_graph
            else:
                raise ValueError(f"Unknown graph method: {graph_method}")
        else:
            return graph_method

    def _read_structure(self, cif_path: str, check_cif: bool = False, site_tolerance: float = 1e-3):
        """
        Read a structure from a cif file

        Parameters
        ----------
        cif_path : str
            Path to the cif file
        check_cif : bool, optional
            Check the cif file for errors, default is False

        Returns
        -------
        Structure
            Structure object of the cif file
        """
        parser = CifParser(cif_path, check_cif=check_cif, site_tolerance=site_tolerance)
        structure = parser.parse_structures(primitive=False)[0]
        return structure

    def __repr__(self):
        """Return a compact debug representation of the PORRAN instance."""
        return f"PORRAN(cif_path={self.cif_path}, graph_method={self.graph_method}, mask_method={self.mask_method})"

    def __str__(self):
        """Return a user-friendly string representation of the PORRAN instance."""
        return f"PORRAN(cif_path={self.cif_path}, graph_method={self.graph_method}, mask_method={self.mask_method})"

    def set_seed(self, seed: int):
        """Set NumPy's global random seed for reproducible sampling."""
        np.random.seed(seed)
