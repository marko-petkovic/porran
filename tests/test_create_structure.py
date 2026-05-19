from collections import Counter

import numpy as np

from porran.create_structure import create_defect_mof


def test_create_defect_mof_returns_finite_capped_structure(
    repo_root,
    cau10_structure,
    cau10_full_mask,
) -> None:
    result = create_defect_mof(
        cau10_structure,
        cau10_full_mask,
        np.array([0]),
        download_path=str(repo_root / "downloads"),
    )

    capped_structure = result[0]
    tail_species = [site.species_string for site in capped_structure[-3:]]

    assert np.isfinite(capped_structure.cart_coords).all()
    assert tail_species.count("O") == 1
    assert tail_species.count("H") == 2


def test_create_defect_mof_single_linker_has_expected_species_delta(
    repo_root,
    cau10_structure,
    cau10_full_mask,
) -> None:
    base_counts = Counter(site.species_string for site in cau10_structure)

    capped_structure = create_defect_mof(
        cau10_structure,
        cau10_full_mask,
        np.array([0]),
        download_path=str(repo_root / "downloads"),
    )[0]

    counts = Counter(site.species_string for site in capped_structure)

    assert len(capped_structure) == len(cau10_structure) - 6
    assert counts["C"] - base_counts["C"] == -8
    assert counts["H"] - base_counts["H"] == 2
    assert counts["O"] == base_counts["O"]
    assert counts["Al"] == base_counts["Al"]


def test_create_defect_mof_multiple_linkers_scale_composition_change(
    repo_root,
    cau10_structure,
    cau10_full_mask,
) -> None:
    base_counts = Counter(site.species_string for site in cau10_structure)

    capped_structure = create_defect_mof(
        cau10_structure,
        cau10_full_mask,
        np.array([0, 1]),
        download_path=str(repo_root / "downloads"),
    )[0]

    counts = Counter(site.species_string for site in capped_structure)

    assert len(capped_structure) == len(cau10_structure) - 12
    assert counts["C"] - base_counts["C"] == -16
    assert counts["H"] - base_counts["H"] == 4
    assert counts["O"] == base_counts["O"]


def test_create_defect_mof_accepts_boolean_replacement_mask(
    repo_root,
    cau10_structure,
    cau10_full_mask,
) -> None:
    replacement_mask = np.zeros(16, dtype=bool)
    replacement_mask[0] = True

    from_indices = create_defect_mof(
        cau10_structure,
        cau10_full_mask,
        np.array([0]),
        download_path=str(repo_root / "downloads"),
    )[0]
    from_mask = create_defect_mof(
        cau10_structure,
        cau10_full_mask,
        replacement_mask,
        download_path=str(repo_root / "downloads"),
    )[0]

    assert len(from_mask) == len(from_indices)
    assert Counter(site.species_string for site in from_mask) == Counter(
        site.species_string for site in from_indices
    )
    assert np.isfinite(from_mask.cart_coords).all()


def test_create_defect_mof_supercell_applies_single_expanded_defect(
    repo_root,
    cau10_structure,
    cau10_full_mask,
) -> None:
    supercell = (2, 1, 1)

    capped_structure = create_defect_mof(
        cau10_structure,
        cau10_full_mask,
        np.array([0]),
        download_path=str(repo_root / "downloads"),
        supercell=supercell,
    )[0]

    assert len(capped_structure) == len(cau10_structure) * 2 - 6
    assert np.isfinite(capped_structure.cart_coords).all()
    assert capped_structure.lattice.a == cau10_structure.lattice.a * 2
    assert capped_structure.lattice.b == cau10_structure.lattice.b
    assert capped_structure.lattice.c == cau10_structure.lattice.c


def test_create_defect_mof_rejects_out_of_bounds_supercell_linker_index(
    repo_root,
    cau10_structure,
    cau10_full_mask,
) -> None:
    try:
        create_defect_mof(
            cau10_structure,
            cau10_full_mask,
            np.array([999999]),
            download_path=str(repo_root / "downloads"),
            supercell=(2, 1, 1),
        )
    except ValueError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("Expected out-of-bounds linker selection to raise ValueError")