def test_readcif_reports_missing_atom_fields(tmp_path) -> None:
    bad_cif = tmp_path / "bad.cif"
    bad_cif.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a 10",
                "_cell_length_b 10",
                "_cell_length_c 10",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "loop_",
                "_atom_site_label",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "Si1 0.0 0.0",
            ]
        )
    )

    from porran.utils import readcif

    try:
        readcif(str(bad_cif))
    except ValueError as exc:
        assert "_atom_site_fract_z" in str(exc)
    else:
        raise AssertionError("readcif should raise a ValueError for missing atom fields")