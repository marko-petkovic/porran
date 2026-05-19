from porran import porran as porran_module


class _DummyStructure:
    def copy(self):
        return self

    def make_supercell(self, supercell):
        self.supercell = supercell


def test_prepare_generation_context_ignores_stale_supercell_kwarg(monkeypatch) -> None:
    seen = {}

    def fake_mof_graph(structure, *args, **kwargs):
        seen["kwargs"] = kwargs
        return "graph"

    monkeypatch.setattr(porran_module, "mof_graph", fake_mof_graph)

    porran = porran_module.PORRAN.__new__(porran_module.PORRAN)
    porran.structure = _DummyStructure()
    porran.mask = "mask"
    porran.mask_method = lambda *args, **kwargs: "mask2"
    porran.mask_method_input = None
    porran.graph_method = fake_mof_graph
    porran.graph_args = ()
    porran.graph_kwargs = {"supercell": (3, 3, 3), "radius": 4.5}
    porran.cif_path = "example.cif"
    porran.download_path = "downloads"

    _, mask_for_creation, graph = porran._prepare_generation_context((2, 2, 2))

    assert mask_for_creation == "mask2"
    assert graph == "graph"
    assert seen["kwargs"]["supercell"] == (2, 2, 2)
    assert seen["kwargs"]["radius"] == 4.5