from stage1.config import Stage1Config


def test_config_roundtrip():
    cfg = Stage1Config.from_mapping({"grid": {"spatial_size": [8, 10, 12]}, "model": {"latent_channels": 16}})
    assert cfg.grid.spatial_size == (8, 10, 12)
    assert cfg.model.latent_channels == 16
    assert Stage1Config.from_mapping(cfg.to_dict()) == cfg
