"""Test cho check thưởng aspect_ratio (box quá mảnh / quá dẹt)."""
from pipeline.checks import aspect_ratio


def run(make_ds, cfg, lines, size=(640, 480)):
    ds = make_ds(images={"a": 1}, labels={"a": lines}, size=size)
    return aspect_ratio.run(ds, cfg)


def found(issues):
    return [(i.image, i.line) for i in issues if i.code == "BOX_ODD_ASPECT"]


def test_normal_boxes_are_silent(make_ds, cfg):
    assert run(make_ds, cfg, ["1 0.5 0.5 0.1 0.4", "0 0.5 0.5 0.3 0.2"]) == []      # người đứng 64x192 (3:1), xe 192x96 (2:1)


def test_thin_and_flat_boxes_are_warned(make_ds, cfg):
    iss = run(make_ds, cfg, ["1 0.5 0.5 0.02 0.5", "0 0.5 0.5 0.9 0.03", "0 0.5 0.5 0.2 0.2"])   # 12.8x240 (18.8:1), 576x14.4 (40:1)
    assert found(iss) == [("a", 1), ("a", 2)]
    assert [i.extra["shape"] for i in iss] == ["mảnh", "dẹt"]
    assert all(i.severity == "warn" for i in iss)


def test_ratio_uses_pixels_not_normalised_coords(make_ds, cfg):
    # w = h = 0.5 nhưng ảnh 1200x100 -> 600x50 px = 12:1
    assert found(run(make_ds, cfg, ["0 0.5 0.5 0.5 0.5"], size=(1200, 100))) == [("a", 1)]
    # w gấp 8 lần h theo toạ độ chuẩn hoá nhưng ảnh 100x800 -> 80x80 px = 1:1
    assert found(run(make_ds, cfg, ["0 0.5 0.5 0.8 0.1"], size=(100, 800))) == []


def test_boundary_is_not_flagged_and_threshold_comes_from_cfg(make_ds, cfg):
    exact = ["0 0.5 0.5 0.6 0.1"]                                   # 384x48 trên 640x480 = 8:1
    cfg["aspect_ratio"] = {"max_aspect": 8}
    assert found(run(make_ds, cfg, exact)) == []                   # đúng bằng ngưỡng: không báo
    cfg["aspect_ratio"] = {"max_aspect": 7.9}
    assert found(run(make_ds, cfg, exact)) == [("a", 1)]


def test_default_threshold_when_cfg_has_no_section(make_ds, cfg):
    assert "aspect_ratio" not in cfg
    assert found(run(make_ds, cfg, ["0 0.5 0.5 0.6 0.1"])) == [("a", 1)]      # 8:1 > mặc định 6:1


def test_bad_lines_do_not_crash_and_keep_line_numbers(make_ds, cfg):
    ds = make_ds(images={"a": 1}, size=(640, 480))
    (ds.labels_dir / "a.txt").write_text("garbage\n\n93 0.5 0.5 0.02 0.5\n1 0.5 0.5 0 0.5\n1 0.5 0.5 0.02 0.5\n")
    assert found(aspect_ratio.run(ds, cfg)) == [("a", 5)]


def test_box_touching_the_border_is_marked(make_ds, cfg):
    iss = run(make_ds, cfg, ["1 0.01 0.5 0.02 0.5", "1 0.5 0.5 0.02 0.5"])   # dòng 1 chạm mép trái, dòng 2 giữa ảnh
    assert found(iss) == [("a", 1), ("a", 2)]
    assert [i.extra["at_border"] for i in iss] == [True, False]


def test_label_without_image_is_skipped(make_ds, cfg):
    ds = make_ds(images={"a": 1}, labels={"a": ["0 0.5 0.5 0.2 0.2"], "orphan": ["1 0.5 0.5 0.02 0.5"]})
    assert aspect_ratio.run(ds, cfg) == []
