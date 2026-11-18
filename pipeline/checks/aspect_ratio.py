"""CHECK THƯỞNG — box lệch tỉ lệ khung hình (quá mảnh / quá dẹt).   (RUBRIC: thưởng +3)

Test:  pytest tests/test_aspect_ratio.py -q
Chạy riêng:  python -m pipeline.checks.aspect_ratio [thư mục dataset, mặc định work/run/ds]

Vì sao cần: một box có cạnh dài gấp nhiều lần cạnh ngắn thường là box vẽ hỏng (kéo nhầm thành một vệt),
box bị cắt còn một mẩu ở mép ảnh, hoặc nhầm w/h khi chuyển định dạng. Các check cú pháp không bắt được vì toạ độ vẫn hợp lệ.

Tỉ lệ đo bằng PIXEL (w*W) / (h*H), không phải toạ độ chuẩn hoá: ảnh 640x480 với w = h = 0.5 là box 320x240, tỉ lệ 1.33.

| code            | severity | khi nào                                                           |
|-----------------|----------|-------------------------------------------------------------------|
| BOX_ODD_ASPECT  | WARN     | cạnh dài / cạnh ngắn (px) > `max_aspect`; extra={"aspect", "w_px", "h_px", "shape"} |

- Chỉ CẢNH BÁO: người nằm, xe bị che một phần vẫn có thể rất dẹt (nhãn người sạch của tập val cũng có ~2% box > 6:1).
- Bỏ qua dòng hỏng / id lạ / box rỗng (label_format đã báo) và ảnh không đọc được kích thước.
- Ngưỡng: `cfg["aspect_ratio"]["max_aspect"]` nếu có, mặc định `DEFAULTS`. Không thêm mục vào configs/pipeline.yaml vì
  check_submission.py khoá mọi mục ngoài 2 khoá CALIBRATE; check này cũng không nằm trong QC_CHECKS (file đăng ký được ghim),
  nên KHÔNG làm đổi kết luận Release/Hold/Rework của pipeline.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from ..common import ROOT, WARN, Dataset, Issue, load_config, parse_yolo_line, write_issues_csv

NAME = "aspect_ratio"
DEFAULTS = {"max_aspect": 6.0,   # chọn từ khảo sát: p95 mọi lớp < 3.7:1 trên raw; nhãn người tập val: 3/158 box > 6:1
            "border_tol": 0.01}  # box chạm mép ảnh (trong 1%) -> có thể là vật bị cắt mép: vẫn báo nhưng ghi at_border=True


def _params(cfg):
    return {**DEFAULTS, **(cfg.get(NAME) or {})}


def run(ds, cfg):
    p = _params(cfg)
    max_aspect, tol = p["max_aspect"], p["border_tol"]
    n_cls = len(ds.classes)
    issues = []
    for lab in ds.label_files():
        stem = lab.stem
        size = ds.image_size(stem) if ds.image_path(stem) else None
        if not size:
            continue
        W, H = size
        for lno, text in ds.read_label_lines(stem) or []:
            try:
                c, x, y, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if not 0 <= c < n_cls or w <= 0 or h <= 0:
                continue
            w_px, h_px = w * W, h * H
            ratio = max(w_px, h_px) / min(w_px, h_px)
            if ratio > max_aspect:
                shape = "dẹt" if w_px > h_px else "mảnh"
                at_border = min(x - w / 2, y - h / 2) <= tol or max(x + w / 2, y + h / 2) >= 1 - tol
                note = " — chạm mép ảnh, có thể vật bị cắt" if at_border else " — giữa ảnh, nghi box vẽ hỏng"
                issues.append(Issue(NAME, WARN, "BOX_ODD_ASPECT", stem,
                                    f"box {ds.classes[c]} quá {shape}: {w_px:.0f}x{h_px:.0f}px (tỉ lệ {ratio:.1f}:1 > {max_aspect:g}:1){note}",
                                    lno, {"aspect": round(ratio, 2), "w_px": round(w_px, 1), "h_px": round(h_px, 1),
                                          "shape": shape, "at_border": at_border}))
    return issues


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else ROOT / "work" / "run" / "ds"
    cfg = load_config()
    ds = Dataset(root, cfg["classes"])
    issues = run(ds, cfg)
    out = ROOT / "submission" / "bonus" / "aspect_ratio.csv"
    write_issues_csv(issues, out)
    print(f"quét {len(ds.image_files())} ảnh trong {root}: {len(issues)} box lệch tỉ lệ (> {_params(cfg)['max_aspect']:g}:1) "
          f"ở {len({i.image for i in issues})} ảnh -> {out.relative_to(ROOT)}")
    for i in issues:
        print(f"  {i.image}.txt:{i.line}  {i.message}")
    print("theo hình dạng:", dict(Counter(i.extra["shape"] for i in issues)),
          "| chạm mép ảnh:", sum(i.extra["at_border"] for i in issues), "/", len(issues))


if __name__ == "__main__":
    main()
