# Báo cáo Day 09 — Phạm Quang Huy, 2A202604055

## 1. Kết luận trước
**HOLD** (mã thoát 10). Lô vào **224 ảnh**; bước nhận loại **22 ảnh** → **202 ảnh** qua QC; sau sửa tự động và người duyệt **không còn lỗi bắt buộc** (`n_errors = 0`),
nhưng **38/202 ảnh (18,8%)** còn cảnh báo chưa người duyệt, ≥ ngưỡng 10% (`08_summary.json`). Dataset 202 ảnh được xuất ở `export/clean_v1/` chỉ là **bản nháp** (`manifest.json`: `decision: Hold`), chưa phải bản phát hành.
Lý do chính: hàng đợi duyệt có **125 ảnh** nhưng ngân sách người chỉ **80 ảnh**.

## 2. Lỗi và phân việc
Từ `01_intake` (mẫu số 224 ảnh) và `02_error_list_v1` (mẫu số 202 ảnh). Phần lỗi bắt buộc (E): 36 lỗi QC + 22 ảnh bị loại ở cửa vào; phần cảnh báo (W): 153.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `DUP_IMAGE_EXACT` / `DUP_IMAGE_NEAR` / `LEAK_TEST` | 8 / 4 / 5 ảnh | máy (loại ở cửa vào) | trùng md5/phash: giữ ảnh đầu; ảnh trùng test làm metric đẹp giả |
| `BAD_SIZE` / `CORRUPT_IMAGE` | 3 / 2 ảnh | máy (loại) | cạnh < 250px / không mở được: không gán nhãn được |
| `BOX_OUT_OF_IMAGE` | 8 dòng / 8 ảnh | máy: `clip_box` (8) | cắt về [0,1], giữ lớp — sửa xác định, không đoán |
| `DUP_BOX` / `BAD_CLASS_ID` / `BOX_ZERO` / `BAD_FORMAT` | 7 / 6 / 4 / 3 dòng | máy: `drop_line` (20) | dòng hỏng, không có cách sửa đúng → xoá, ghi log trước → sau (`03_fixes.csv`) |
| `BAD_CLASS_NAME` | 2 lỗi / `data.yaml` (`Person`, `Bus`) | máy: `rename_classes` | ghi lại đúng `cfg["classes"]` |
| `MISSING_LABEL_FILE` | 6 ảnh | **người** (`needs_label`) | máy không được bịa nhãn; file rỗng nghĩa là "ảnh nền", khác "chưa gán" |
| `POSSIBLE_MISSING` | 53 / 52 ảnh | người | model lớn thấy vật mà nhãn không có — có thể model sai |
| `POSSIBLE_FALSE_POS` / `POSSIBLE_LOOSE_BOX` / `POSSIBLE_WRONG_CLASS` | 29 / 28 / 18 cờ (27 / 24 / 18 ảnh) | người | lỗi ngữ nghĩa: chỉ nghi ngờ, cần mắt người |
| `EMPTY_LABEL` | 12 ảnh (15 sau sửa) | người | ảnh nền hay thiếu nhãn? +3 ảnh do autofix xoá dòng duy nhất (img_0050, img_0104, img_0194) |
| `BOX_TINY` / `BOX_HUGE` / `CLASS_RARE` | 6 dòng / 3 ảnh / 4 (lớp −1, 4, 7, 93) | người / tự hết | `CLASS_RARE` là id rác, hết sau khi xoá dòng `BAD_CLASS_ID` |

**Hàng đợi** (`05_review_queue.csv`, 125 ảnh) xếp theo điểm = Σ trọng số mã × (1 + conf của model). Đầu hàng: `img_0020` (11,5: vừa bị xoá dòng, vừa MISSING + WRONG_CLASS),
rồi 6 ảnh `NEEDS_LABEL` (10,0 — không có người thì chắc chắn Rework), rồi các ảnh nhiều cờ ngữ nghĩa conf cao. Ngân sách 80 ảnh dừng ở điểm 3,343 (#80), ảnh #81 là 3,333.
80 ảnh được duyệt gồm 6/6 ảnh thiếu nhãn, 50/52 ảnh `POSSIBLE_MISSING`, 18/18 `WRONG_CLASS`, 12/23 `FALSE_POS`, 8/24 `LOOSE_BOX` — tức người xem gần hết lỗi ngữ nghĩa nặng; phần để lại chủ yếu là `LOOSE_BOX`/`FALSE_POS` một cờ.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | NEAR: 1,00 / 0,50 → 1,00 / 0,67 (EXACT giữ 1,00 / 1,00) | lỗi này **loại ảnh** nên ưu tiên precision. 10–14 cho kết quả như nhau trên calib **và** raw (17 ảnh trùng/rò rỉ); chọn giữa khoảng để có biên an toàn hai phía |
| `second_opinion.conf` | 0,25 | **0,6** | MISSING 0,50 / 1,00 → 0,57 / 1,00; WRONG_CLASS 0,56 / 0,83 → 0,62 / 0,83; LOOSE_BOX 0,29 / 0,40 → 0,20 / 0,20 | đây là **cảnh báo cho người**: giá của FN (thiếu nhãn lọt vào dataset) cao hơn FP. 0,6 là conf cao nhất còn giữ recall MISSING = 1,00; 0,7 / 0,8 giảm hàng đợi nhưng recall MISSING còn 0,88 / 0,75 |

**Chỗ calibration đánh lừa:** trên calib, `phash_dist = 16` có F1 NEAR cao nhất (0,91). Nhưng chạy trên raw, số ảnh bị loại vì trùng/rò rỉ tăng từ **17 lên 26** (LEAK_TEST 5 → 11).
Mở xem các cặp: ở 16, ảnh trượt tuyết `img_0008` bị coi là trùng ảnh tennis `img_0011`, ảnh bàn laptop `img_0013` bị coi là rò rỉ `test_087` (sân bóng chày) — ảnh khác hẳn nhau bị xoá oan.
Calib chỉ có 6 ca NEAR nên mỗi ca = 17% recall; ngưỡng ăn khớp calib lại sai trên raw. Tương tự, chọn conf = 0,8 cho ra **Release** trên raw nhưng là "Release giả" vì checker bỏ sót nhiều hơn.
`LOOSE_BOX` chỉ có 5 lỗi chèn nên số của nó nhảy 20% mỗi ca — không dùng nó để chọn conf.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn thật:** `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` → kết luận **REWORK**, **mã thoát 20**, log in
"KHÔNG xuất: còn lỗi bắt buộc (Rework)", thư mục `work/exp_nb/clean_v1` **không được tạo**. Lý do: 6 ảnh `MISSING_LABEL_FILE` (img_0031, 0053, 0126, 0135, 0170, 0219) không có người gán;
máy chỉ ghi `needs_label`, không bịa nhãn. Với ngân sách 80, 6 ảnh này nằm ở #2–#7 hàng đợi nên được gán và lô chuyển từ Rework sang Hold.

**(b) Điểm mù có bằng chứng:**
1. **Thiếu nhãn trên ảnh nhãn rỗng:** 4 ảnh `EMPTY_LABEL` chưa duyệt (`img_0013`, `img_0097`, `img_0129`, `img_0192`). Mở xem: `img_0192` rõ ràng có **người** nhưng file nhãn rỗng,
   và `second_opinion` không cờ `POSSIBLE_MISSING` (người bị che phần lớn, model không thấy đủ tin cậy). `EMPTY_LABEL` chỉ có trọng số 1 nên các ảnh này rơi ngoài 80 ảnh được duyệt → thiếu nhãn còn trong bản nháp.
2. **Ảnh trùng bị dịch/cắt:** cặp `img_0074` / `img_0084` (cùng ảnh, khoảng cách phash 16) chỉ bị bắt ở `phash_dist = 16`, mà 16 lại gộp nhầm ảnh khác nhau → với 12 cặp này lọt qua.
3. **Checker cũng sai:** 25/80 ảnh **đã thay bằng nhãn người** vẫn bị `second_opinion` cờ (11 MISSING, 10 FALSE_POS) — model lớn bất đồng ~30% ngay cả với nhãn đúng, nên nhầm lớp mà model cũng nhầm sẽ không được cờ.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Colab T4, `yolo26n.pt`, 20 epoch, imgsz 640, batch 16, seed 0 và 1; cùng val (`gold_val`, 40 ảnh) và test (100 ảnh nhãn người). `raw` = 224 ảnh nhãn gốc; `clean` = 202 ảnh của bản nháp Hold (`comparison.md`):

| biến thể | #ảnh train | số seed | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0,726 ± 0,000 | 0,512 ± 0,004 | 0,764 ± 0,061 | 0,649 ± 0,028 |
| clean | 202 | 2 | 0,701 ± 0,025 | 0,487 ± 0,008 | 0,759 ± 0,003 | 0,625 ± 0,019 |

AP50 theo lớp (TB 2 seed), raw → clean: car 0,738 → 0,738 · person 0,822 → 0,823 · bus 0,695 → 0,703 · **truck 0,648 → 0,538**.

**Kết luận trung thực:** chênh mAP50 (clean − raw) = **−0,025**, bằng đúng độ lệch giữa 2 seed của `clean` (0,025; seed 0 = 0,718, seed 1 = 0,683). Với 2 seed và test 100 ảnh, **chưa kết luận được** clean tốt hơn hay kém hơn raw —
chắc chắn **không** có bằng chứng pipeline làm mAP tăng. Gần như toàn bộ chênh lệch nằm ở **truck**; ba lớp còn lại khác nhau ≤ 0,01.

**Giả thuyết và cách kiểm:**
1. **Rò rỉ test làm điểm `raw` bị thổi phồng:** `raw` chứa 5 ảnh trùng test (`LEAK_TEST`: test_009, 033, 039, 044, 085 — 20 box, có 1 truck) nên model `raw` đã "thấy trước" 5% tập test; `clean` đã loại chúng.
   *Kiểm:* đánh giá lại cả hai model trên test **bỏ 5 ảnh này**.
2. **Truck là lớp ít mẫu nên rất nhiễu:** test chỉ có 24 box truck / 16 ảnh; ngay trong `clean`, AP truck giữa 2 seed đã chênh 0,10 (0,590 vs 0,486). Vài box đổi đúng/sai là đủ tạo ra −0,11.
   *Kiểm:* chạy ≥ 5 seed, hoặc bootstrap tập test để có khoảng tin cậy theo lớp.
3. **`clean` mới sửa được một phần lỗi ngữ nghĩa:** người chỉ duyệt 80/202 ảnh; 122 ảnh còn lại giữ nhãn máy (38 ảnh còn cảnh báo), trong khi `raw` có thêm 22 ảnh (gồm 12 ảnh trùng). Lỗi cú pháp của `raw` phần lớn bị Ultralytics tự bỏ qua khi train nên ít gây hại.
   *Kiểm:* thêm biến thể `oracle` (nhãn người, chặn trên) để xem dư địa thật sự của việc làm sạch.

**Việc tiếp theo:** đánh giá lại trên test bỏ ảnh rò rỉ, chạy thêm seed + biến thể `oracle`, rồi mới kết luận về giá trị của pipeline đối với mAP.

## 6. Quyết định và ghi chú AI
**(a) Quyết định: HOLD.** Không phát hành; bản nháp `clean_v1` chỉ dùng thử nghiệm. Để thành **Release** cần ≤ 20/202 ảnh còn cảnh báo chưa duyệt, tức người xem thêm ≥ 18 ảnh
ở #81–#125 hàng đợi (ưu tiên 4 ảnh `EMPTY_LABEL` ở mục 4b) hoặc tăng ngân sách duyệt lên ~100 ảnh; **không** hạ ngưỡng `conf` chỉ để hàng đợi vừa ngân sách.
Lưu ý: trên Windows `check_submission.py` báo `pipeline\calibrate.py` nhắc `MANIFEST`/`data/calib` — đó là file đề (sha1 khớp `scripts/_protected.json`), lỗi do so đường dẫn `\` với `/`.

**(b) Dùng AI (Claude Code):
- *Prompt chính:* "Viết `pipeline/<module>.py` theo docstring , ngưỡng đọc từ `cfg`, không sửa `data/raw`, dòng hỏng không được làm chết; chạy `pytest tests/test_<module>.py`."
- *Cách kiểm:* đọc từng hàm; chạy unit test (79/79 xanh) và thêm ca khó (1 dòng 2 lỗi, dòng trống giữa file, nhiều dự đoán/ảnh, biên đúng 10% với mẫu số 10…1000); chạy autofix 2 lần trên bản sao raw (lần 2 chỉ còn `needs_label`) và so sha1 `data/raw` trước/sau.
- *Một lần AI sai:* bản nháp đầu của báo cáo dựa vào nhãn tham chiếu của người duyệt cho ảnh **chưa duyệt** — tức là xem đáp án lô raw (RULES.md cấm); đã sửa mục 4b chỉ dùng bằng chứng mở ảnh và báo cáo pipeline. Ngoài ra, lần kiểm đầu của `second_opinion`, AI chạy thẳng trên `data/raw` nên script chết ở 2 ảnh hỏng không có trong cache (img_0010, img_0136); thực tế pipeline loại 2 ảnh này ở bước nhận nên phải kiểm sau bước nhận. AI cũng sinh `calibration_report.md` trước khi đổi config, phải chạy lại để bảng "cấu hình hiện tại" khớp ngưỡng đã chọn.

(`submission/bonus/aspect_ratio.csv`).
Check chạy riêng, **không** nằm trong `QC_CHECKS` (file đăng ký được ghim) và ngưỡng để trong code với `cfg["aspect_ratio"]` ghi đè được (config bị khoá ngoài 2 khoá CALIBRATE), nên không đổi kết luận Hold.
