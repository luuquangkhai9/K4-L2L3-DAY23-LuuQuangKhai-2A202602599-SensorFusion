# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Báo cáo kết quả chạy thực tế. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Lưu Quang Khải
- MSSV: 2A202602599
- Email: toilaquangkhai@gmail.com (theo cấu hình Git hiện có)
- Link repo (fork): https://github.com/luuquangkhai9/K4-L2L3-DAY23-LuuQuangKhai-2A202602599-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Commit bàn giao được đánh dấu bằng tag `day23-submission`; lấy hash 40 ký tự bằng `git rev-parse day23-submission`. Tag trỏ tới commit chứa đầy đủ báo cáo và artifacts, tránh ghi hash tự tham chiếu trong chính commit đó.

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0,198]` (199 frame/mode), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, seed `0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `0.9700934579439252`, `0.7004048582995951`, `519 / 16 / 222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.15032268781360134 m`, `502`, `11.343649056695735 m²`, `0`, `239`, `2.522613065326633`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.1358667883353908 m`, `502`, `9.26681165463209 m²`, `0`, `239`, `2.522613065326633`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Fused giảm RMSE `0.014455899478210549 m` (khoảng `9.62%`) so với LiDAR. Hai mode giữ cùng 502 matches, 0 ghost và 239 miss, nên sự cải thiện không đến từ việc giảm số cặp được đánh giá. Kết quả phù hợp với vai trò camera tinh chỉnh vị trí của track đã có; riêng log tổng hợp không chứng minh danh tính từng cặp ghép hoàn toàn giống nhau giữa hai mode.

| Chỉ số tracking | LiDAR-only | LiDAR + camera | Mục tiêu rubric |
|---|---:|---:|---:|
| RMSE 3D (m) | 0.150323 | 0.135867 | <= 0.45 |
| Matches | 502 | 502 | Đọc cùng RMSE |
| Ghost track-frames | 0 | 0 | Đọc cùng RMSE |
| Missed GT-frames | 239 | 239 | Đọc cùng RMSE |
| Precision track | 1.0 | 1.0 | >= 0.75 |
| Coverage | 0.9672447013487476 | 0.9672447013487476 | >= 0.70 |

`precision_track = 502/(502+0) = 1`; `coverage = 502/519 ≈ 96.72%`.
Hiệu `rmse_fused - rmse_lidar = -0.014456 m <= 0.05 m`.
Cả hai mode đạt ngưỡng RMSE, precision_track và coverage trong rubric.
Đây là kiểm tra ngưỡng tự động; điểm báo cáo/vấn đáp do giảng viên đánh giá.

Có tổng 741 GT vehicle-frames. Detection bỏ sót 222; tracking miss 239,
chênh 17 lượt so với tổng detection FN, phù hợp với việc tracking cần thời
gian xác nhận và duy trì track. Không đồng nhất chênh lệch này với một
nguyên nhân duy nhất vì hai phép ghép đánh giá khác nhau. Ví dụ frame 0–3,
mỗi frame detector có 2 TP nhưng tracker chưa confirmed nên có 2 miss;
frame 4 có 2 confirmed, 2 matches, 0 miss. Frame 198 có 9 GT, 3 detection TP,
3 matches và 6 miss ở cả hai mode. Các ví dụ nằm trong log per-mode.

Đã kiểm chứng bằng `fusion_lab.evaluation.validate_metrics_records` cho log
gộp và từng log riêng: đủ 398 record (199/mode), đủ frame 0–198,
không trùng cặp mode/frame, các invariant và tổng/trung bình khớp metrics.
Detection giống nhau giữa hai mode; log riêng trùng phần tương ứng của log gộp.
Sáu artifacts được runner sinh tự động, không sửa tay.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

Môi trường chạy: Windows, Python 3.12.15, NumPy 2.5.3, SciPy 1.18.1,
PyTorch 2.14.1+cpu; inference CPU. Trên PowerShell dùng:

```powershell
$env:PYTHONUTF8 = '1'
$env:DAY23_STUDENT_ROOT = (Resolve-Path student).Path
$env:FUSION_LAB_PLATFORM = (Resolve-Path platform).Path
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.venv/Scripts/python.exe -m pytest student/tests -q
.venv/Scripts/fusion-run-lab.exe --config student/config/paths.yaml --fusion compare --seed 0
.venv/Scripts/python.exe tools/check_submission.py
```

`paths.yaml` local đặt frame_start=0, frame_end=198; không commit cấu hình
local, dữ liệu hoặc weights. PYTHONUTF8 xử lý encoding tiếng Việt của các
test/checker trên Windows, không thay đổi thuật toán hay bộ test.

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   State chung là `(px, py, pz, vx, vy, vz)`. LiDAR đo tâm hộp 3D:
   z có kích thước 3×1, đơn vị mét; R 3×3 với phương sai mặc định
   `(0.1², 0.1², 0.1²)` m². Mô hình đo đổi vị trí vehicle sang sensor,
   H 3×6 tuyến tính. Camera đo `(u,v)` pixel, z 2×1, R 2×2 với
   `(5²,5²)` pixel²; h(x) là phép chiếu pinhole phi tuyến, H là Jacobian
   2×6 do platform cung cấp. Trong `workspace/kalman.py:ekf_update`, cùng
   công thức gamma, S và K dùng được cho cả hai số chiều. Camera không cho
   trực tiếp độ sâu như LiDAR. R là covariance EKF giả định; runner thực tế
   thêm nhiễu camera có độ lệch chuẩn 0.5 pixel, không phải 5 pixel
   (`platform/fusion_lab/scripts/run_lab.py:_front_observations`).

2. Vì sao cần gating Mahalanobis trước khi gán?

   `workspace/association.py` tính `d² = gamma.T @ S^-1 @ gamma`,
   với `S = H @ P @ H.T + R`, rồi so với phân vị chi-square 0.995
   theo dim_meas (3 cho LiDAR, 2 cho camera). Một residual lớn có thể hợp lý
   khi uncertainty lớn; khoảng cách Euclidean không xét điều này.
   Gate loại cặp không tương thích trước greedy matching để giảm nguy cơ
   gán nhầm xe. Kiểm tra FOV trước MHD còn ngăn chiếu điểm sau camera,
   độ sâu <= 1e-6 hoặc tọa độ không hữu hạn. Cặp bị loại giữ chi phí inf;
   track/measurement vẫn nằm trong danh sách chưa ghép để quản lý đúng.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Là track-then-fuse: mỗi mode giữ một danh sách track và một EKF.
   Trong `platform/fusion_lab/scripts/run_lab.py:run`, runner predict mọi
   track một lần mỗi frame, sau đó gọi association/update/lifecycle LiDAR,
   rồi association/update camera trên chính các track ấy khi có FRONT.
   Camera không tạo một tracker riêng và không trộn raw point cloud với ảnh.
   Compare là hai lần chạy độc lập (LiDAR-only và fused) để đối chiếu.
   Trong lab, hai sensor có `Measurement.t = frame_index * 0.1`, cùng thời
   điểm logic nên không predict lần thứ hai giữa hai update.

   Log `artifacts/grade_run.log` có record phân biệt `mode=lidar/fused`
   cho từng frame. Frame 0–3 có detection nhưng confirmed=0; frame 4 bắt
   đầu có confirmed track, thể hiện detection khác với tracking có lịch sử.
   Log chứa thống kê cuối frame, không ghi từng EKF update; thứ tự cập nhật
   được chứng minh bằng runner ở trên, không suy ra riêng từ JSONL.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Sai extrinsic làm p_s và h(x) lệch có hệ thống so với tâm hộp quan sát:
   gamma camera có bias hoặc thay đổi theo vị trí/độ sâu thay vì chỉ nhiễu
   quanh 0. Nếu d² vượt gate, camera measurement không được gán; nếu bias
   vẫn lọt gate, update có thể kéo x sai và làm RMSE fused tăng.
   Cần xem residual theo thời gian, số cặp camera được chấp nhận và RMSE
   cùng matches/ghost/miss để chẩn đoán. Đây là giải thích cơ chế qua
   `workspace/camera_fusion.py:camera_measurement_prediction` và
   `workspace/association.py:mahalanobis_distance`, không phải kết quả
   thí nghiệm calibration đã thực hiện.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi meas_list rỗng, không thể lấy sensor từ measurement đầu tiên.
   Sensor tường minh cho manager biết đây là lượt LiDAR phải xử lý miss
   hay lượt camera không thay đổi lifecycle. Hàm luôn gọi manage_tracks,
   kể cả ma trận rỗng/toàn inf. LiDAR cung cấp vị trí 3D và là nguồn quyết
   định tồn tại: hit cộng score, miss trong FOV trừ score, unmatched đo
   tạo track, track yếu/uncertain bị xóa. Camera chỉ tinh chỉnh x/P của
   track đã có. `platform/fusion_lab/tracking/manager.py` chặn scoring
   trong handle_updated_track và trả về ngay trong manage_tracks khi
   sensor không phải LiDAR. Vì vậy thiếu camera không được coi là miss
   LiDAR và camera cũng không cứu hoặc sinh track bằng score.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   `workspace/track_management.py` khởi tạo state initialized, score=1/6,
   vận tốc 0; vị trí và covariance measurement được đổi sang vehicle.
   Mỗi hit LiDAR tăng 1/6, chặn trên 1; track chưa confirmed thành tentative
   khi có hit và confirmed khi score **> 0.8**. Score đúng 0.8 chưa xác nhận.
   Miss trong FOV giảm 1/6; ngoài FOV không trừ. Track đã confirmed giữ
   trạng thái confirmed sau miss, nhưng vẫn có thể bị xóa nếu score quá thấp.
   Xóa khi một trong các điều kiện đúng: Pxx hoặc Pyy **> 9 m²**;
   confirmed có score **< 0.6**; chưa confirmed có score **<= 0**.
   Biên P=9 hoặc confirmed score=0.6 không bị xóa vì riêng điều kiện đó.
   Tất cả quyết định tồn tại chỉ diễn ra trong lượt LiDAR.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không thực hiện bonus; ưu tiên hoàn thành và kiểm chứng phần bắt buộc.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng OpenAI Codex (ChatGPT).
- Dùng cho phần nào (hàm, câu hỏi, debug): Đọc yêu cầu và lập PLAN.md;. Code có hỗ trợ trực tiếp từ AI.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; 128 tests pass, không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên implementation cung cấp sẵn
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Repo đã đổi tên theo quy định, push code/báo cáo/artifacts lên GitHub; bản bàn giao gắn tag `day23-submission`
- [ ] Đã nộp link repo + commit hash trên LMS (học viên tự thực hiện sau bàn giao)
