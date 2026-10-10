# Kế hoạch hoàn thành Day 23 — Sensor Fusion

> Cập nhật thực hiện: Đã hoàn thành E–H và chạy compare đầy đủ frame 0–198,
> seed 0. Bộ test gốc có 128 test pass; RMSE LiDAR 0.150323 m, fused
> 0.135867 m. Báo cáo và sáu artifacts nằm trong student/. Hiện trạng ở mục 1
> bên dưới là ảnh chụp ban đầu khi lập kế hoạch, không phải trạng thái cuối.

## 1. Mục tiêu và hiện trạng

Hoàn thành tracker EKF 6D kết hợp LiDAR và camera, chạy so sánh trên Waymo,
nộp code E–H, sáu artifacts và báo cáo có số liệu kiểm chứng được.
Kế hoạch dựa trên README.md, CHECKPOINTS.md, RULES.md, RUBRIC.md,
SUBMISSION.md, docs/HUONG_DAN_KY_THUAT.md, code workspace và bộ test hiện có.

Hiện trạng kiểm tra ngày 10/10/2026:

- Part A–D đã có implementation: range image → BEV → FPN-ResNet → detection metrics.
- Part E–H còn đủ **17 hàm TODO**: E có 6, G có 3, F có 5, H có 3.
- `student/SUBMISSION.md` còn trống; `student/artifacts/` chỉ có `.gitkeep`.
- Chưa thấy dữ liệu, weights hoặc `student/config/paths.yaml` trong repo.
  Nếu dữ liệu đang ở ổ khác, dùng cấu hình local trỏ tới đó.
- Python trên PATH là 3.14.7; có uv và conda. Chạy `python -m pytest student/tests -q`
  bị **9 lỗi collection** do thiếu `fusion_lab`, Waymo reader, `easydict`, `cv2`.
  Chưa đánh giá được độ đúng của code qua lượt chạy này.
- Git working tree sạch trước khi tạo kế hoạch.
- Origin hiện dùng tên `K4-Track4-Day23-LuuQuangKhai-2A202602599-SensorFusion`;
  tài liệu nộp yêu cầu `K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion`.
  Kiểm tra thông báo giảng viên để xác nhận tên được chấp nhận trước khi nộp.

Phạm vi implementation: bốn module E–H trong `student/workspace/`, giữ API.
Không sửa platform, tests hoặc Part A–D để thay đổi kết quả chấm.

## 2. Thứ tự thực hiện và ngân sách thời gian

CP0 thực hiện trước; thời gian download và inference phụ thuộc máy/mạng.
Lịch 120 phút sau CP0 bám theo lịch gốc, có thể cần thêm thời gian chạy Waymo.

| Bước | Thời lượng dự kiến | Việc chính | Điều kiện hoàn tất |
|---|---:|---|---|
| CP0 | 20–40 phút + tải dữ liệu | Môi trường, paths, đọc A–D | Test chạy được, chỉ TODO E–H xfail; CLI hoạt động |
| CP1 / E | 25 phút | EKF predict/update | Test Kalman pass |
| CP2 / G | 25 phút | FOV, projection, camera covariance | Test camera pass |
| CP3 / F | 25 phút | Mahalanobis, chi-square gate, greedy association | Test association pass; lifecycle phụ thuộc H kiểm lại ở CP4 |
| CP4 / H | 20 phút | Khởi tạo, score, xác nhận, xóa track | Toàn bộ tests pass, không còn xfail |
| CP5 / I | 15 phút hoặc lâu hơn | Smoke run, compare đầy đủ | Artifacts hợp lệ và đánh giá chất lượng |
| CP6 | 10 phút | Báo cáo, kiểm tra, commit, nộp | Checker sẵn sàng; link và hash đúng |

Commit từng checkpoint bằng message `CPx: ...` sau khi kiểm tra kết quả.
Làm phần bắt buộc trước, bonus sau khi đủ bằng chứng nộp.

## 3. CP0 — Chuẩn bị trên Windows PowerShell

Tạo môi trường Python 3.12 riêng, chạy các lệnh từ gốc repo:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e platform/third_party/waymo_reader -e platform -e student pytest
$env:DAY23_STUDENT_ROOT = (Resolve-Path student).Path
$env:FUSION_LAB_PLATFORM = (Resolve-Path platform).Path
Copy-Item student/config/paths.example.yaml student/config/paths.yaml
.venv/Scripts/python.exe -m pytest student/tests -q
.venv/Scripts/fusion-run-lab.exe --help
```

Chỉ copy cấu hình nếu file local chưa tồn tại. Nạp lại hai biến môi trường mỗi
khi mở terminal mới. Có thể dùng conda theo `environment.yml` thay uv.

Chuẩn bị segment mặc định theo `data/README.md`:

- `data/Waymo/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- `data/weights/pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth`
- `waymo_dir` và `weights_dir` trong paths.yaml được giải quyết tương đối từ `student/`.
- Cấu hình mẫu là frame 0–20 để thử; lượt nộp phải dùng **0–198**, gồm cả hai đầu.

Đọc A–D để hiểu tâm hộp 3D là measurement LiDAR và detection metrics khác
tracking metrics. Đọc thêm adapters `tracking/sensors.py`, `filter.py`,
`manager.py` và tracking params trong platform để hiểu API, không chỉnh chúng.
Chưa có dữ liệu vẫn làm CP1–CP4 được.

## 4. CP1 — Part E: kalman.py

Implement sáu hàm theo hợp đồng hiện có:

1. `build_F`: ma trận 6×6 vận tốc không đổi, các ô vị trí–vận tốc bằng dt.
2. `build_Q`: dùng **mô hình lab `Q = dt * q * I6`**; đọc mặc định qua
   `get_tracking_params()`, không thay bằng mô hình nhiễu gia tốc khác.
3. `ekf_predict`: `x_pred = F @ x`, `P_pred = F @ P @ F.T + Q`.
4. `innovation`: `gamma = z - h(x)`.
5. `innovation_covariance`: `S = H @ P @ H.T + R`.
6. `ekf_update`: lấy H từ sensor, tính K, cập nhật x và P theo stub.

Giữ kích thước x 6×1 và P 6×6; phép toán phải đúng với ndarray và np.matrix.
Dùng `@` để nhân ma trận; có thể dùng linear solve để tính K/Mahalanobis.

Kiểm tra: `.venv/Scripts/python.exe -m pytest student/tests/test_kalman.py -v`.
Đạt khi cả shape, dự báo số học và update trường hợp scalar đều pass.

## 5. CP2 — Part G: camera_fusion.py

Implement ba hàm:

1. `is_in_field_of_view`: đổi vị trí vehicle → sensor, kiểm tra hữu hạn và
   góc `atan2(y_s, x_s)` trong sensor.fov. Camera cần `x_s > 1e-6`.
2. `camera_measurement_prediction`: `u = c_i - f_i*y_s/x_s`,
   `v = c_j - f_j*z_s/x_s`; trả ma trận 2×1. Với tọa độ không hữu hạn hoặc
   độ sâu không hợp lệ, raise ValueError chứa ngữ cảnh tọa độ trước phép chia.
3. `build_camera_measurement`: z 2×1, R là diag phương sai pixel từ params,
   trả dict có z, R, sensor.

Dùng extrinsic và FOV do sensor cung cấp; Jacobian H đã được platform viết.
Kiểm tra điểm trước/sau camera, độ sâu 0, NaN/Inf, extrinsic và Jacobian:

```powershell
.venv/Scripts/python.exe -m pytest student/tests -v -k "camera_invalid or camera_front or camera_jacobian or camera_fov or student_camera"
```

## 6. CP3 — Part F: association.py

Load Kalman bằng `load_workspace_module("kalman")`, sau đó implement năm hàm:

1. `mahalanobis_distance`: trả scalar bình phương khoảng cách `gamma.T @ S^-1 @ gamma`.
2. `chi2_gate`: dùng `chi2.ppf(gating_threshold, sensor.dim_meas)`;
   số chiều LiDAR là 3, camera là 2.
3. `association_cost_matrix`: khởi tạo inf, kiểm tra FOV **trước** projection/MHD;
   chỉ giữ chi phí vượt qua gate.
4. `pick_next_pair`: lấy cặp hữu hạn nhỏ nhất, xóa đúng hàng/cột và đối tượng;
   ma trận rỗng/toàn inf trả nan, nan và giữ danh sách chưa ghép.
5. `associate_and_update`: update cặp đã gán, gọi handle_updated_track với sensor;
   luôn kết thúc bằng manage_tracks, kể cả không có measurement.

Không predict trong association: runner đã predict một lần mỗi frame.
Không kiểm tra rồi bỏ FOV sau khi cặp đã bị xóa khỏi danh sách chưa ghép.
Camera chỉ tinh chỉnh EKF; manager đã phân quyền lifecycle theo sensor.

```powershell
.venv/Scripts/python.exe -m pytest student/tests -v -k "mahalanobis or greedy or out_of_fov or association"
```

Các test liên quan lifecycle có thể còn xfail do Part H; phải chạy lại sau CP4.

## 7. CP4 — Part H: track_management.py

Implement ba hàm:

1. `init_track_state_from_meas`: đổi đo LiDAR sang vehicle frame, vận tốc ban đầu 0;
   covariance vị trí là R xoay sang vehicle, covariance vận tốc từ bình phương
   sigma_p44/55/66; state initialized và score `1/window`.
2. `update_track_score`: hit tăng `1/window`, tối đa 1; miss giảm `1/window`.
   Xác nhận khi **score > confirmed_threshold**; track đã confirmed giữ trạng thái
   sau miss; hit của track chưa đủ ngưỡng đưa sang tentative.
3. `should_delete_track`: OR của Pxx/Pyy **> max_P**, confirmed có score
   **< delete_threshold**, hoặc chưa confirmed có score **<= 0**.

Manager chỉ gọi score khi LiDAR hit/miss trong FOV và chỉ init/delete ở lượt LiDAR.
Kiểm tra sát biên ngưỡng, camera rỗng/có hit, LiDAR rỗng và track ngoài FOV.

```powershell
.venv/Scripts/python.exe -m pytest student/tests/test_tracking_regressions.py student/tests/test_tracking_contract.py -v
.venv/Scripts/python.exe -m pytest student/tests -q
```

Đạt CP4 khi toàn bộ tests pass và không còn TODO xfail. Không sửa tests để đạt.

## 8. CP5 — Chạy Waymo và kiểm chứng kết quả

Smoke run với paths.yaml frame 0–20 và seed 0; sau đó đổi frame_end thành 198
và chạy lại cùng segment mặc định:

```powershell
.venv/Scripts/fusion-run-lab.exe --config student/config/paths.yaml --fusion compare --seed 0
```

Pipeline mỗi frame: predict một lần → LiDAR association/update/lifecycle →
camera association/update. Compare chạy hai tracker độc lập cho hai mode.

Sáu file cần sinh tự động trong `student/artifacts/`:

- metrics.json, metrics_lidar.json, metrics_fused.json.
- grade_run.log, grade_run_lidar.log, grade_run_fused.log.

Kiểm chứng trước khi viết báo cáo:

- fusion_mode compare, seed 0, frames [0,198], segment mặc định.
- Log gộp có 398 records, mỗi mode có 199 records, không trùng/mất frame.
- Mỗi record: matches + ghosts = confirmed; matches + misses = valid_gt.
- Detection của hai mode giống nhau; tổng đếm, sum_sq_err và trung bình confirmed
  tái tạo đúng metrics. RMSE 3D = sqrt(sum_sq_err/matches).
- Mục tiêu đủ chất lượng: mỗi mode RMSE <= 0.45 m,
  precision_track = matches/(matches+ghost_track_frames) >= 0.75,
  coverage = matches/det_tp >= 0.70; rmse_fused - rmse_lidar <= 0.05 m.
  Nếu mẫu số 0 hoặc RMSE null thì chưa có bằng chứng đạt chất lượng.

Nếu tests pass nhưng metrics chưa đạt, đối chiếu frame lỗi, transform, Q/R,
gating, covariance initialization và score policy; sửa đúng thuật toán rồi chạy lại.
Không thay platform/params hoặc sửa tay artifacts. Chạy lại sau lần sửa code cuối.

## 9. CP6 — Báo cáo và bàn giao

Điền `student/SUBMISSION.md`:

- Thông tin học viên, repo và commit hash theo hướng dẫn nộp.
- Số liệu detection, tracking cả hai mode lấy trực tiếp từ metrics/log.
- Giải thích RMSE cùng matches, ghost/miss; camera dùng nhãn FRONT 2D thêm nhiễu,
  nên không đánh giá chất lượng camera detector thật.
- Trả lời đủ sáu câu về z/R, gating, track-then-fuse, calibration,
  sensor tường minh ở frame rỗng và ngưỡng lifecycle; dẫn tới code/log cụ thể.
- Khai báo Codex/ChatGPT đã hỗ trợ đọc dự án và lập kế hoạch;
  bổ sung phạm vi hỗ trợ thực tế nếu tiếp tục dùng AI cho implementation/debug.
- Bonus ghi Không nếu chưa thực hiện.

Commit code theo CP1–CP4, artifacts theo CP5, báo cáo theo CP6; kiểm tra những
file được stage để tránh đưa dữ liệu, weights, .env, paths.yaml lên Git.
Checker yêu cầu artifacts đã được Git theo dõi và thay đổi trong student đã commit:

```powershell
.venv/Scripts/python.exe tools/check_submission.py
```

Sửa các vấn đề rồi commit và chạy lại cho đến `KẾT QUẢ: SẴN SÀNG NỘP`.
Push và nộp link/hash trên LMS theo hướng dẫn; xác nhận hash đã có trên remote.
Deadline lấy theo ngày học lab/thông báo coach, không suy ra từ ngày đọc repo.

## 10. Bonus tùy chọn sau phần bắt buộc

Ưu tiên trực quan hóa 2–5 ảnh (+3), export CVAT kèm kiểm tra (+3),
hoặc thí nghiệm ít nhất ba mức lệch calibration (+4) nếu còn thời gian.
Đặt bằng chứng ở student/bonus và giữ giới hạn dữ liệu theo RULES.md.
Sau thí nghiệm, chạy lại cấu hình chấm điểm chuẩn để artifacts khớp code nộp.

## Checklist kết thúc

- [ ] CP0: môi trường hoạt động, dữ liệu và weights sẵn sàng.
- [ ] CP1–CP4: đủ 17 hàm; giữ API; toàn bộ tests pass, không xfail.
- [ ] CP5: compare seed 0, frame 0–198, sáu artifacts khớp nhau.
- [ ] Đối chiếu chất lượng tracking với rubric và ghi nhận kết quả thật.
- [ ] CP6: báo cáo đủ sáu câu, khai báo AI, checkpoint commits.
- [ ] Checker báo sẵn sàng; tên repo đúng yêu cầu hoặc được coach chấp nhận.
- [ ] Push và nộp link/hash trên LMS.
