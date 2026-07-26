# Kết quả benchmark — dự báo nhu cầu (fresh retail)

> Mọi số trong file này chạy hoàn toàn cục bộ (`StatevectorEstimator`), **không có
> cuộc gọi QuApp cloud nào**. Credit còn lại dành riêng cho `quapp_proof_run.py`.

**Test set dùng chung cho toàn bộ file:** 10.000 dòng lấy từ
`data/features/fresh_retail_eval_data.csv`, seed `2024`. Train/validation lấy từ
file train riêng nên không giao nhau. Thống kê target test: trung bình 1,2091 ·
trung vị 0,8 · p99 8,0 · max 41,8 · 4,0% giá trị bằng 0 (phân phối lệch phải, đuôi nặng).

---

## 1. Hai giao thức xử lý target — và vì sao phải tách

Trước đây hai cách xử lý target bị trộn lẫn khiến các con số không so sánh được.
Từ nay **mọi bảng đều ghi rõ thuộc giao thức nào**, và không bao giờ để số của hai
giao thức trong cùng một bảng so sánh.

| | **Giao thức P** (số sản phẩm) | **Giao thức Q** (so với quantum) |
|---|---|---|
| Xử lý target | `log1p` → `expm1` | winsorize train p99 → `log1p` → `MinMaxScaler(-1,1)` → đảo ngược |
| Winsorize | **Không** | **Có** (p99) |
| Scale target | Không | Có, về `[-1,1]` |
| Dùng khi nào | Báo cáo baseline classical lên deck | So sánh công bằng với VQR |
| Vì sao cần | Không cắt đuôi phân phối | **Bắt buộc**: output VQR bị chặn trong `[-1,1]` |

Giao thức Q không phải lựa chọn thẩm mỹ mà là **ràng buộc vật lý**: VQR đọc ra
giá trị kỳ vọng của toán tử Pauli-Z, luôn nằm trong `[-1,1]`. Muốn VQR biểu diễn
được target thì phải đưa target vào đúng dải đó. Nhưng chính phép biến đổi ấy
làm hỏng khả năng dự báo đuôi — nên **không được dùng số giao thức Q làm số sản phẩm**.

---

## 2. Ablation winsorize — nguyên nhân chênh lệch 0,39 vs 0,20

XGBoost, pool leakage-free, chấm trên cùng test set 10.000 dòng:

| Giao thức | n_train | k | winsorize | R²(real) | R²(log) | MAE | RMSE | std ratio | r | max(y_pred) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Q | 50.000 | 15 | Có | 0,2041 | 0,1214 | 0,7594 | 1,8287 | 0,2280 | 0,6418 | 7,09 |
| **P** | 50.000 | 15 | Không | **0,4019** | 0,1508 | 0,7299 | 1,5853 | 0,4111 | 0,7344 | 17,69 |
| Q | 1.000 | 4 | Có | 0,0842 | −0,2424 | 0,9036 | 1,9616 | 0,3878 | 0,3397 | 7,57 |
| P | 1.000 | 4 | Không | 0,2397 | −0,2425 | 0,8914 | 1,7874 | 0,5423 | 0,5122 | 20,11 |

**ΔR² do winsorize:**

| Cấu hình | ΔR²(real) | ΔR²(log) | max(y_pred) |
|---|---:|---:|---|
| n=50.000, k=15 | **+0,1978** | +0,0294 | 7,09 → 17,69 |
| n=1.000, k=4 | **+0,1555** | −0,0001 | 7,57 → 20,11 |

### Kết luận

**Winsorize chính là nguyên nhân, không phải rò rỉ dữ liệu.** Ba bằng chứng:

1. Bỏ winsorize đưa R² từ 0,2041 lên **0,4019** — đúng dải 0,35–0,40 đã dự đoán,
   khớp với con số 0,3938 báo cáo trước đây.
2. **R²(log) gần như không đổi** (+0,029 và −0,0001). Nghĩa là winsorize hầu như
   không ảnh hưởng tới việc học cấu trúc trong không gian log; nó chỉ phá khả
   năng chạm tới đuôi khi quy về đơn vị thật.
3. `max(y_pred)` bị chặn ở ~7,1–7,6 khi winsorize (do target train bị cắt ở
   ngưỡng 5,80 / 6,505 trước khi `log1p`), trong khi test có giá trị tới **41,8**.
   R² đơn vị thật bị chi phối bởi chính cái đuôi đó.

### ⚠️ Đính chính một kết luận sai trước đây

Ở lần chạy trước tôi kết luận *"0,3938 cao là do rò rỉ cùng ngày (stockout)"* —
**kết luận này sai chiều và đã được rút lại**. Thực tế con số 0,3938 là của bản
**leakage-free**; bản full (có cột stockout cùng ngày) chỉ đạt 0,2976 theo run
`train_xgboost.py` trước đó. Tức là thêm cột stockout cùng ngày làm kết quả
**tệ đi**, không phải tốt lên. Nguyên nhân chênh lệch 0,39 vs 0,20 là winsorize,
đúng như bảng ablation ở trên.

*(Ghi chú trung thực: con số 0,2976 của bản full lấy từ run trước, tôi chưa đo lại
trong phiên này.)*

---

## 3. Số baseline sản phẩm — Giao thức P

Cấu hình: XGBoost, pool leakage-free, `n_train = 50.000`, `k = 15`, `log1p → expm1`,
không winsorize. Model đã lưu: `ml/training/evaluation_result/protocol_p_xgboost.joblib`.

| Chỉ số | Giá trị |
|---|---:|
| **R² (đơn vị thật)** | **0,4019** |
| R² (log space) | 0,1508 |
| MAE | 0,7299 |
| RMSE | 1,5853 |
| MAPE | 77,67% |
| std(y_pred)/std(y_true) | 0,4111 |
| Pearson r | 0,7344 |
| max(y_pred) | 17,69 (y_true max = 41,80) |

**Đây là con số classical để đưa lên deck: R² = 0,40.**

---

## 4. Lưới quét k — Giao thức Q

Mục đích: tách ảnh hưởng của **số feature** và **lượng dữ liệu**. XGBoost, pool
leakage-free, cùng test set 10.000 dòng.

| n_train | k | R²(real) | R²(log) | MAE | RMSE | std ratio | r |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 4 | 0,0909 | −0,2206 | 0,8970 | 1,9545 | 0,3817 | 0,3461 |
| 1.000 | 6 | 0,1372 | 0,0128 | 0,7964 | 1,9040 | 0,2199 | 0,5220 |
| 1.000 | 8 | 0,1367 | 0,0245 | 0,7836 | 1,9046 | 0,2037 | 0,5659 |
| 1.000 | 12 | 0,1581 | 0,0525 | 0,7840 | 1,8809 | 0,2146 | 0,5669 |
| 1.000 | 15 | 0,1459 | 0,0407 | 0,7809 | 1,8944 | 0,2046 | 0,5780 |
| 50.000 | 4 | 0,1645 | 0,0854 | 0,7720 | 1,8737 | 0,2040 | 0,5959 |
| 50.000 | 6 | 0,1844 | 0,1178 | 0,7589 | 1,8513 | 0,2062 | 0,6412 |
| 50.000 | 8 | 0,2064 | 0,1104 | 0,7629 | 1,8261 | 0,2320 | 0,6405 |
| 50.000 | 12 | 0,2032 | 0,1280 | 0,7605 | 1,8298 | 0,2214 | 0,6408 |
| 50.000 | 15 | 0,2003 | 0,1168 | 0,7601 | 1,8331 | 0,2276 | 0,6351 |

PNG: `ml/training/evaluation_result/ksweep_r2_real.png` ·
`ml/training/evaluation_result/ksweep_r2_log.png` ·
CSV: `ml/training/evaluation_result/ksweep_results.csv`

### Diễn giải

- **Độ dốc theo k**: n=1.000 từ k=4→12 tăng +0,067; n=50.000 từ k=4→8 tăng +0,042.
- **Khoảng cách giữa hai đường n** (cùng k): trung bình +0,058.

Hai hiệu ứng **xấp xỉ ngang nhau** — không cái nào áp đảo. Nhưng chi tiết quan trọng hơn:
gần như toàn bộ lợi ích của k nằm ở đoạn **k=4 → k=6**; sau k=6 đường gần như phẳng
(dao động ±0,02, nhiều khả năng là nhiễu). Riêng k=4 là điểm tệ nhất trên cả hai đường
và là nơi khoảng cách dữ liệu lớn nhất (+0,074).

→ **Với VQR bị chặn ở 4 qubit, k=4 là vị trí đặc biệt bất lợi** — nằm ngay trước khúc
cua dốc nhất của đường cong.

> ⚠️ Đây là **ablation chẩn đoán**, không phải chọn model. Toàn bộ đường cong được báo
> cáo. Không được chọn k tốt nhất trên test rồi lấy đó làm "kết quả".

---

## 5. VQR theo số qubit — Giao thức Q

Cấu hình đóng băng, chỉ k thay đổi: `zz_feature_map(k, reps=2)`,
`real_amplitudes(k, reps=3)`, `COBYLA(maxiter=300)`, 3 restart có seed chọn theo
validation, n_train = 800 (1.000 trừ 20% validation), cùng cách xử lý target Q.

| k (qubit) | tham số | R²(real) | R²(log) | MAE | RMSE | std ratio | r | evals (chạm trần 300?) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 16 | **−0,1059** | −0,4519 | 1,0737 | 2,1557 | 0,2776 | **−0,0408** | 165/192/154 — không |
| 5 | 20 | −0,0385 | −0,4126 | 1,0820 | 2,0890 | 0,2172 | 0,0603 | 204/208/263 — không |
| 6 | 24 | −0,0413 | −0,4683 | 1,1256 | 2,0918 | 0,1169 | 0,0229 | 269/236/**300** — **có** |
| 7 | 28 | **−0,0307** | **−0,2932** | 1,0005 | 2,0811 | 0,2179 | 0,0438 | 272/**300**/**300** — **có** |

Objective 3 restart (chọn theo validation MSE, **không bao giờ theo test**):

| k | seed 0 | seed 1 | seed 2 | seed được chọn |
|---:|---:|---:|---:|---:|
| 4 | 0,204808 | 0,189893 | 0,216478 | 1 |
| 5 | 0,264639 | 0,247298 | 0,236148 | 2 |
| 6 | 0,262426 | 0,259183 | 0,243764 | 2 |
| 7 | 0,240031 | 0,240659 | 0,250418 | 0 |

Thời gian train (StatevectorEstimator cục bộ, 3 restart mỗi k): k=5 ≈ **26,1 phút**,
k=7 ≈ **64,9 phút**. Tổng phiên bổ sung 93,6 phút. k=4 và k=6 được **skip**, đọc lại
từ `vqr_ksweep_results.json` nên số cũ không bị chạy lại hay ghi đè.

### Diễn giải

- **k=4 là điểm tệ nhất** và là điểm duy nhất có **r âm** (−0,0408): mô hình gần như
  không tương quan với thực tế. So sánh: XGBoost cùng giao thức Q, cùng k=4 đạt 0,0909.
- **Toàn bộ 4 điểm đều có R² âm** ở cả đơn vị thật lẫn không gian log. Nghĩa là ở mọi
  k đã đo, VQR vẫn **thua baseline đoán trung bình**.
- **Xu hướng KHÔNG đơn điệu.** R²(real) đi −0,1059 → −0,0385 → −0,0413 → −0,0307:
  tăng mạnh từ k=4→5 rồi **gãy tại k=6** (giảm nhẹ), sau đó tăng lại. R²(log) gãy
  cùng chỗ, còn rõ hơn: −0,4519 → −0,4126 → −0,4683 → −0,2932. Biên độ dao động (~0,01 ở real,
  ~0,06 ở log) **cùng cỡ với khoảng cách giữa các điểm**, nên phần lớn "xu hướng" ở
  đây có thể chỉ là nhiễu tối ưu hoá.
- **Ngân sách tối ưu hoá đã chạm trần từ k=6.** k=4 và k=5 hội tụ trong 154–263 evals,
  nhưng k=6 và k=7 có restart chạm đúng trần `maxiter=300`. Tức là từ k≥6, **COBYLA bị
  cắt giữa chừng chứ không phải hội tụ** — số của k=6 và k=7 là cận dưới, và một phần
  hình dạng đường cong phản ánh ngân sách tối ưu hoá chứ không chỉ sức biểu diễn của
  mạch. Muốn kết luận sạch về scaling thì phải nới `maxiter` trước.
- **std ratio 0,12–0,28** ở mọi k: mô hình co mạnh về trung bình, giống toàn bộ các
  model khác trong file này.

Model đã lưu: `vqr_weights_k{4,5,6,7}.npy`, `vqr_artifacts_k{4,5,6,7}.joblib`.

---

## 5b. Dự phóng scaling — ⚠️ NGOẠI SUY, KHÔNG PHẢI SỐ ĐO

Biểu đồ: `ml/training/evaluation_result/vqr_trend_r2_real.png` ·
`ml/training/evaluation_result/vqr_trend_r2_log.png`
(sinh bởi `ml/training/plot_vqr_trend.py`)

**Ranh giới đo thật / dự phóng:**

| | Phần ĐO THẬT | Phần DỰ PHÓNG |
|---|---|---|
| Khoảng k | **k = 4, 5, 6, 7** | k > 7, kéo tới k = 12 |
| Nguồn | `vqr_ksweep_results.csv` — chạy thật, test set 10.000 dòng | `np.polyfit` bậc 1 trên đúng 4 điểm đo |
| Cách vẽ | marker vuông, nét liền, màu đậm | nét đứt, màu nhạt, kèm dải ±1 s.e. dự báo |
| Được dùng làm gì | kết quả benchmark, trích dẫn được | **chỉ là phép ngoại suy — không trích dẫn như số đo** |

**Kết quả fit tuyến tính:**

| Chỉ số | Hệ số góc / k | Chặn | std phần dư | Cắt y=0 tại | Giá trị dự phóng tại k=12 |
|---|---:|---:|---:|---:|---:|
| R²(real) | +0,02229 | −0,17669 | 0,0240 | **k ≈ 7,9** | +0,0908 ± 0,0748 |
| R²(log) | +0,04203 | −0,63768 | 0,0704 | không cắt tới k=12 | −0,1333 ± 0,2192 |

### Giả định của phép ngoại suy — và vì sao nên nghi ngờ nó

Con số "k ≈ 7,9" **chỉ đúng nếu tất cả các giả định sau đều đúng**:

1. **Xu hướng tiếp tục tuyến tính** ngoài khoảng đã đo. Chưa có gì bảo đảm; các đường
   cong k-sweep của XGBoost ở mục 4 **bão hoà sau k≈6–8** chứ không tăng tuyến tính.
   Nếu VQR bão hoà giống vậy thì đường dự phóng sai hẳn.
2. **Giữ nguyên ansatz/optimizer/ngân sách**: `real_amplitudes(reps=3)`,
   `COBYLA(maxiter=300)`, 3 restart, n_train = 800. Số tham số tăng tuyến tính theo k
   (4k), nên ở k lớn hơn **maxiter=300 sẽ càng thiếu** — mà k=6 và k=7 đã chạm trần rồi.
   Giả định này gần như chắc chắn **vỡ** trước k=12.
3. **Fit chỉ dựa trên 4 điểm, dof = 2**, trong đó dãy đo **không đơn điệu** (gãy tại
   k=6). Dải ±1 s.e. tại k=12 đã rộng ±0,075 (real) và ±0,219 (log) — riêng dải bất
   định ở log đã **rộng hơn toàn bộ khoảng biến thiên đo được**.
4. **Bỏ qua barren plateau.** Mạch biến phân sâu hơn/rộng hơn thường **khó tối ưu hơn**,
   không dễ hơn. Ngoại suy tuyến tính giả định ngầm điều ngược lại.

> ⚠️ **Không được trình bày "VQR đạt R² ≥ 0 tại k ≈ 8" như một kết quả.** Đó là điểm
> cắt của một đường thẳng fit qua 4 điểm âm, chưa có một phép đo nào ở k ≥ 8 xác nhận.
> Cách phát biểu trung thực: *"trong khoảng k=4..7 đã đo, R² tăng dần nhưng vẫn âm ở
> mọi k; nếu xu hướng tuyến tính tiếp diễn — điều chưa được kiểm chứng — thì mốc hoà
> vốn với baseline rơi vào khoảng k≈8."*

**Cách kiểm chứng:** chạy thật k=8 (và k=9) với `maxiter` nới rộng đủ để không chạm
trần. Nếu k=8 rơi ra ngoài dải bất định, giả định tuyến tính bị bác bỏ.

---

## 6. Hạn chế và điều chưa làm được

1. **Test set từng quá nhỏ, gây thổi phồng ~2,4 lần.** XGBoost giao thức Q (n=1.000,
   k=4) đo được R² 0,2144 trên test set cũ ~1.000 dòng, nhưng chỉ **0,0909** trên test
   set 10.000 dòng. Mọi số trước khi chuyển sang test 10k đều không đáng tin.

2. **Model VQR k=4 lần đầu bị mất.** Run đầu không lưu weights/scaler, tiến trình thoát
   là mất sạch, phải train lại từ đầu. Đã bổ sung `_persist()` — từ nay mọi run VQR đều
   lưu weights + selector + 2 scaler + clip threshold ngay sau khi chọn restart.

3. **VQR chưa vượt được classical ở bất kỳ k nào đã đo.** Cả 4 điểm k=4,5,6,7 đều có
   R² âm (−0,1059 … −0,0307), tức thua cả baseline đoán trung bình, và thua rõ XGBoost
   cùng giao thức Q. Chưa có bằng chứng nào cho thấy lợi thế lượng tử trên bài toán này
   ở quy mô hiện tại. Mốc "hoà vốn k≈8" ở mục 5b là **ngoại suy, không phải số đo**.

4. **Chưa có feature trễ (lag).** Toàn bộ feature hiện tại là lịch/thời tiết/khuyến mãi
   của **chính ngày cần dự báo**, không có doanh số các ngày trước. Vì vậy đây là bài
   toán **nowcasting**, chưa phải forecasting đúng nghĩa. Đây gần như chắc chắn là
   nguyên nhân lớn nhất khiến trần R² thấp — thêm lag feature là hướng cải thiện đáng
   giá nhất, hơn hẳn việc tăng số qubit.

5. **Mọi model đều co mạnh về giá trị trung bình.** std(y_pred)/std(y_true) chỉ nằm
   trong khoảng 0,20–0,41 ở tất cả cấu hình. Kể cả baseline P tốt nhất cũng chỉ tái tạo
   41% độ phân tán thật, và bỏ lỡ gần như toàn bộ các đỉnh (xem
   `demo_actual_vs_pred_slice.png`). Với target zero-inflated đuôi nặng cộng loss bình
   phương, đây là hành vi dự đoán được.

6. **Sai lệch có hệ thống theo hướng dự báo thiếu.** Baseline P có bias **−0,372**,
   tỷ lệ dự báo thiếu **51,5%**, mức thiếu trung bình **1,069**. Trong bối cảnh
   cold-chain: dự báo thiếu → đặt hàng thiếu → hết hàng. Đây là rủi ro nghiệp vụ cần
   nêu rõ, không nên giấu sau con số R².

7. **Chưa đo lại bản "full"** (có cột stockout cùng ngày) trong phiên này; con số 0,2976
   trích từ run trước.

8. **Ngân sách optimizer chạm trần từ k=6.** `COBYLA(maxiter=300)` bị cắt giữa chừng ở
   k=6 và k=7 (có restart dừng đúng 300 evals). Số của hai k này là **cận dưới**, và
   chưa tách được đâu là giới hạn của mạch, đâu là giới hạn của ngân sách tối ưu hoá.
   Muốn quét k cao hơn thì phải nới `maxiter` trước, nếu không mọi kết luận về scaling
   đều lẫn hai hiệu ứng.

9. **Chưa đo k ≥ 8 cho VQR.** Lưới hiện tại dừng ở k=7. Mọi phát biểu về k=8..12
   (mục 5b) là ngoại suy từ 4 điểm, chưa có phép đo nào xác nhận.

---

## Phụ lục — file kết quả

**Model đã lưu:** `protocol_p_xgboost.joblib` ·
`vqr_weights_k4.npy` · `vqr_weights_k5.npy` · `vqr_weights_k6.npy` · `vqr_weights_k7.npy` ·
`vqr_artifacts_k4.joblib` · `vqr_artifacts_k5.joblib` · `vqr_artifacts_k6.joblib` ·
`vqr_artifacts_k7.joblib`

**Bảng số:** `ablation_winsorize.csv` · `ksweep_results.csv` ·
`vqr_ksweep_results.csv` · `vqr_ksweep_results.json` · `protocol_p_metrics.json` ·
`demo_error_stats.json`

**Biểu đồ:** `ksweep_r2_real.png` · `ksweep_r2_log.png` ·
`vqr_trend_r2_real.png` · `vqr_trend_r2_log.png` ·
`vqr_vs_xgb_r2_real.png` · `vqr_vs_xgb_r2_log.png` ·
`demo_actual_vs_pred_slice.png` · `demo_error_hist.png` · `demo_scatter.png`

Tất cả nằm trong `ml/training/evaluation_result/`.
