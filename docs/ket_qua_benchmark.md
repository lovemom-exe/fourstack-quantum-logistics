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
| 6 | 24 | *đang chạy* | | | | | | |
| 8 | 32 | *đang chạy* | | | | | | |

Objective 3 restart tại k=4: `{seed 0: 0,204808, seed 1: 0,189893, seed 2: 0,216478}`
→ chọn seed 1 theo validation MSE.

**Nhận xét k=4:** R² âm và **r ≈ −0,04**, tức mô hình gần như **không có tương quan**
với thực tế — VQR ở 4 qubit chưa học được gì đáng kể. So sánh: XGBoost cùng giao thức Q,
cùng k=4 đạt R² 0,0909.

Model đã lưu: `vqr_weights_k4.npy`, `vqr_artifacts_k4.joblib`.

---

## 6. Hạn chế và điều chưa làm được

1. **Test set từng quá nhỏ, gây thổi phồng ~2,4 lần.** XGBoost giao thức Q (n=1.000,
   k=4) đo được R² 0,2144 trên test set cũ ~1.000 dòng, nhưng chỉ **0,0909** trên test
   set 10.000 dòng. Mọi số trước khi chuyển sang test 10k đều không đáng tin.

2. **Model VQR k=4 lần đầu bị mất.** Run đầu không lưu weights/scaler, tiến trình thoát
   là mất sạch, phải train lại từ đầu. Đã bổ sung `_persist()` — từ nay mọi run VQR đều
   lưu weights + selector + 2 scaler + clip threshold ngay sau khi chọn restart.

3. **VQR chưa vượt được classical.** Ở k=4, VQR có R² âm (−0,1059) và r ≈ 0, thua rõ
   XGBoost cùng điều kiện (0,0909). Chưa có bằng chứng nào cho thấy lợi thế lượng tử
   trên bài toán này ở quy mô hiện tại.

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

---

## Phụ lục — file kết quả

**Model đã lưu:** `protocol_p_xgboost.joblib` · `vqr_weights_k4.npy` ·
`vqr_artifacts_k4.joblib`

**Bảng số:** `ablation_winsorize.csv` · `ksweep_results.csv` ·
`vqr_ksweep_results.csv` · `protocol_p_metrics.json` · `demo_error_stats.json`

**Biểu đồ:** `ksweep_r2_real.png` · `ksweep_r2_log.png` ·
`demo_actual_vs_pred_slice.png` · `demo_error_hist.png` · `demo_scatter.png`

Tất cả nằm trong `ml/training/evaluation_result/`.
