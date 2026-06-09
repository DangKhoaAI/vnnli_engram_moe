Đọc INFO.md để  hiểu về data bài toán NLI , các base model cần finetuning và Git

Task : viêt một source code finetuning base model cho bài toán NLI
Yêu cầu :

1. môi trường
   Python version: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
   PyTorch version: 2.10.0+cu128
   Transformers version: 5.0.0
   NOTE : thư viện kể  trên là của production kaggle GPU  RTX 6000 Pro  , đây là môi trường dev local , local có GPU  RTX 5050 linux ubuntu 
2. tạo môi trường : tạo uv ở dir hiện tại , cài thư viện vào trong để run code thử nghiệm 
3. Phương pháp finetuning

 Design 1: sử dụng FFN 
Cách các mô hình kiến trúc họ BERT (như mBERT, PhoBERT) được tùy chỉnh để thực hiện NLI tuân theo quy trình tiêu chuẩn sau (tương tự như nguyên lý tác giả áp dụng ở lớp cuối của mô hình NLIMoE ):  Đầu vào (Input Representation): Câu Premise và Hypothesis được ghép lại thành một chuỗi duy nhất, ngăn cách bởi các token đặc biệt (ví dụ: [CLS] Premise [SEP] Hypothesis [SEP]).Trích xuất đặc trưng (Pooling): Mô hình sử dụng vector biểu diễn của token [CLS] ở lớp ẩn cuối cùng (last hidden state). Vector này đại diện cho toàn bộ thông tin ngữ nghĩa của cặp câu.Lớp phân loại (Classification Head): Vector [CLS] được đưa qua một lớp kết nối đầy đủ (Fully Connected/Linear Layer) mới được khởi tạo ngẫu nhiên. Lớp này được cấu hình số lượng nút đầu ra (output dimension) chính xác bằng 3.Dự đoán: Một hàm Softmax được áp dụng lên đầu ra của lớp Linear để tính toán xác suất cho 3 nhãn độc lập: Entailment, Contradiction và Neutral. 

In our experiments, we focus on using state-of-the-art transformer models, including
mBERT [52], XLM-R [19], CafeBERT [62], and PhoBERT [60].
The mBERT, XLM-R, and CafeBERT models are powerful multilingual transformer models trained on large doc-
ument corpora, while PhoBERT is a monolingual transformer model developed specif-
ically for Vietnamese. Since Vietnamese uses both single words (one token) and com-
pound words (multiple tokens), we use the VnCoreNLP tool [72] to tokenize the Viet-
namese text, ensuring proper word segmentation for PhoBERT input.

Additionally, the hyperparameters selected for mBERT, XLM-R, CafeBERT, and PhoBERT
during model training are designed to optimize performance for most models after eval-
uation. The basic parameters are as follows: max_length=256, learning_rate=1e-05,
eval_frequency=400, batch_size=16, weight_decay=0.0, adam_epsilon=1e-08, dropout=0.4.
Furthermore, we set epochs=7 since this yielded the best accuracy for the models. Re-
ducing the number of epochs resulted in suboptimal performance, while increasing the
epochs led to either saturation or a decline in model accuracy.

Design 2 :  sử dụng MoE architecture  (future implement )
A routing network R (𝑥) assigns each token 𝑥 to
the top-𝑘 most relevant experts. The MoE output is the weighted sum of these selected experts:
MoE(𝑥) =
𝑘
∑︁
R (𝑥)𝑖 · 𝐸 𝑖 (𝑥)
(4)
𝑖=1
Within each expert 𝐸 𝑖 , the traditional ReLU/GELU activations are replaced with the SwiGLU activation
function to enhance non-linear representational capacity. For an input projection, the SwiGLU operation is
defined as:
SwiGLU(𝑥) = (Swish(𝑥𝑊) ⊙ 𝑥𝑉)𝑊2
(5)
where 𝑊, 𝑉 ∈ R𝑑×𝑑 𝑓 𝑓 and 𝑊2 ∈ R𝑑 𝑓 𝑓 ×𝑑 are learnable weight matrices, and ⊙ denotes element-wise
multiplication

Design 3 :  sử dụng  Engram + MoE architecture  (future implement ) 
- đây là một thay đổi về kiến trúc , đang nghiên cứu , tuy nhiên , hãy viết code handling sẵn chuyện này , trách refactor lớn về sau 

4. về việc training 
- tạo endpoint  cho với project repo này , tôi mở kaggle , mở chạy lệnh gitclone sẽ clone dự án vào folder /kaggle/working/{repo} 
- viết tôi một "wrapup notebook"  ; tôi upload lên kaggle , chạy các shell nó sẽ tự clone repo dự án này  , tự chạy các setup , training , và ra output 
5. có testing để code đảm bảo chạy 
6. các tham số sẽ được centralize tại một nơi, một file nào đó để thuận tiện cho hyper tuning 
7. có documentation : README mô tả thông tin dự án , thư mục docs/ chứa PROEJECT.md mô tả về project overview  , Architecture and Tech Stack , Directory structure , Module / Function List) , file STATE.md lưu tiến độ dự án  , bạn có thể thêm các file cần thiết khác 