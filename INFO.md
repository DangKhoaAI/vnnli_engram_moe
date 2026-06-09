## Overview 
- Bài toán NLI : là bài toán phân loại văn bản nhằm xác định mối quan hệ logic giữa hai câu: câu giả thuyết (Premise, ký hiệu là P) và câu hệ quả (Hypothesis, ký hiệu là H).
Không gian nhãn đầu ra Y thông thường gồm 3 lớp phân biệt:
Entailment (Kéo theo/Hệ quả): Nếu P đúng thì H chắc chắn đúng.
Contradiction (Mâu thuẫn): Nếu P đúng thì H chắc chắn sai.
Neutral (Trung lập): Nếu P đúng thì không đủ cơ sở để kết luận H đúng hay sai.

- Model : Finetuning model 
### Data infomation 
The task of Adversarial Natural Language Inference in the Vietnamese language aims
to rigorously evaluate and enhance the robustness of language models in discerning
logical semantic relationships based on complex and adversarial linguistic constructs.
This task is defined as follows.
• Input: A pair of Vietnamese sentences, comprising a premise, extracted from di-
verse news articles, and a corresponding hypothesis, meticulously crafted to probe
model reasoning capabilities.
• Output: A label of the logical relationship between the premise and hypothesis into
one of three classes: entailment, where the hypothesis is logically derivable from
the premise; contradiction, where the hypothesis negates the premise; or neutral,
where the hypothesis neither supports nor contradicts the premise.

ViANLI:
- Train : 8012 
- Dev : 1000 
- Test : 1000 
- Total :  10012 

Dataset Column : 
- uid
- premise 
- hypothesis 
- label :

1 Sample : 
- uid: uit_Adver_365_3_11_02
- premise : Tọa đàm do Tổng cục Du lịch phối hợp với báo điện tử VnExpress tổ chức ngày 3/4 tại FLC Sầm Sơn, Thanh Hóa.
- hypothesis : Đầu tháng 4 có một buổi gặp mặt trao đổi, nói chuyện thân mật của Tổng cục Du lịch.
- label : entailment

## Model Information 
Các model được chọn để finetuning gồm 
including mBERT [52], XLM-R [19], CafeBERT [62], and PhoBERT [60]. The mBERT, XLM-R, and
CafeBERT models are powerful multilingual transformer models trained on large doc-
ument corpora, while PhoBERT is a monolingual transformer model developed specif-
ically for Vietnamese. Since Vietnamese uses both single words (one token) and com-
pound words (multiple tokens), we use the VnCoreNLP tool [72] to tokenize the Viet-
namese text, ensuring proper word segmentation for PhoBERT input.

| Model          |             Variant / checkpoint |                                                          Số tham số | Ghi chú                                                                                                                                                    |
| -------------- | -------------------------------: | ------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **mBERT**      |   `bert-base-multilingual-cased` |      khoảng **179M**; checkpoint thực tế thường tính ra **177.85M** | 12 layers, hidden 768, 12 heads; 104 ngôn ngữ. HF docs ghi 179M; issue từ repo BERT tính trực tiếp ra 177,853,440. ([Hugging Face][1])                     |
| **mBERT**      | `bert-base-multilingual-uncased` |                                                     khoảng **168M** | 12 layers, hidden 768, 12 heads; 102 ngôn ngữ. ([Hugging Face][1])                                                                                         |
| **XLM-R**      |               `xlm-roberta-base` | khoảng **270M** trong paper; thường tính checkpoint khoảng **278M** | 12 layers, hidden 768, 12 heads, vocab 250k. Paper ghi 270M. ([arXiv][2])                                                                                  |
| **XLM-R**      |              `xlm-roberta-large` | khoảng **550M** trong paper; thường tính checkpoint khoảng **559M** | 24 layers, hidden 1024, 16 heads, vocab 250k. ([arXiv][2])                                                                                                 |
| **XLM-R**      |                       `XLM-R XL` |                                                     khoảng **3.5B** | Variant mở rộng trong nghiên cứu scaling XLM-R, không phải variant phổ biến bằng base/large.                                                               |
| **XLM-R**      |                      `XLM-R XXL` |                                                    khoảng **10.7B** | Variant mở rộng rất lớn, chủ yếu dùng trong nghiên cứu.                                                                                                    |
| **CafeBERT**   |                `uitnlp/CafeBERT` |                                                     khoảng **560M** | Model card nói CafeBERT dựa trên XLM-RoBERTa; checkpoint public tương ứng kiến trúc cỡ XLM-R-large, nên số tham số xấp xỉ XLM-R-large. ([Hugging Face][3]) |
| **PhoBERT**    |             `vinai/phobert-base` |                                                     khoảng **135M** | RoBERTa-base style: 12 layers, hidden 768, 12 heads, vocab 64k. PhoBERT paper/repo có hai bản base và large. ([GitHub][4])                                 |
| **PhoBERT**    |            `vinai/phobert-large` |                                                     khoảng **370M** | RoBERTa-large style: 24 layers, hidden 1024, 16 heads, vocab 64k. ([GitHub][4])                                                                            |
| **PhoBERT v2** |          `vinai/phobert-base-v2` |                                                     khoảng **135M** | Vẫn là base-sized PhoBERT; khác chủ yếu ở dữ liệu/tiền huấn luyện, không phải scale kiến trúc. ([Hugging Face][5])                                         |



mBERT, XLM-R, CafeBERT, and PhoBERT during model training are designed to optimize performance for most models after evaluation. The basic parameters are as follows: max_length=256, learning_rate=1e-05,
eval_frequency=400, batch_size=16, weight_decay=0.0, adam_epsilon=1e-08, dropout=0.4.

## Git 
project has git 
- local : avaiable 
- cloud : in https://github.com/DangKhoaAI/vnnli_engram_moe


## Môi trường 
thư viện kể  trên là của production kaggle GPU  RTX 6000 Pro  , đây là môi trường dev local , local có GPU  RTX 5050 linux ubuntu , đã cài driver , cuda , chạy được nvidia-smi  