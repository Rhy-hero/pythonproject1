# Project 1：NumPy 图像编码器复现（完成版）

本版本补全学生包 `image_encoder.py` 的 TODO-01 至 TODO-06，使用固定参数完成 OpenCLIP ViT-B-32 图像塔前向计算。输入为有限的 `float32 [B,3,224,224]`（B 为正整数），输出为归一化的 `float32 [B,512]`。模型推理不读取参考答案、不训练参数、不需要 GPU、PyTorch、OpenCLIP 包或网络。

## 1. 放置和运行

**先解压老师的 `project1-学生版 (2).zip`，再把本提交包中的文件复制到其 `project1-numpy-student/` 目录内。** 已提供的 `weights/` 和 `assets/` 保持原样。本提交包按作业要求不重复包含大型权重、测试图片与参考数据，因此不能脱离老师原包直接运行模型。

建议使用 Python 3.11 或更新版本（原课程程序最低要求 3.9；本次实测 3.12.14）：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output results/output_tokens.npz
```

自选图片：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

每一行输出与输入顺序一致，重复图片保留为重复行。命令行会保存 `ids` 和 `output_tokens` 两个数组。预处理由原包提供，包括短边缩放、中心裁剪、RGB 转换和固定通道归一化。输出目录相对当前命令行目录；模型资源相对脚本目录。

## 2. 完成的六组实现

| 题号 | 方法 | 核心计算 |
|---|---|---|
| TODO-01 | `ImageEncoder.patchify` | `[B,3,7,32,7,32]` 换轴为 `[B,7,7,3,32,32]`，展平得到 `[B,49,3072]` |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,768] → [B,N,12,64] → [B,12,N,64]` |
| TODO-03 | `MultiHeadAttention.__call__` | QK 转置内积、除以 √64、注意力系数乘 V、合并多头与输出投影 |
| TODO-04 | `TransformerBlock.__call__` | 两个残差连接与 `768 → 3072 → 768` 的 GELU MLP |
| TODO-05 | `ImageEncoder.embed` | 无偏置 patch 投影、前置 CLS、相加位置嵌入 |
| TODO-06 | `ImageEncoder.__call__` | 第 0 行 CLS、post-Norm、`visual.proj` 正向投影、单位化 |

`visual.proj` 为 `[768,512]`，不要转置；其他线性参数按 `[out,in]` 存储，要使用 `.T`。输入预处理、参数加载、数值辅助函数和教师检查器均未改动。

## 3. 实测验收

环境：Python 3.12.14、NumPy 2.3.5、Pillow 12.3.0、imageio 2.38.2；CPU 运行。为避免过量线程，本次运行设置了 `OPENBLAS_NUM_THREADS=2`、`OMP_NUM_THREADS=2`，这是性能选项，不是模型计算的依赖条件。

| 阶段 | 检查形状 | 最大绝对误差 | 平均绝对误差 |
|---|---|---:|---:|
| preprocess | `(10,3,224,224)` | 0 | 0 |
| patches | `(2,49,3072)` | 0 | 0 |
| tokens | `(2,50,768)` | 4.387e-5 | 1.335e-6 |
| block0 | `(2,50,768)` | 7.147e-6 | 3.348e-7 |
| last_tokens | `(2,50,768)` | 4.196e-5 | 2.633e-6 |
| output | `(10,512)` | 5.662e-7 | 8.688e-8 |
| batch | `(3,512)` | 4.061e-7 | 7.848e-8 |

资产完整性及以上七个阶段全部 PASS。教师判据为逐元素 `abs(actual-reference) <= atol + rtol*abs(reference)`，`rtol=2e-4`；预处理 `atol=2e-6`，最终输出与批次 `atol=2e-5`，其余 `atol=2e-4`。浮点最后几位可能因 BLAS/平台变化，判断是否通过以该逐元素判据为准。

原始检查记录是 `results/check_all.txt`。`output_tokens.npz` 是本实现新计算的 10×512 输出，不是提供的参考数组。

## 4. 补充验证与错误定位

```bash
python -m unittest discover -s tests -v
python debug_shape_bug.py
python validate_additional.py
python practice_solutions/numpy_homework.py
```

- 6 项独立小型回归测试覆盖 patch 顺序、多头索引、注意力数值、因果屏蔽、残差及四维转置。
- `validate_additional.py` 需要先运行上面的 `run.py`。它生成一张参考集之外的 RGB 渐变图片，验证批次换序/重复一致、输出为有限单位向量且受输入变化影响；同时核对命令行输出与 ID 顺序。
- `debug_shape_bug.py` 是主动构造的错误复现实验：对 `[B,H,N,64]` 的 K 使用 `K.T` 会颠倒全部轴并导致矩阵乘法失败；修正为 `K.swapaxes(-1,-2)`。该实验不声称教师原框架存在这条错误实现。
- `practice_solutions/numpy_homework.py` 把之前的 12 道练习合并为题目要求的独立单文件答案，保留 `angle_deg`、`unit_vector`、`angles_to`、`top_k_cosine` 函数名。12 题输出逐题与原代码一致，记录见 `results/practice_equivalence.txt`。

## 5. 报告与 Overleaf

报告 PDF 和 Overleaf 工程单独提供，本代码仓库不重复存放；工程中的 `main.tex` 为报告，`metadata.tex` 集中填写个人信息，正文 4 页、封面另计。报告涵盖模型结构、形状变换、逐阶段误差、错误修正、局限与辅助说明。

Overleaf：上传独立的 Overleaf ZIP，选择 **XeLaTeX** 编译器和 `main.tex` 主文件；修改 `metadata.tex` 后重新编译，下载 PDF。本地可在 `report/` 中运行 `xelatex main.tex` 两次。

图表和表格已包含，不需要运行 Python 即可编译报告。重新生成图表（在项目根目录）：

```bash
python -m pip install -r requirements-report.txt
python make_report_assets.py
```

报告图从 `results/output_tokens.npz` 和老师的参考文件计算；编码器不读参考文件。参考文件仅供 `check.py`、附加验证和报告统计使用。

## 6. 来源、贡献及辅助说明

- 本人已有材料：之前提交的 12 道 NumPy 练习代码。本次将其整合，未将练习结果写死。
- 教师提供：框架、模型参数、图片、预处理、LayerNorm/GELU/注意力归一化、运行及检查脚本、参考输出。
- 本次新增：六组编码器实现、注释、回归测试、错误复现、结果图表和报告修订；使用 **ChatGPT 辅助编写与整理，并在执行环境进行数值核验**。未声称新代码由本人独立完成。
- 模板参考：ChaoFan996/Experiment-Report-Template-SZU（详见 `THIRD_PARTY_NOTICES.md`）。

## 7. 交什么

按新附件要求提交 **`image_encoder.py`、本 README 和填写封面后的报告 PDF**。`numpy_homework.py` 是练习题要求的单文件答案，按老师是否单独收取练习一并上传。其余测试、日志、Overleaf 工程为复现和审阅补充。不要重复上传 `weights/image_encoder.npz` 或 `assets/reference_outputs.npz`。

若后续复用到期末工程，复制完成的 `image_encoder.py` 与教师提供的 `helpers.py`；期末工程仍需它自身的完整图文模型参数与分词器，不能用本项目图像专用权重替换。
