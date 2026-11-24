# Project 1：NumPy 图像编码器

代码仓库：https://github.com/Rhy-hero/pythonproject1

已完成 `image_encoder.py` 的 TODO-01 至 TODO-06。固定权重下，输入为 `float32 [B,3,224,224]`，输出为单位长度的 `float32 [B,512]`。模型仅依赖 NumPy 与教师提供的辅助函数，不使用 PyTorch、不训练、不读取参考答案。

## 如何运行

先解压老师的学生包，再把本仓库中的文件复制到 `project1-numpy-student/` 目录，保留原有 `weights/` 和 `assets/`。仓库按要求未重复包含大型权重或参考数据。

```bash
python -m pip install -r requirements.txt
python check.py --stage all > results/check_all.txt
python run.py --output results/output_tokens.npz
python -m unittest discover -s tests -v
python validate_additional.py
python practice_solutions/numpy_homework.py
```

自选图像：`python run.py --images a.jpg b.png --output my_tokens.npz`。
实测 Python 3.12.14 / NumPy 2.3.5 / Pillow 12.3.0 / imageio 2.38.2；完整环境可用 `requirements-validated.txt` 固定版本。

## 已验证

- 资产校验和教师提供的七个数值阶段全部 PASS；10 张图像输出最大绝对误差 5.662e-7。
- 6 项独立回归测试通过；新图像、输入变化、换序和重复批次均已验证。
- 12 道既有练习已合并为 `practice_solutions/numpy_homework.py`，输出与原代码一致。
- 本仓库提供代码和说明，先前的实测日志及数值结果未公开上传；它们随完整提交包另行提供。执行上述命令可在本地生成 `results/check_all.txt`，详细步骤见 `REPRODUCTION.md`。

## 报告与提交

报告 PDF 和 Overleaf 源文件作为单独的交付文件提供，不包含在本代码仓库中。修改报告的 `metadata.tex` 填写个人信息，以 XeLaTeX 编译；正文 4 页，封面另计。

按老师要求提交 **`image_encoder.py`、本 README 和填写封面后的 PDF**；练习答案可按老师安排另交。测试、日志和 Overleaf 源码为补充材料。不要重复提交大型模型参数和参考数据。

## 来源与辅助

之前的 12 道练习来自本人已有代码；框架、参数、数据和辅助函数由课程提供。本次六组编码器实现、测试及报告使用 **ChatGPT 辅助编写与整理，并实际运行核验**。红色标注表示相对教师框架新增的实现，不表示未使用辅助。模板与来源见 `THIRD_PARTY_NOTICES.md`。
