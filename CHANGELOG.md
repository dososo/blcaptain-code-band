# 更新记录

## 0.1.0（2026-11-03）

首个公开版。

- `code-band-bgm` skill：中文需求 → 编排计划 → 乐谱 → 本地 FluidSynth 采样渲染 → WAV 与可编辑工程。
- 音色注册表 61 个包（CC0、CC BY 3.0/4.0、MIT、Iowa MIS），音色文件走 Release 附件，`scripts/fetch_soundfonts.py` 下载并校验 SHA-256。
- `arrange-check` 编排检查、`revise` / `score_edit.py` 定向修改、`check` 音频检查。
- 不含：插件合成器路径（Surge XT / pedalboard）、谱模型音色拆解（sms-tools）。
