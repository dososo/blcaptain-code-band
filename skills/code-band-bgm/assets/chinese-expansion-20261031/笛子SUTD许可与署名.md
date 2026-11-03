# SUTD有膜G调笛子：来源、许可、修改与边界

取得和验证日期：2026-10-31。特定官方来源为[Zenodo6333932](https://zenodo.org/records/6333932)，题为“Dízi Fingering Dataset: Acoustic Impedance and Audio Spectra, with Sound Files”，发布日期2022-03-07。作者Hsien Han Ng、Yi Xian Tan、Teck Seng Ng、Chen Jer-Ming；机构包括Singapore University of Technology and Design和Asian Music Technology。官方记录明确[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)（[完整条款](https://creativecommons.org/licenses/by/4.0/legalcode.zh-hans)），描述涵盖G调笛子35种标准/替代指法、24个半音/半音间隔不连续音名的实录音频和声学谱，并非只有标注获许可。

声学测量说明涉及UNSW悉尼实验室、John Smith/Joe Wolfe、Paul Dickens和Noel Hanna；不将声学测量人员自动改称本批演奏者。当前SUTD交互官网有访问拦截，本项目未绕过；固定Zenodo官方API和原文件公开正常可取得。旧简略记录6333670也为CC BY4，但本次音频只依赖描述更明确的6333932，不混两个版本。

完整`dizi-repository.zip` 58,855,726字节，官方MD5 `eacdaa660a55eec1047f4cbec1ef2764`已真实核验；SHA256 `7ec167acceea4a58863396b2c821e813a4cdc83395afa9322919738b08f4bdd4`。原包395目录条目中69个有效WAV，有膜34、无膜35；元数据中的35种指法不能直接变成本次35个有膜声音或24个默认可用键。包内无额外文本许可冲突；官方原metadata、完整校验与每个成员CRC/SHA随包。

## 署名和实际修改

建议随库及派生作品附：Hsien Han Ng、Yi Xian Tan、Teck Seng Ng、Chen Jer-Ming，2022，“Dízi Fingering Dataset: Acoustic Impedance and Audio Spectra, with Sound Files”，https://zenodo.org/records/6333932 ，CC BY4.0，https://creativecommons.org/licenses/by/4.0/ 。本项目只选有膜独立单音、排除风险高音、保留默认/替代指法、测量并固定细调音准、去明显前静音留5ms、淡入淡出、重采样/量化及SF2重打包。不得暗示作者或机构背书。

34份有膜原WAV实际均48kHz、24位、单声道，合计32,131,764字节，保留于`Dizi-SUTD/dizi-repository/`；指法TIF原图也保留，不自动识别图中孔位或编造具体半孔方式。无膜录音未提取进本库，不把它当有膜音色。SF2制库透明转换44.1kHz/PCM16、原单声道等量复制左右，双声道不增加原生空间信息；最终48k/24位作品不等于恢复制库量化信息。

## 正式音高与指法范围

`Dizi-SUTD-Membrane-v2.sf2`仅3预设、32个真实独立单音/指法组合（64个左右声道区域），完善笛子一种乐器，不累计成3件乐器，也不外推梆笛、曲笛等未由作者具体命名的细型。

| bank/program | 资源 | 实际开放MIDI键 |
|---|---|---|
| 0/0 | dizi_sutd_g_membrane：默认指法条目 | 74、76、77、78、79、81、82、83、84、85、86、88、89、90、91、93、94、95、96、98、100 |
| 0/1 | dizi_sutd_g_alt1：第一替代指法条目 | 82、83、84、86、95、96、97、98、100 |
| 0/2 | dizi_sutd_g_alt2：第二替代指法条目 | 96、97 |

只有表中键可触发，无半音补齐或音域外推。C#7=97只有替代条目，无有膜standard条目。单录音力度，没有真实多力度层、自动RR、legato、人工循环或无限延音；指法变体不重叠混成RR。实录自然音高与气息变化保留，常量细调修正平均偏差，不声称整个采样每一刻均等于平均律。

作者标签F#7/G7的原WAV有841/664个近满幅点、最长4/3连续帧；F#7成熟pYIN约102.95，偏近G7而非标签102。保留原音和`原录满幅诊断.json`，以过载/标签风险拒绝正式映射，不通过衰减或移调掩盖缺陷。其他32条经librosa0.11.0完整pYIN测量和局部周期互证后才映射，原标签与真实测量都保留。

## 验证、算法版本与复现

32个音/指法真实渲染、48,768组音高/力度区域检查、渲染后代表周期音准均小于25音分、固定float重复渲染和SF2二次打包逐字节相同通过。独立自编30秒试奏WAV为48k/24位/双声道、1440000帧，样本峰0.253598，见`笛子SUTD验证结果.json`；人工听感待验收，未测真峰/响度，不将样本峰混作这些指标。

高音验证发现资产内旧周期测量器误拒搜索端点：纯音MIDI100的真实周期约16.72帧，离散局部峰为17帧，恰碰整取边界。新笛子显式启用`allow_edge_peak=True`，仅允许确与左右邻点构成局部峰的端点，非峰仍拒；36–103共68个纯音不误拒，实际最大误差4.879音分，纯音100邻点断言保留。旧资产调用保持默认旧策略，252个有效旧PCM测量中位数与旧方法完全一致；旧SF2、旧源和已交测量未改写。这是测量修正，不是新增音色或听感结论。

新算法使C7第二替代指法的校正量变化19音分，故新笛子正式采用版本化v2；首个v1及其技术试奏保留工作历史，未覆盖。其余31条PCM及校正未变。

权威离线源码`build/准备SUTD笛子.py`及`build/验证SUTD笛子.py`仅使用本包原WAV、源hash和冻结成熟测量，不联网；薄写库器`build/打包.cpp`调用固定SF2cute提交`3c5fc83b6ba3d1feb377f9c86021fd77499eb7c0`，原文Zlib许可与构建方法见`说明.md`。从项目根执行并选择新输出：

```sh
.venv/bin/python code-band-bgm/assets/chinese-expansion-20261031/build/准备SUTD笛子.py --work work/SUTD笛子重建
c++ -std=c++14 -I/path/to/sf2cute/include code-band-bgm/assets/chinese-expansion-20261031/build/打包.cpp /path/to/sf2cute-build/libsf2cute.a -o work/SUTD笛子重建/打包器
work/SUTD笛子重建/打包器 work/SUTD笛子重建/zones.tsv work/SUTD笛子重建/重建.sf2
.venv/bin/python code-band-bgm/assets/chinese-expansion-20261031/build/验证SUTD笛子.py --work work/SUTD笛子重建
```

验证命令检查本包正式v2，新重建文件另比SHA256。其他平台需重新技术验收；机构发布声明不是对全部权利的绝对保证。回滚时仅解除独立asset及三个资源注册，保留原音、作品及历史。
