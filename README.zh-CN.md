# ParaFDEONet

本仓库对应当前精简稿 **ParaFDEONet: A Unified Framework for Forward Prediction and Parameter Identification in Parameterized Functional Differential Equations**，覆盖竞争系统、SEI、Nicholson、八维电网和延迟冷却 CSTR。

## 使用

在仓库根目录运行，推荐使用 Python 3.11 或 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
python run.py smoke all
python run.py test all
```

`smoke` 使用小数据集、小网络和少量训练步数，检查数据生成、训练、保存和评估能否运行。它不用于判断是否达到论文精度；CSTR 的三步训练会保留正式精度门槛未通过的记录。

正式训练示例：

```bash
python run.py train competition --device cuda:0 --cpus 16 --output-dir outputs/competition
python run.py train sei --device cuda:0 --cpus 16 --output-dir outputs/sei
python run.py train nicholson --device cuda:0 --cpus 16 --output-dir outputs/nicholson
python run.py train smart_grid --device cuda:0 --cpus 16 --output-dir outputs/smart_grid
python run.py train cstr --device cuda:0 --cpus 16 --output-dir outputs/cstr
```

前三个系统默认使用各自配置中的全部方法和五个训练种子。额外的逐状态参数分支模型保留在配置中；正文展示的方法以[论文对应表](docs/PAPER_MAP.md)为准。可用 `--seed` 选择单个种子，CSTR 的种子由配置文件指定。

## 内容

- `experiments/`：五类系统的方程、数据生成、网络、损失、训练配置及原有测试。
- `benchmarks/`：三系统共享面板反演、参数 Jacobian 验证和特征复用计时。
- `results/paper/`：归档图表、逐案例结果、汇总数据、CSTR 事件数据和电网 GBT 标签。
- `scripts/`：统一检查、结果重绘和训练产物整理。
- `docs/`：复现步骤、版本对应、文件来源与实测记录。

当前正文采用的反演入口是 `python run.py entry inverse-lm`。原 `inverse` 入口中的 `frozen` 对应旧 Adam 版本，不能直接当作正文的 ParaFDEONet-LM。完整运行方式见[复现说明](docs/REPRODUCTION.md)。

## 上传

上传本目录内的文件即可。代码仓库包含必要代码、配置、依赖和图表源数据；大型训练集、模型权重、运行日志、缓存及论文草稿未放入发布包。权重可通过正式训练重新生成，整理方式见[数据与权重](docs/ARTIFACTS.md)。

已移除原通知地址，默认关闭 W&B 和通知；运行不依赖作者电脑或服务器的绝对路径。原项目的网络格式标识和子实验结构保留，方便兼容既有检查点。

仓库未自行添加开源许可证；所提供的原始项目没有可沿用的许可证。作者与文章标题已填写在 `CITATION.cff` 中。
