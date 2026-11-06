"""ccCFM: Context-Conditioned Conditional Flow Matching
for Cross-Cell-Line Genetic Perturbation Prediction.

中文:
    - 骨架阶段. 主要导出 VelocityNet 和 CCCFMLitModule 两个核心类.
    - 每个子模块都有独立 docstring 说明角色与 TODO.
"""

__version__ = "0.1.0"

try:
    from cccfm.velocity_net import VelocityNet
    from cccfm.flow_module import CCCFMLitModule
    __all__ = ["VelocityNet", "CCCFMLitModule"]
except ImportError:
    __all__ = []
