from rich.traceback import install

from .cfgs.collect_cfg import cfg

install()


def train():
    from .experimenters.trainer import Trainer
    
    # 检查分布式环境是否初始化
    if torch.distributed.is_initialized():
        print(f"当前进程 Rank: {dist.get_rank()}, 总进程数: {dist.get_world_size()}")
    
    # 初始化 Trainer 前手动检查路径
    test_dir = "/content/Continual-Reg/save"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
    assert os.access(test_dir, os.W_OK), f"目录 {test_dir} 无写入权限！"

    trainer = Trainer(cfg)
    trainer.train()


def test():
    from .experimenters.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.test()
