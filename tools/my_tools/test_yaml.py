import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=False, default="tools/infer.cfg.yaml", type=str, help="provide config file for this module")
    args = parser.parse_args()

    from    torchpack.utils.config import configs
    from    mmcv import Config
    from    mmdet3d.utils import recursive_eval

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg = cfg.visualize
    layout = cfg.configs[cfg.config_to_use]
    if(layout['layout'] =='type1'):
        print(f'-------------type 1')
if __name__ == "__main__":
    main()