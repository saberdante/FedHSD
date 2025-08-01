# debugProxy.py
import os, sys, runpy

# 1. cd WORKDIR
os.chdir('/home/dante/Coode/PFed/FedHSD/system')
# configs/cifar100/dkd/res32x4_res8x4_aug.yaml
# 2A. python test.py 4 5
# configs/cifar100/mldist/res32x4_res8x4.yaml

# python tools/train_ours.py --cfg configs/cifar100/ours/res32x4_res8x4.yaml
# python tools/train.py --cfg configs/cifar100/dist/res32x4_res8x4.yaml
args = 'python main.py -data Cifar100 -ncl 100 -m CNN -algo FedHSD -gr 200 -did 0 -jr 1.0 -ls 5 -lbs 128 -bt 1.0 -lam 1.0 -sg 0.9'

# args = 'python train_baseline.py --model RegNetY_400MF \
#     --data-folder data/cifar100 \
#     --checkpoint-dir download_ckpts/cifar_teachers/RegnetY_400mf_vanilla'

# 2B. python -m mymodule.test 4 5
# args = 'python -m mymodule.test 4 5'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args"""
    args.pop(0)
if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path
sys.argv.extend(args[1:])
fun(args[0], run_name='__main__')
