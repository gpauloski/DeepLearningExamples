## Copy Data

Copy the data to `/tmp/` using `mpiexec -hostfile hostfile -N 1 ./scripts/cp_coco_to_tmp.sh`.

## To Run:

Launch one `run_on_node.sh` on each node:
```
$ mpiexec -hostfile /path/to/hostfile 1 ./scripts/run_on_node.sh -n $NGPU_PER_NODE -N $NNODES -m $MASTER_ADDR -c configs/e2e_mask_rcnn_R_50_FPN_1x_16GPU.yaml
```
or use `./scripts/run.sh` which wraps the mpiexec command.

The training script, `tools/train_net.py`, takes an optional flag `--kfac` to use KFAC when training.
By default, `scripts/run_on_node.sh` has KFAC enabled.
The KFAC preconditioner is initialized in `test()` in `tools/train_net.py`, and all KFAC hyperparams are listed there.

This requires the `experimental` KFAC branch.
