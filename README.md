# RGB-only six degrees of freedom pose estimation with neural radiance fields

Special thanks to:
- [https://github.com/bmild/nerf](https://github.com/bmild/nerf)
- [https://github.com/yenchenlin/nerf-pytorch/](https://github.com/yenchenlin/nerf-pytorch)
- [https://github.com/salykovaa/inerf](https://github.com/salykovaa/inerf)

## Example run
```
python main.py --config configs/general.txt --model_name arm_tr_200000 --obj_name arm_tr --obs_img_num 2 --experiment arm_2
```

## Used architecture
![Used architecture](/assets/imgs/full_work_arch_whitebg.png)

<p float="left">
  <img src="/assets/imgs/0_afast_i1batch512_TEST.gif" width="150" />
  <img src="/assets/imgs/1_bnfast_i2batch512_TEST.gif" width="150" /> 
  <img src="/assets/imgs/2_gyfast_i1batch512_TEST.gif" width="150" />
</p>
