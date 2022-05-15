# RGB-only six degrees of freedom pose estimation with neural radiance fields

## Example run:
```
python main.py --config configs/general.txt --model_name arm_tr_200000 --obj_name arm_tr --obs_img_num 2 --experiment arm_2
```

Special thanks to:
- [https://github.com/bmild/nerf](https://github.com/bmild/nerf)
- [https://github.com/yenchenlin/nerf-pytorch/](https://github.com/yenchenlin/nerf-pytorch)
- [https://github.com/salykovaa/inerf](https://github.com/salykovaa/inerf)

## Used architecture
![Used architecture](/assets/imgs/full_work_arch_whitebg.png)
![alt-text-1](/assets/imgs/0_afast_i1batch512_TEST.gif | width=150) ![alt-text-2](/assets/imgs/1_bnfast_i2batch512_TEST.gif | width=150) ![alt-text-2](/assets/imgs/2_gyfast_i1batch512_TEST.gif | width=150)