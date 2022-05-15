# RGB-only six degrees of freedom pose estimation with neural radiance fields

<p float="left" text-align='center'>
  <img src="/assets/imgs/0_afast_i1batch512_TEST.gif" width="30%" />
  <img src="/assets/imgs/1_bnfast_i2batch512_TEST.gif" width="30%" /> 
  <img src="/assets/imgs/2_gyfast_i1batch512_TEST.gif" width="30%" />
</p>

Special thanks to:
- [https://github.com/bmild/nerf](https://github.com/bmild/nerf)
- [https://github.com/yenchenlin/nerf-pytorch/](https://github.com/yenchenlin/nerf-pytorch)
- [https://github.com/salykovaa/inerf](https://github.com/salykovaa/inerf)

## Example run
```
python main.py --config configs/general.txt --model_name arm_tr_200000 --obj_name arm_tr --obs_img_num 2 --experiment arm_2
```

## Used objects & architecture
![Used object](/assets/imgs/tilted_merged_5.png)
![Used architecture](/assets/imgs/full_work_arch_whitebg.png)

## Connected links
- [Example runs](https://bit.ly/3yA0N2J)
- [Data generation](https://github.com/nellika/synth-data-generator)
- [Parse logs](https://github.com/nellika/parse-thesis-results)

