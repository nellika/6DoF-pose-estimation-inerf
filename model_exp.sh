#!/bin/bash

k=3
name=""

for i in {1..4}
do

for j in 10
do

for m in 7
do

for b in 512
do
name="fast_i${i}batch${b}_dotted_huber"
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 0 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 0_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 1 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 1_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 2 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 2_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 3 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 3_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 4 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 4_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 5 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 5_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 6 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 6_a$name


name="brief_i${i}batch${b}_dotted_huber"
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 0 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 0_a$name
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 1 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 1_a$name
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 2 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 2_a$name
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 3 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 3_a$name
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 4 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 4_a$name
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 5 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 5_a$name
python main_2.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 6 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 6_a$name


done
done
done
done
