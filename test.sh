
cost_class=0.5
eos_coef=0.1

for cost_box in 1 5;
	do
	for cost_giou in 2 5;
	do
		for bbox_loss_coef in 1 5;
		do
			for giou_loss_coef in 2 5;
			do
				echo ${cost_class} $cost_box $cost_giou $bbox_loss_coef $giou_loss_coef
				# python3 main.py \
				# --coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_$cost_class\_$cost_box\_$cost_giou\_$bbox_loss_coef\_$giou_loss_coef\_$eos_coef \
				# --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
				# --set_cost_class $cost_class --set_cost_bbox $cost_box --set_cost_giou $cost_giou \
				# --bbox_loss_coef $bbox_loss_coef --giou_loss_coef $giou_loss_coef --eos_coef $eos_coef
			done
		done
	done
done