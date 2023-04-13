#### Instructions
```shell
code/
├── config_script
│   ├── config.py				# argument
├── data_loader_script
	├── bert_joint_threelink_prompt.py			# convert data as the link input of DPED 
	├── bert_mutiviewer_prompt_muti_dist.py			# convert data as the mutiview input of DPED
├── data_process_pkl
│   ├── ori_data_summary_55_14			# data pkl
├── data_process_script
│   ├── data_process.py			#  data pre-process
├── model_script
│   ├── Bert_joint_threelink_muti_dist_prompt.py				# Link Prompt model with dist
|	├── Bert_mutiviewer_prompt_muti_dist.py				# Two Promt model with 
|	├── util.py				# model util
├── util_script				# system util
|	├── CRF_F1.py				# CRF eval
|	├── mata_data_calss.py				# meta data
|	├── metrics.py				# eval metrics
|	├── optimizer_wf.py				# optimizer
├── vail_test_data_script
|	├── bert_base_joint.py				# test eval script
├── README.md
├── run_DPED.py			# run model
├── train_script.py
├── train_script_for_MLM.py
└── requirements.txt


```
Before running code, you need to installed the requirements.txt environment successfully.

```shell
pip install -r requirements.txt
```
Then you can run 

```shell
python ./run_DPED.py
```

