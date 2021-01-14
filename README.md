# GeneralExplorationPolicy

TEST via
python rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run PPO
--env mars_explorer:explorer-v01 --video-dir .

python rollout.py /home/dkoutras/ray_results/PPO_config_serach/2_1_1/checkpoint_191/checkpoint-191 --run PPO --env mars_explorer:exploConf-v01 --episodes 3 --save-info --out rollouts_2_1_1.pkl --video-dir . --config config.json


python rollout.py /home/dkoutras/ray_results/PPO_config_serach/2_1_1/checkpoint_191/checkpoint-191 --run PPO --env mars_explorer:exploConf-v01 --episodes 3 --save-info --out rollouts_2_1_1.pkl --video-dir . --config '{"env_config": {"conf": { "margins": [5,5]}}}'
