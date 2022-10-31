from ml_collections import ConfigDict as D

def get_config():
	config = {}
	
	config["model"] = D({
		"sample_size": 32,
		"in_channels": 3,
		"out_channels": 3,
		"down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
		"mid_block_type": "FlaxResnetBlock2D", #for cross-attention, use FlaxUNetMidBlock2DCrossAttn
		"up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
		"block_out_channels": (128, 256, 512, 512),
		"layers_per_block": 2,
		"output_type": "eps"
	})

	config["dataset"] = D({
		"dataset_name": "cifar10"
	})

	config["schedule_name"] = "ddpm"
	
	config["schedule"] = D({
		"num_train_timesteps": 1000,
		"beta_start": 0.0001,
		"beta_end": 0.02,
		"beta_schedule": "linear"
	})

	config["optimizer"] = D({
		"lr": 1e-4,
		"b1": 0.9,
		"b2": 0.999,
		"ema_decay": 0.9999,
		"eps": 1e-8,
	})

	config["training_args"] = D({		
        "iterations": 1000000,
		"batch_size": 256,
		"checkpoint_dirs": ["{}/checkpoints_recent", "{}/checkpoints_permanent"],  
        "num_checkpoints": [10, 999999],
        "save_freq": [2500, 50000],

        "log_dir": "{}/logs",
        "log_freq": 500
	})

	return D(config)
