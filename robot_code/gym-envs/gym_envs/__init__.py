from gym.envs.registration import register 

register(
	id='RobotReach-v1',
	entry_point='gym_envs.envs:RobotReachEnv',
	max_episode_steps=40, #20,
	)

register(
	id='RobotDrawerClose-v1',
	entry_point='gym_envs.envs:RobotDrawerCloseEnv',
	max_episode_steps=20,
	)

register(
	id='RobotDrawerCloseRight-v1',
	entry_point='gym_envs.envs:RobotDrawerCloseRightEnv',
	max_episode_steps=25,
	)

register(
	id='RobotInsertPegYellow-v1',
	entry_point='gym_envs.envs:RobotInsertPegEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotInsertPegGreen-v1',
	entry_point='gym_envs.envs:RobotInsertPegEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotInsertPegBlue-v1',
	entry_point='gym_envs.envs:RobotInsertPegEnv',
	max_episode_steps=30,
	) 

register(
	id='RobotBoxOpen-v1',
	entry_point='gym_envs.envs:RobotBoxOpenEnv',
	max_episode_steps=40,
	) 

register(
	id='RobotBoxOpenRight-v1',
	entry_point='gym_envs.envs:RobotBoxOpenRightEnv',
	max_episode_steps=40,
	) 

register(
	id='RobotPour2D-v1',
	entry_point='gym_envs.envs:RobotPour2DEnv',
	max_episode_steps=30,
	)

register(
	id='RobotPour2DTop-v1',
	entry_point='gym_envs.envs:RobotPour2DEnv',
	max_episode_steps=30,
	)