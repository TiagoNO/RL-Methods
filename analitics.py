from RL_Methods.utils.Logger import Logger

import matplotlib.pyplot as plt 

filename = "experiments/rainbow/log_file"
logger = Logger(filename)
logger.load()


try:
    plt.plot(logger.data["time/sample_time"], label='sample_time')
    plt.plot(logger.data["time/log_prob_time"], label='log_prob_time')
    plt.plot(logger.data["time/next_distrib_time"], label='next_distrib_time')
    plt.plot(logger.data["time/projection_time"], label='projection_time')
    plt.plot(logger.data["time/prios_time"], label='prios_time')
    plt.plot(logger.data["time/action_time"], label='action_time')
    plt.plot(logger.data["time/trajectory_time"], label='trajectory_time')
    plt.legend()
    plt.show()
except:
    print("Did not log time data")


plt.plot(logger.data['train/timesteps'], logger.data['parameters/feature_extractor_L1_avg_noisy'], label='fe_L1') 
plt.plot(logger.data['train/timesteps'], logger.data['parameters/feature_extractor_L3_avg_noisy'], label='fe_L3') 
plt.plot(logger.data['train/timesteps'], logger.data['parameters/advantage_L1_avg_noisy'], label='ad_L1') 
plt.plot(logger.data['train/timesteps'], logger.data['parameters/advantage_L3_avg_noisy'], label='ad_L3') 
plt.plot(logger.data['train/timesteps'], logger.data['parameters/value_L1_avg_noisy'], label="v_L1") 
plt.plot(logger.data['train/timesteps'], logger.data['parameters/value_L3_avg_noisy'], label="v_L3") 
plt.legend()
plt.show()
plt.close()

plt.plot(logger.data['train/timesteps'], logger.data['train/avg_ep_rewards'])
plt.show()
plt.close()

plt.plot(logger.data['train/timesteps'], logger.data['parameters/beta'])
plt.plot(logger.data['train/timesteps'], logger.data['train/avg_ep_rewards'])
plt.show()
plt.close()