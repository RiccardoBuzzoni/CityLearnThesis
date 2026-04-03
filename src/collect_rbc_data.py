import pandas as pd
import os
import sys
from pathlib import Path
from utils import Config, default_env_config, select_env_config

sys.path.append(str(Path(__file__).parent.parent))

from CityLearn.citylearn.citylearn import CityLearnEnv
from main_rbc import AdvancedRBC

def collect_data(agent_clas, env):
    """
    Runs a complete simulation of AdvancedRBC and returns a DataFrame with
    observations and actions.
    
    Args:
        agent_cass  : Will determin the agent (AdvancedRBC)
        env         : The environment
    
    Returns:
        DataFrame   : A .csv file containing observations and actions.
    """
    agent = agent_clas(env)
    observations, _ = env.reset()
    done = False
    rows = []

    obs_names = env.observation_names[0]
    action_names = env.action_names[0]

    while not done:
        actions = agent.predict(observations)
        next_observations, rewards, terminated, truncated, info = env.step(actions)

        def get_obs(name):
            return observations[0][obs_names.index(name)] if name in obs_names else None

        def get_act(name):
            return actions[0][action_names.index(name)] if name in action_names else None
        
        row = {
            # Observations
            "hour":                   get_obs("hour"),
            "indoor_temp":            get_obs("indoor_dry_bulb_temperature"),
            "cooling_setpoint":       get_obs("indoor_dry_bulb_temperature_cooling_set_point"),
            "outdoor_temp":           get_obs("outdoor_dry_bulb_temperature"),
            "outdoor_temp_predicted": get_obs("outdoor_dry_bulb_temperature_predicted_1"),
            "cooling_demand":         get_obs("cooling_demand"),
            "elec_price":             get_obs("electricity_pricing"),
            "carbon_intensity":       get_obs("carbon_intensity"),
            "solar_generation":       get_obs("solar_generation"),
            "occupant_count":         get_obs("occupant_count"),
            "electrical_storage_soc": get_obs("electrical_storage_soc"),
            "dhw_storage_soc":        get_obs("dhw_storage_soc"),
            "dhw_demand":             get_obs("dhw_demand"),
            # Actions
            "cooling_device":  get_act("cooling_device"),
            "dhw_storage":      get_act("dhw_storage"),
            "electrical_storage":  get_act("electrical_storage"),
            # Reward
            "reward": rewards[0],
        }

        rows.append(row)
        obs = next_observations
        done = terminated or truncated

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    config = Config()
    args = config.args

    if args.data is None:
        args.data = 'citylearn_challenge_2023_phase_1'

    # Load CityLearn schema
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data) # type: ignore

    env = CityLearnEnv(schema=schema, central_agent=True)

    os.makedirs('datasets', exist_ok=True)

    print(f"Collectin data: {args.data}...")
    df = collect_data(AdvancedRBC, env)

    output_path = 'datasets/advanced_rbc_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"AdvancedRBC data was successfully collected: {df.shape}")
    print(f"File saved in: {output_path}")
