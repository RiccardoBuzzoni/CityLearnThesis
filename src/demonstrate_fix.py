"""Demonstrate the timing issue and proper fix"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from CityLearn.citylearn.citylearn import CityLearnEnv
from main_rbc import AdvancedRBC
from utils import Config, default_env_config

if __name__ == "__main__":
    config = Config()
    args = config.args
    
    if args.data is None:
        args.data = 'citylearn_challenge_2023_phase_1'
    
    schema = default_env_config(args.data)
    env = CityLearnEnv(schema=schema, central_agent=True)
    
    agent = AdvancedRBC(env)
    obs, _ = env.reset()
    
    obs_names = env.observation_names[0]
    action_names = env.action_names[0]
    building = env.buildings[0]
    
    print("=" * 80)
    print("DEMONSTRATING THE TIMING ISSUE")
    print("=" * 80)
    print()
    
    print("Step 0:")
    print(f"  Observation electrical_storage_soc (from reset): {obs[0][obs_names.index('electrical_storage_soc')]:.4f}")
    print(f"  Battery soc[0] (internal): {building.electrical_storage.soc[0]:.4f}")
    
    actions = agent.predict(obs)
    elec_action = actions[0][action_names.index('electrical_storage')]
    print(f"  Action for electrical_storage: {elec_action:.3f}")
    
    next_obs, reward, _, _, _ = env.step(actions)
    
    print()
    print(f"After step (WRONG - reads from time_step=1):")
    print(f"  Observation electrical_storage_soc: {next_obs[0][obs_names.index('electrical_storage_soc')]:.4f}")
    print(f"  Battery soc[0] (was updated): {building.electrical_storage.soc[0]:.4f}")
    print(f"  Battery soc[1] (returned obs reads this): {building.electrical_storage.soc[1]:.4f}")
    print()
    
    print(" THE FIX:")
    print("  Use soc[time_step - 1] in collect_data if you want the result of current actions")
    print("  Or capture the battery.soc BEFORE stepping:")
    
    print()
    print("=" * 80)
    print("STEP 1 WITH FIX")
    print("=" * 80)
    
    # Reset for next step
    obs = next_obs
    
    # Capture SOC BEFORE stepping
    soc_before_action = building.electrical_storage.soc[building.time_step]
    print(f"  SOC before action (captured before step): {soc_before_action:.4f}")
    
    actions = agent.predict(obs)
    elec_action = actions[0][action_names.index('electrical_storage')]
    print(f"  Action for electrical_storage: {elec_action:.3f}")
    
    next_obs, reward, _, _, _ = env.step(actions)
    
    # The SOC from the PREVIOUS action gets stored in battery.soc[time_step - 1]
    soc_result_from_action = building.electrical_storage.soc[building.time_step - 1]
    soc_from_obs = next_obs[0][obs_names.index('electrical_storage_soc')]
    
    print(f"  SOC result of action (soc[time_step-1]): {soc_result_from_action:.4f}")
    print(f"  SOC from observation: {soc_from_obs:.4f}")
