"""Curriculum learning implementations for robotic tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def radius_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor, **kwargs) -> None:
    """
    Curriculum learning function that gradually increases the bounding radius based on success rate.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to apply curriculum to.
        **kwargs: Additional parameters including:
            - initial_radius: Starting bounding radius (default: 1.0)
            - final_radius: Target bounding radius (default: 2.5)
            - success_rate_threshold: Success rate threshold to increase radius (default: 0.7)
            - min_episodes_per_stage: Minimum episodes before considering radius increase (default: 100)
            - adaptive_increment: Whether to use adaptive increment based on performance (default: True)
            - window_min: Minimum window size (default: 50)
            - window_k: Multiplier for environment count (default: 10)
    """
    # Get parameters from kwargs - check if parameters are nested in "kwargs"
    params = kwargs.get("kwargs", kwargs)
    
    initial_radius = params.get("initial_radius", 1.0)
    final_radius = params.get("final_radius", 2.5)
    success_rate_threshold = params.get("success_rate_threshold", 0.7)
    min_episodes_per_stage = params.get("min_episodes_per_stage", 100)
    adaptive_increment = params.get("adaptive_increment", True)
    
    # Calculate window size using formula: window_size = max(W_min, k × N_envs)
    window_min = params.get("window_min", 50)
    window_k = params.get("window_k", 10)
    window_size = params.get("window_size", max(window_min, window_k * env.num_envs))
    
    # Initialize curriculum state
    if not hasattr(env, '_curriculum_state'):
        env._curriculum_state = {
            'current_radius': initial_radius,
            'success_history': [],
            'episodes_at_current_stage': 0,
            'total_episodes': 0,
            'stage': 0
        }
        print(f"[Curriculum] Window size: {window_size} (max({window_min}, {window_k} × {env.num_envs}))")
        print(f"[Curriculum] Initializing environment to radius {initial_radius:.2f}")
        
        # Initialize environment to starting radius
        _update_environment_radius(env, initial_radius)
    
    state = env._curriculum_state
    
    # Update curriculum at episode end
    if hasattr(env, 'reset_buf') and env.reset_buf.any():
        reset_env_ids = env_ids[env.reset_buf[env_ids]]
        
        if len(reset_env_ids) > 0:
            # Check for episode success
            is_success = _check_episode_success(env)
            state['success_history'].append(1 if is_success else 0)
            state['episodes_at_current_stage'] += len(reset_env_ids)
            state['total_episodes'] += len(reset_env_ids)
            
            # Keep history manageable
            if len(state['success_history']) > window_size * 2:
                state['success_history'] = state['success_history'][-window_size:]
            
            # Check if we should advance the curriculum
            if state['episodes_at_current_stage'] >= min_episodes_per_stage:
                recent_episodes = min(len(state['success_history']), min_episodes_per_stage)
                recent_success_rate = sum(state['success_history'][-recent_episodes:]) / recent_episodes
                
                if recent_success_rate >= success_rate_threshold and state['current_radius'] < final_radius:
                    # Calculate increment
                    if adaptive_increment:
                        if recent_success_rate >= 0.9:
                            increment = 0.5
                        elif recent_success_rate >= 0.8:
                            increment = 0.3
                        else:
                            increment = 0.2
                    else:
                        increment = 0.25
                    
                    # Update radius
                    new_radius = min(state['current_radius'] + increment, final_radius)
                    state['current_radius'] = new_radius
                    state['stage'] += 1
                    state['episodes_at_current_stage'] = 0
                    
                    print(f"[Curriculum] Stage {state['stage']}: Radius {new_radius:.2f} "
                          f"(Success rate: {recent_success_rate:.2f}, Episodes: {state['total_episodes']})")
                    
                    # Update environment
                    _update_environment_radius(env, new_radius)


def _check_episode_success(env) -> bool:
    """Check if current episode was successful based on termination conditions."""
    if not hasattr(env, 'termination_manager'):
        return False
        
    # Look for success termination conditions
    # Access _terms.keys() since _terms is a dictionary
    if hasattr(env.termination_manager, '_terms'):
        for term_name in env.termination_manager._terms.keys():
            if 'success' in term_name.lower():
                try:
                    term_values = env.termination_manager.get_term(term_name)
                    if term_values is not None and term_values.any():
                        return True
                except:
                    continue
    return False


def _update_environment_radius(env: ManagerBasedRLEnv, new_radius: float) -> None:
    """Update the bounding radius in environment components."""
    print(f"[Curriculum] sampling_radius updated to {new_radius:.2f}")
    
    # Update command manager
    if hasattr(env, 'command_manager') and hasattr(env.command_manager, '_terms'):
        for term in env.command_manager._terms.values():
            if hasattr(term, 'cfg') and hasattr(term.cfg, 'sampling_radius'):
                term.cfg.sampling_radius = new_radius
    
    # Update event manager - check different possible attribute names
    if hasattr(env, 'event_manager'):
        event_terms = None
        if hasattr(env.event_manager, '_terms'):
            event_terms = env.event_manager._terms.values()
        elif hasattr(env.event_manager, '_mode_term_names'):
            # Try to get terms through the mode-based access
            try:
                event_terms = []
                for mode, term_names in env.event_manager._mode_term_names.items():
                    for term_name in term_names:
                        if hasattr(env.event_manager, '_term_names') and term_name in env.event_manager._term_names:
                            idx = env.event_manager._term_names.index(term_name)
                            if idx < len(env.event_manager._terms):
                                event_terms.append(env.event_manager._terms[idx])
            except:
                pass
        
        if event_terms:
            for term in event_terms:
                if hasattr(term, 'cfg') and hasattr(term.cfg, 'params'):
                    if 'sampling_radius' in term.cfg.params:
                        term.cfg.params['sampling_radius'] = new_radius
    
    # Update reward manager
    if hasattr(env, 'reward_manager') and hasattr(env.reward_manager, '_terms'):
        for term in env.reward_manager._terms.values():
            if hasattr(term, 'cfg') and hasattr(term.cfg, 'params'):
                if 'radius' in term.cfg.params:
                    term.cfg.params['radius'] = new_radius + 0.5  # give some buffer for penalties


def calculate_window_size(num_envs: int, window_min: int = 50, window_k: int = 10) -> int:
    """Calculate window size using formula: max(window_min, window_k * num_envs)"""
    return max(window_min, window_k * num_envs)

