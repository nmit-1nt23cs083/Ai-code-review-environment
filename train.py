"""
Training script for RL agent on AI Code Review environment.

Trains a PPO agent for ~50k timesteps across different tasks and saves the model.

Usage:
  python train.py
  
Monitor training:
  tensorboard --logdir ./ppo_code_review_logs/
"""
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from train_env import CodeReviewEnv


class AccuracyLoggerCallback(BaseCallback):
    """Print episode accuracy score when an episode ends."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                accuracy = info.get("accuracy_score")
                if accuracy is not None:
                    print(f"✅ Episode accuracy score: {accuracy:.2f}")
        return True


def train():
    """Train PPO agent on code review environment."""
    
    # Create directories
    os.makedirs("ppo_code_review_logs", exist_ok=True)
    os.makedirs("ppo_code_review_models", exist_ok=True)
    
    # Detect device (GPU/CPU)
    if torch.cuda.is_available():
        device = "cuda"
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        print(f"✅ CUDA detected - {gpu_info}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("⚠️  CUDA not available - training on CPU (slower)")
    
    print(f"🚀 Starting RL training for Code Review Agent on {device.upper()}...")
    print("=" * 60)
    
    # Create vectorized environment (4 parallel envs for faster training)
    def make_env(task=None):
        return CodeReviewEnv(task_name=task)
    
    # Train across all task difficulties
    tasks = ["fix_syntax_and_obvious_bugs", "logic_and_security_review", "design_and_architecture_review"]
    
    # Start with single env, train on easy first
    env = DummyVecEnv([lambda: CodeReviewEnv(task_name="fix_syntax_and_obvious_bugs")])
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # tensorboard_log="./ppo_code_review_logs/",  # Disabled temporarily
    )
    
    # Checkpoint callback (save every 5k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./ppo_code_review_models/",
        name_prefix="ppo_codereview",
    )
    callback_list = CallbackList([checkpoint_callback, AccuracyLoggerCallback()])
    
    # Train phase 1: Easy task (10k steps)
    print("\n📚 PHASE 1: Training on easy task (fix_syntax_and_obvious_bugs)...")
    print(f"Steps: 0 → 10000")
    model.learn(total_timesteps=10000, callback=callback_list, log_interval=1, progress_bar=False)
    
    # Train phase 2: Medium task (20k steps)
    print("\n🔧 PHASE 2: Training on medium task (logic_and_security_review)...")
    env.close()
    env = DummyVecEnv([lambda: CodeReviewEnv(task_name="logic_and_security_review")])
    model.set_env(env)
    print(f"Steps: 10000 → 30000")
    model.learn(total_timesteps=20000, callback=callback_list, log_interval=1, progress_bar=False)
    
    # Train phase 3: Hard task (20k steps)
    print("\n🏆 PHASE 3: Training on hard task (design_and_architecture_review)...")
    env.close()
    env = DummyVecEnv([lambda: CodeReviewEnv(task_name="design_and_architecture_review")])
    model.set_env(env)
    print(f"Steps: 30000 → 50000")
    model.learn(total_timesteps=20000, callback=callback_list, log_interval=1, progress_bar=False)
    
    # Final save
    model.save("ppo_code_review_models/ppo_codereview_final")
    env.close()
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📦 Model saved to: ppo_code_review_models/ppo_codereview_final.zip")
    print(f"📊 Logs saved to: ppo_code_review_logs/")
    print(f"🖥️  Device used: {device.upper()}")
    print("\nNext: Use trained_inference.py to run the trained agent")
    print("=" * 60)


if __name__ == "__main__":
    train()
