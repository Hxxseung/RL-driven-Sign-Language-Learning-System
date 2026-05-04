import torch
from stable_baselines3 import PPO
from SignCorrectionEnv import SignCorrectionEnv

# GPU 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
env = SignCorrectionEnv()

# PPO 모델 생성 및 학습
model = PPO("MlpPolicy", env, verbose=1, device=device)

print(f"🚀 {device}에서 GRAPS 모델 학습 시작...")
model.learn(total_timesteps=50000)

# 결과 저장
model.save("graps_slp_model")
print("✅ 모델 저장 성공: graps_slp_model.zip")