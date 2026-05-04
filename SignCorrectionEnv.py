import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SignCorrectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 전처리된 데이터 로드
        self.word_embeddings = np.load("dataset/word_embeddings.npy", allow_pickle=True).item()
        self.gt_sequences = np.load("dataset/gt_sequences.npy", allow_pickle=True).item()
        self.word_list = list(self.word_embeddings.keys())
        
        # State: [단어 임베딩(32) + 현재 좌표(63)]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(95,), dtype=np.float32)
        # Action: 다음 프레임 좌표 변화량
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(63,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_word = np.random.choice(self.word_list)
        self.current_frame_idx = 0
        self.current_pose = np.copy(self.gt_sequences[self.current_word][0])
        return np.concatenate([self.word_embeddings[self.current_word], self.current_pose]), {}

    def step(self, action):
        self.current_pose += action
        self.current_frame_idx += 1
        
        # Reward: 정답 좌표와의 거리 기반 보상
        target_pose = self.gt_sequences[self.current_word][self.current_frame_idx]
        reward = -np.linalg.norm(self.current_pose - target_pose)
        
        terminated = self.current_frame_idx >= 89
        obs = np.concatenate([self.word_embeddings[self.current_word], self.current_pose])
        return obs, reward, terminated, False, {}