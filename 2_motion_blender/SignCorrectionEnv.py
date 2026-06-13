"""
SignCorrectionEnv.py
수어 표제어 사전 기반 RL 환경

관찰(obs): 복합어를 분해한 글로스들의 임베딩 평균
행동(action): 각 글로스별 후보 키 인덱스 선택 (이산)
보상:
    R_validity  : 선택한 글로스가 표제어 사전에 등록된 단어인가
    R_synonym   : 선택한 키의 글로스가 목표 글로스와 같은 표제어 그룹인가
    R_coverage  : 복합어의 모든 글로스를 빠짐없이 커버했는가
"""

import os
import re
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ── 표제어 사전 로드 헬퍼 ────────────────────────────────────────
def load_sign_dictionary(xlsx_path: str):
    """
    표제어 xlsx → (word_to_id, id_to_words) 반환
    word_to_id  : { '여자': 12687, '여성': 12687, ... }
    id_to_words : { 12687: ['여성', '부녀', '여자', ...], ... }
    """
    df = pd.read_excel(xlsx_path)
    word_to_id  = {}
    id_to_words = {}

    for _, row in df.iterrows():
        tid  = int(row['수어 표제어 번호'])
        expr = str(row['한국어 대응표현'])
        # 괄호 안 내용 제거 후 쉼표 split
        cleaned = re.sub(r'\(.*?\)', '', expr)
        words   = [w.strip() for w in cleaned.split(',') if w.strip()]
        id_to_words[tid] = words
        for w in words:
            word_to_id[w] = tid

    return word_to_id, id_to_words


class SignCorrectionEnv(gym.Env):
    """
    에피소드 구조
    - reset(options): 복합어 + 분해된 글로스 목록 + 글로스별 후보 키 목록 주입
    - step(action) : 각 글로스에 대해 후보 키 인덱스를 선택
                     → 표제어 사전 기반 reward 계산
    - terminated   : 1스텝 Contextual Bandit (글로스 선택은 단발성 결정)
    """

    MAX_GLOSSES    = 4   # 한 복합어당 최대 글로스 수
    MAX_CANDIDATES = 5   # 글로스당 최대 후보 키 수

    def __init__(self,
                 data_dir:  str = "dataset_processed",
                 dict_xlsx: str = "sign_dict_vocab.xlsx"):
        super().__init__()

        # ── 데이터셋 로드
        gt_data  = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))
        emb_data = np.load(os.path.join(data_dir, "word_embeddings_ALL.npz"))
        self.gt_sequences    = {k: v.astype(np.float32) for k, v in dict(gt_data).items()}
        self.word_embeddings = {k: v.astype(np.float32) for k, v in dict(emb_data).items()}
        self.all_keys        = list(self.gt_sequences.keys())

        sample_key      = self.all_keys[0]
        self.embed_dim  = self.word_embeddings[sample_key].shape[0]

        # ── 표제어 사전 로드
        if os.path.exists(dict_xlsx):
            self.word_to_id, self.id_to_words = load_sign_dictionary(dict_xlsx)
            print(f"📚 표제어 사전 로드 완료: {len(self.word_to_id):,}개 단어")
        else:
            print(f"⚠️  표제어 사전({dict_xlsx}) 없음 → validity/synonym 보상 비활성화")
            self.word_to_id  = {}
            self.id_to_words = {}

        # ── Action Space
        # 글로스마다 후보 인덱스(0~MAX_CANDIDATES-1) 하나씩 선택
        # MultiDiscrete: [MAX_CANDIDATES] * MAX_GLOSSES
        self.action_space = spaces.MultiDiscrete(
            [self.MAX_CANDIDATES] * self.MAX_GLOSSES
        )

        # ── Observation Space
        # 복합어 전체 임베딩 (글로스 임베딩 평균)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.embed_dim,),
            dtype=np.float32
        )

        # 에피소드 상태
        self.current_glosses    = []   # ['교통', '사고']
        self.candidates         = {}   # {'교통': [key0, key1, ...], '사고': [...]}
        self.current_obs        = None

    # ── reset ────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "glosses" in options and "candidates" in options:
            # Qwen 파이프라인에서 실제 글로스 + 후보 주입
            self.current_glosses = options["glosses"][:self.MAX_GLOSSES]
            self.candidates      = {
                g: options["candidates"].get(g, [])[:self.MAX_CANDIDATES]
                for g in self.current_glosses
            }
        else:
            # 학습 중 무작위 샘플링
            n = np.random.randint(2, self.MAX_GLOSSES + 1)
            sampled_keys = np.random.choice(self.all_keys, size=n, replace=False)
            self.current_glosses = [k.split('_')[0] for k in sampled_keys]
            self.candidates = {
                g: [k for k in self.all_keys if k.split('_')[0] == g][:self.MAX_CANDIDATES]
                for g in self.current_glosses
            }

        # obs: 유효 후보 임베딩들의 평균
        emb_list = []
        for g, keys in self.candidates.items():
            for k in keys:
                if k in self.word_embeddings:
                    emb_list.append(self.word_embeddings[k])
        if emb_list:
            self.current_obs = np.mean(emb_list, axis=0).astype(np.float32)
        else:
            self.current_obs = np.zeros(self.embed_dim, dtype=np.float32)

        return self.current_obs, {}

    # ── step ─────────────────────────────────────────────────────
    def step(self, action):
        """
        action: np.array, shape=(MAX_GLOSSES,), dtype=int
                각 원소 = 해당 글로스의 후보 키 인덱스
        """
        total_reward = 0.0
        n_glosses    = len(self.current_glosses)

        for i, gloss in enumerate(self.current_glosses):
            keys      = self.candidates.get(gloss, [])
            chosen_idx = int(action[i]) % max(len(keys), 1)

            if not keys:
                # 후보 없음 → 패널티
                total_reward -= 1.0
                continue

            chosen_key   = keys[chosen_idx]
            chosen_gloss = chosen_key.split('_')[0]

            # ── R_validity: 선택한 글로스가 표제어 사전에 있는가
            r_validity = 1.0 if chosen_gloss in self.word_to_id else -1.0

            # ── R_synonym: 선택 글로스와 목표 글로스가 같은 표제어 그룹인가
            r_synonym = 0.0
            if chosen_gloss in self.word_to_id and gloss in self.word_to_id:
                if self.word_to_id[chosen_gloss] == self.word_to_id[gloss]:
                    r_synonym = 2.0   # 같은 수어 동작 그룹 → 최고 보상
                else:
                    r_synonym = -0.5  # 다른 그룹
            elif chosen_gloss == gloss:
                r_synonym = 2.0       # 표제어 사전에 없어도 완전 일치면 OK

            # ── R_diversity: 여러 글로스 중 같은 키를 중복 선택하지 않았는가
            # (이미 선택된 키 목록과 비교)
            r_diversity = 0.0  # 아래 전체 루프 후 계산

            gloss_reward = 1.5 * r_validity + 2.0 * r_synonym
            total_reward += gloss_reward

        # ── R_coverage: 모든 글로스에 유효 후보가 있었는가
        covered = sum(1 for g in self.current_glosses if self.candidates.get(g))
        r_coverage = (covered / max(n_glosses, 1)) * 1.0
        total_reward += r_coverage

        # ── R_diversity: 선택된 키가 중복되지 않았는가
        selected_keys = []
        for i, gloss in enumerate(self.current_glosses):
            keys = self.candidates.get(gloss, [])
            if keys:
                idx = int(action[i]) % len(keys)
                selected_keys.append(keys[idx])
        if len(selected_keys) == len(set(selected_keys)):
            total_reward += 0.5   # 중복 없음 보너스

        terminated = True
        info = {
            "selected_keys": selected_keys,
            "glosses":       self.current_glosses,
        }
        return self.current_obs, total_reward, terminated, False, info

    # ── 외부에서 직접 호출: 최적 키 선택 ────────────────────────
    def select_keys(self, glosses: list, candidates: dict, model) -> dict:
        """
        PPO 모델로 글로스별 최적 후보 키를 선택해 반환합니다.
        반환: { '교통': 'key_chosen', '사고': 'key_chosen' }
        """
        obs, _ = self.reset(options={"glosses": glosses, "candidates": candidates})
        action, _ = model.predict(obs, deterministic=True)

        result = {}
        for i, gloss in enumerate(self.current_glosses):
            keys = self.candidates.get(gloss, [])
            if keys:
                idx = int(action[i]) % len(keys)
                result[gloss] = keys[idx]
        return result