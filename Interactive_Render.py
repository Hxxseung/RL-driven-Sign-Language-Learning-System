import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 모델 및 데이터 로드
model = PPO.load("graps_slp_model")
word_embeddings = np.load("dataset/word_embeddings.npy", allow_pickle=True).item()

while True:
    word = input("\n👉 생성할 단어를 입력하세요 (q: 종료): ").strip()
    if word.lower() in ['q', 'exit']: break
    if word not in word_embeddings:
        print(f"⚠️ '{word}'는 학습되지 않은 단어입니다.")
        continue

    # 실시간 동작 생성 루프
    emb = word_embeddings[word]
    current_pose = np.zeros(63, dtype=np.float32)
    sequence = [current_pose.copy()]

    for _ in range(89):
        obs = np.concatenate([emb, current_pose])
        action, _ = model.predict(obs, deterministic=True)
        current_pose += action
        sequence.append(current_pose.copy())

    # 3D 시각화 및 이미지 파일 저장
    final_pose = sequence[-1].reshape(21, 3)
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_pose[:,0], final_pose[:,1], final_pose[:,2], c='r', s=50)
    ax.set_title(f"Generated Result: {word}") 
    
    file_name = f"result_{word}.png"
    plt.savefig(file_name)
    plt.close()
    print(f"✅ 결과 저장 완료: {file_name}")