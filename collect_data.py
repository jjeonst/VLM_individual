import os
import numpy as np
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

def main():
    # 1. 설정 파일 로드 (유저님이 만든 pr2l_objectnav.yaml 사용)
    # 최신 버전에 맞는 방식으로 설정을 불러옵니다.
    config = habitat.get_config("pr2l_objectnav.yaml")
    
    # 2. 데이터를 저장할 폴더 준비
    save_dir = "offline_data"
    os.makedirs(save_dir, exist_ok=True)
    print(f"데이터 저장 준비 완료: {save_dir}/")

    # 3. 시뮬레이터 환경 초기화
    with habitat.Env(config=config) as env:
        print("시뮬레이터 초기화 성공! 데이터 수집을 시작합니다.\n")
        
        # 총 에피소드 개수 확인
        num_episodes = len(env.episodes)
        
        # 4. 각 집(에피소드)을 돌면서 데이터 수집
        for i in range(num_episodes):
            observations = env.reset()
            episode = env.current_episode
            print(f"[{i+1}/{num_episodes}] 씬(Scene) 입장: {episode.scene_id.split('/')[-1]}")
            
            # 최신 버전의 길 찾기 도우미 (return_cg 옵션 제거됨)
            follower = ShortestPathFollower(env.sim, 0.2)
            
            rgb_frames = []
            actions = []
            step_cnt = 0
            max_steps = 500  # 무한 루프 방지용 (yaml 설정과 동일하게)
            
            while step_cnt < max_steps:
                # 현재 로봇이 보는 화면 저장
                rgb_frames.append(observations["rgb"])
                
                # 목표 지점(정답)까지 가기 위한 다음 행동 계산
                # ObjectNav는 목표물이 여러 개일 수 있으므로 첫 번째(0) 목표를 향해 감
                best_action = follower.get_next_action(episode.goals[0].position)
                
                # 최신 버전 호환성 처리 (None 이거나 0번 정지 명령인 경우 종료)
                if best_action is None:
                    break
                
                # 배열이든 튜플이든 순수 행동 숫자(int)만 추출
                if isinstance(best_action, (tuple, list, np.ndarray)):
                    action_idx = int(best_action[0])
                else:
                    action_idx = int(best_action)
                
                if action_idx == 0:  # 0은 STOP 행동
                    break
                    
                # 행동 기록 (나중에 딥러닝 모델 정답지로 사용)
                actions.append(action_idx)
                
                # 최신 버전 문법: 딕셔너리 형태로 행동 전달
                observations = env.step({"action": action_idx})
                step_cnt += 1
            
            # 5. 한 에피소드가 끝나면 numpy 파일로 디스크에 저장
            # 모양: (프레임수, 세로, 가로, 3채널)
            if 0 < step_cnt < 500:  # 0보다 크고 500보다 작아야 '성공적인 최단 경로'임
                np.save(f"{save_dir}/ep_{i}_rgb.npy", np.array(rgb_frames))
                np.save(f"{save_dir}/ep_{i}_actions.npy", np.array(actions))
                print(f"✅ 에피소드 {i} 저장 성공!")
            else:
                print(f"❌ 에피소드 {i} 실패 (데이터 오염 방지를 위해 버림)")
            
            print(f"  👉 수집 완료! (총 {step_cnt} 프레임 저장됨)")

if __name__ == "__main__":
    main()