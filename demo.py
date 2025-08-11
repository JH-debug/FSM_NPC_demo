import gradio as gr
import yaml
import openai
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langsmith import traceable
from langsmith import Client
from openai import OpenAI


load_dotenv()
client = Client()

openai_client = openai.OpenAI()


def get_default_emotion_state():
    return {"호감도": 50, "신뢰도": 50}

def get_likeability_level(score):
    if score <= 19: return "적대적"
    if score <= 39: return "경계"
    if score <= 69: return "중립적"
    if score <= 89: return "우호적"
    return "신뢰"


def format_emotion_for_display(emotion_state):
    if not emotion_state: return ""
    likeability = emotion_state.get("호감도", 50)
    level = get_likeability_level(likeability)
    
    # 각 점수를 프로그레스 바 형태로 시각화
    lines = [f"### ❤️ NPC의 마음: {level}\n"]
    for key, value in emotion_state.items():
        # 프로그레스 바 생성 (10칸 기준)
        filled_blocks = int(value / 10)
        empty_blocks = 10 - filled_blocks
        progress_bar = "🟩" * filled_blocks + "⬜️" * empty_blocks
        lines.append(f"- **{key}**: {progress_bar} ({value}/100)")
    return "\n".join(lines)



def parse_llm_response(reply_full):
    # 기본값 설정
    reply = reply_full.strip()
    next_state = None
    emotion_changes = {}

    # 호감도 변화 파싱
    if "호감도 변화:" in reply:
        reply, change_part = reply.rsplit("호감도 변화:", 1)
        # 예: "호감도 +10 (친절한 말투), 신뢰도 -5 (의심)"
        # 정규표현식을 사용하여 key, sign, value를 추출
        changes = re.findall(r'(\w+)\s*([+\-])\s*(\d+)', change_part)
        for key, sign, value in changes:
            delta = int(value) if sign == '+' else -int(value)
            emotion_changes[key] = delta

    if "다음 상태:" in reply:
        reply, state_part = reply.rsplit("다음 상태:", 1)
        next_state = state_part.strip()
    
    return reply.strip(), next_state, emotion_changes


def apply_emotion_changes(current_emotion, changes):
    new_emotion = current_emotion.copy()
    for key, delta in changes.items():
        # get의 두 번째 인자로 기본값을 설정하여 없는 키에 대한 에러 방지
        current_score = new_emotion.get(key, 50) 
        new_emotion[key] = max(0, min(100, current_score + delta))
    return new_emotion


def load_character_prompt(path="prompt/character_default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["Character_persona"]


def load_session_with_fsm(path="prompt/session_default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    session_prompt = data.get("session_setting", "").strip()
    fsm = data.get("fsm", {})
    initial_state = fsm.get("initial_state", fsm["states"][0]["name"])
    return session_prompt, fsm, initial_state


def save_new_session_prompt(yaml_text, folder="prompt/session_prompts"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"session_{timestamp}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return path


def list_saved_sessions(folder="prompt/session_prompts"):
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.endswith(".yaml")])


def generate_session_prompt_with_fsm():
    system_prompt = """너는 호감도 기반 NPC 상호작용 게임의 기획자이다.  
NPC와 세계관을 바탕으로, 플레이어의 선택에 따라 호감도가 변하고 대화 흐름이 바뀌는 세션 프롬프트(session_setting)와 상태 기반 FSM(fsm)을 YAML 형식으로 생성하라.
FSM은 상태(name, description)와 전이 조건(condition, next)을 포함하라."""

    user_prompt = """다음 내용을 포함한 YAML을 생성하세요:
- session_setting: 세션 배경 및 목적
- fsm:
    initial_state: FSM의 초기 상태 이름
    states: 상태(name), 설명(description), 전이(transitions)

출력은 다음 두 필드를 포함하는 YAML이어야 함:
- session_setting: |
    (여기에 설명, session_title, player_role, location, atmosphere, session_goal, player_state, npc_behavior_guidelines 등의 내용을 포함할 것)
- fsm:
    initial_state: GREETING
    states:
      - name: GREETING
        description: ...
        transitions:
          - condition: ... 
            next: ...
    
캐릭터 요약:
- NPC는 시간의 감시자이며 크로노스탑을 지키는 사명을 지님
- 모험가에게 파편화된 순간을 회수하고 시간의 균열을 봉합하게 하려는 목적을 가지고 있음

세계 상황 예:
- 시간의 균열이 세계 전역에 확산 중
- 중요한 순간들이 소멸하고 있음

대화 상황 예시:
- 플레이어가 처음 NPC를 만나는 장면
- 시간 균열 속에서 함께 탈출한 직후

YAML 형식으로 한국어로 출력해줘.
"""

    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def build_system_prompt(char_prompt, session_prompt, current_state, fsm, emotion_state, npc_memory=None):
    state_info = next((s for s in fsm['states'] if s['name'] == current_state), {})
    transitions = state_info.get("transitions", [])
    
    transition_text = "\n".join(
        [f"- {t['condition']} → {t['next']}" for t in transitions]
    )

    likeability_level = get_likeability_level(emotion_state.get("호감도", 50))
    emotion_text = "\n".join([f"- {k}: {v}" for k, v in emotion_state.items()])

    memory_text = ""
    if npc_memory:
        recent = npc_memory[-1]
        memory_text = f"\n\n### 지난 기억\n- {recent['summary']}\n- 감정: {recent['emotion']}\n- 태그: {', '.join(recent['tags'])}"

    fsm_text = f"""
### 현재 게임 정보
- **FSM 상태**: {current_state} ({state_info.get('description', '설명 없음')})
- **플레이어와의 관계**: {likeability_level}
- **세부 호감도 수치**:
{emotion_text}

### 너의 임무
1.  **페르소나 유지**: 너는 'NPC'이며, 현재 플레이어와의 관계({likeability_level})에 맞는 톤으로 대화해야 한다.
2.  **상태 전이**: 사용자의 마지막 말에 가장 적합한 다음 상태를 아래 '가능한 상태 전이' 목록에서 선택해야 한다.
3.  **호감도 변화 판단**: 사용자의 말이 긍정적, 부정적, 중립적인지 판단하여 호감도와 신뢰도를 변경해야 한다. 변화량은 -10에서 +10 사이가 적절하다.
4.  **응답 형식 준수**: 모든 응답 끝에 다음 형식을 반드시 포함해야 한다.

### 가능한 상태 전이 조건:
{transition_text}

응답 형식:
(대화 응답 내용)

다음 상태: (FSM의 다음 상태 이름)
호감도 변화: 호감도 +5 (이유), 신뢰도 -5 (이유)
"""
    return f"{char_prompt.strip()}\n\n{memory_text}\n\n{session_prompt.strip()}\n\n{fsm_text.strip()}"


@traceable
def chat_with_npc_fsm(message, history, char_prompt, session_prompt, current_state, fsm, emotion_state):
    system_prompt = build_system_prompt(char_prompt, session_prompt, current_state, fsm, emotion_state)
    print(system_prompt)

    messages = [{"role": "system", "content": system_prompt}]

    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": message})

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini", # "ax4",
        messages=messages,
        temperature=0.6
    )
    reply_full = response.choices[0].message.content

    reply, next_state_str, emotion_changes = parse_llm_response(reply_full)

    # 다음 상태 결정 (유효하지 않으면 현재 상태 유지)
    valid_states = [s['name'] for s in fsm['states']]
    if next_state_str and next_state_str in valid_states:
        next_state = next_state_str
    else:
        next_state = current_state

    # 호감도 업데이트
    updated_emotion = apply_emotion_changes(emotion_state, emotion_changes)
    history.append((message, reply))

    client.create_run(
        name="AlmaniaFSMChat",
        run_type="llm",
        inputs={"state": current_state, "user_input": message},
        outputs={"assistant_reply": reply, "next_state": next_state},
        tags=["fsm", f"from:{current_state}", f"to:{next_state}"],
        project_name="almania_fsm_chat"
    )

    return history, next_state, updated_emotion


def send_and_clear_fsm(user_msg, history, char_prompt, session_prompt, fsm_state, fsm_data):
    history, updated, next_state = chat_with_npc_fsm(
        user_msg, history, char_prompt, session_prompt, fsm_state, fsm_data
    )
    return updated, updated, "", next_state, f"### 🔄 현재 상태: `{next_state}`"


def fsm_to_mermaid(fsm_data, current_state=None):
    """
    Mermaid 플로우차트 문자열을 생성
    """
    if not isinstance(fsm_data, dict) or "states" not in fsm_data or not fsm_data.get("states"):
        return "```mermaid\nflowchart TD\n    ERROR[FSM 데이터가 올바르지 않거나 비어있습니다.];\n```"

    # 교훈 1: ID는 단순하고 안전하게 만듭니다.
    def make_safe_id(text):
        if not isinstance(text, str): text = str(text)
        # 알파벳, 숫자, 밑줄(_)만 허용합니다.
        safe_text = re.sub(r'[^a-zA-Z0-9_]', '', text)
        if not safe_text: return f"id_{hash(text)}"
        # 숫자로 시작하는 것을 방지합니다.
        if safe_text[0].isdigit(): safe_text = 's_' + safe_text
        return safe_text

    # 교훈 2 & 3: 라벨은 HTML 태그나 Markdown 특수문자 없이, 엔티티 코드를 사용합니다.
    def make_safe_label(text):
        if not isinstance(text, str): text = str(text)
        safe_text = text.replace('"', '"')
        safe_text = safe_text.replace('\n', '')
        return f'"{safe_text}"'

    lines = ["flowchart TD"]
    
    # --- 상태 노드 정의 ---
    states = fsm_data.get("states", [])
    for state in states:
        name = state.get("name")
        if not name: continue
        
        safe_id = make_safe_id(name)
        desc = state.get('description', '')
        
        # 이름과 설명을 파이썬의 \n으로 먼저 연결합니다.
        node_text_content = f"{name} \n{desc}"
        # 그 다음, 이 문자열을 안전한 최종 라벨로 변환합니다.
        label_content = make_safe_label(node_text_content)
        
        # 최종 포맷: nodeId["이름 설명"]
        lines.append(f'    {safe_id}[{label_content}]')

    # --- 전이(transition) 정의 ---
    for state in states:
        name = state.get("name")
        if not name: continue
        
        from_safe_id = make_safe_id(name)
        
        for t in state.get("transitions", []):
            next_state = t.get("next")
            if not next_state: continue
            
            to_safe_id = make_safe_id(next_state)
            condition_text = make_safe_label(t.get("condition", "")) # 화살표 라벨도 안전하게 처리
            
            lines.append(f'    {from_safe_id} -->|{condition_text}| {to_safe_id}')
            
    # --- 현재 상태 스타일링 ---
    if current_state:
        current_safe_id = make_safe_id(current_state)
        lines.append(f"    style {current_safe_id} fill:#f9f,stroke:#333,stroke-width:4px")
        
    mermaid_code = "```mermaid\n" + "\n".join(lines) + "\n```"
    # 디버깅을 위해 터미널에 출력
    print("--- Generated Mermaid Code ---")
    print(mermaid_code)
    print("---------------------------------------------")
    return mermaid_code


@traceable
def summarize_session(history, final_emotion, npc_memory):
    conversation_text = "대화 기록: " + "\n".join([f"플레이어: {u}\nNPC: {parse_llm_response(a)[0]}" for u, a in history])
    emotion_summary = "호감도 상태 요약: " + "\n".join([f"{k}: {v}" for k, v in final_emotion.items()])
    system_msg = """당신은 기억의 정령 ‘NPC’입니다. 당신은 코키투스의 문지기로서, 인간들과의 상호작용에서 감정의 흔적을 수집합니다. 아래는 당신과 한 모험가 사이의 대화 기록입니다.

이 대화를 기반으로, NPC 특유의 **약간 엉뚱하고 몽환적인 말투**로 **짧은 총평**을 작성하세요.

총평은 다음 조건을 따라야 합니다:
- 존대말을 사용하세요.
- 말투는 너무 시적이거나 고풍스럽지 말고, 감성적이되 직관적이고 귀엽게 표현하세요.
- NPC의 속마음을 혼잣말처럼 적어주세요.
- 너무 정리된 느낌 없이, 살짝 들뜬 감정이나 아쉬움, 기분을 담아주세요.
- 3문장 이내로 간결하게 작성하세요.

포함할 내용:
- 플레이어가 어떤 사람처럼 느껴졌는지 (예: 낯가리는 고양이 같은 분이셨어요)
- 대화를 통해 느낀 감정의 변화
- 앞으로의 관계에 대한 기대나 걱정

분석적이거나 진지한 표현은 피하세요."""
    
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": conversation_text + '\n' + emotion_summary},
        ],
        temperature=0.5
    )

    summary = response.choices[0].message.content.strip()
    memory_entry = {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "summary": summary,
        "emotion": final_emotion,
        "tags": []
    }

    npc_memory.append(memory_entry)
    dropdown_choices = list_memory_options(npc_memory)

    return summary, npc_memory, gr.update(choices=dropdown_choices)


# Gradio 인터페이스 
with gr.Blocks(theme=gr.themes.Soft(), title="NPC NPC 대화 시스템") as demo:
    gr.Markdown("## NPC와의 대화 세션")

    def load_and_update_all_states(yaml_text):
        try:
            data = yaml.safe_load(yaml_text)
            if not isinstance(data, dict): raise ValueError("YAML 형식이 올바르지 않습니다.")
            session_prompt = data.get("session_setting", "")
            fsm = data.get("fsm")
            if not isinstance(fsm, dict) or "states" not in fsm or not fsm["states"]:
                raise ValueError("FSM 데이터가 없거나 'states' 리스트가 비어있습니다.")
            initial_state = fsm.get("initial_state") or fsm.get("initial_states") or fsm["states"][0].get("name")
            if not initial_state: raise ValueError("초기 상태를 찾을 수 없습니다.")
            
            initial_emotion = get_default_emotion_state()
            mermaid_chart = fsm_to_mermaid(fsm, initial_state)
            state_display = f"### 🧭 현재 상태: `{initial_state}`"
            emotion_display_text = format_emotion_for_display(initial_emotion)
            status = f"✅ 세션 로드 완료! 초기 상태: '{initial_state}'"
            
            return yaml_text, session_prompt, fsm, initial_state, initial_emotion, [], [], state_display, emotion_display_text, mermaid_chart, status
        except Exception as e:
            error_msg = f"❌ YAML 오류: {e}"
            print(error_msg)
            return (gr.skip(),) * 11


    # --- 내부 상태(State) 정의 ---
    char_prompt_state = gr.State(load_character_prompt())
    session_prompt_state = gr.State()
    fsm_data_state = gr.State()
    fsm_state = gr.State()
    emotion_state = gr.State()
    history_state = gr.State([])
    npc_memory_state = gr.State([])


    with gr.Row():
        with gr.Column(scale=2):
            chatbox = gr.Chatbot(label="💬 NPC와의 대화", height=500, bubble_full_width=False)
            user_input = gr.Textbox(label="당신의 말", placeholder="여기에 메시지를 입력하세요...")
            with gr.Row():
                send_btn = gr.Button("전송")
                clear_btn = gr.Button("대화 초기화")
            fsm_state_display = gr.Markdown(label="🧭 현재 상태")
            fsm_chart_display = gr.Markdown(label="📊 상태 전이 차트")
            
        with gr.Column(scale=1):
            emotion_display = gr.Markdown(label="❤️ NPC의 마음")
            with gr.Tabs():
                with gr.TabItem("📝 세션 설정 (YAML)"):
                    session_editor = gr.Code(language="yaml", lines=25)
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 자동 생성 (GPT)")
                        save_btn = gr.Button("💾 현재 설정 저장")
                    session_selector = gr.Dropdown(label="저장된 세션 불러오기", choices=list_saved_sessions())
                    status_msg = gr.Markdown()
                # with gr.TabItem("📊 상태 전이 차트 (Mermaid)"):
                #     fsm_chart_display = gr.Markdown()

    with gr.Column():
        summarize_btn = gr.Button("📘 세션 요약 및 총평 보기")
        summary_output = gr.Textbox(label="📘 세션 요약 및 NPC의 총평", lines=5)

    memory_dropdown = gr.Dropdown(label="저장된 기억", choices=[], interactive=True)
    memory_summary_box = gr.Textbox(label="기억 요약", lines=4)
    memory_tags_box = gr.Textbox(label="태그 (쉼표로 구분)", placeholder="예: 조용한, 경계심 많음")
    save_memory_btn = gr.Button("💾 기억 수정 저장")

    # 연결: 버튼 클릭 시 요약 생성 + 상태 업데이트 + 드롭다운 갱신
    summarize_btn.click(
        fn=summarize_session,
        inputs=[history_state, emotion_state, npc_memory_state],
        outputs=[summary_output, npc_memory_state, memory_dropdown]
    )

    all_states_outputs = [session_editor, session_prompt_state, fsm_data_state, fsm_state, emotion_state, chatbox, history_state, fsm_state_display, emotion_display, fsm_chart_display, status_msg]

    # 앱 로드 시 초기 YAML 파일로 모든 상태 초기화
    def initial_load():
        try:
            with open("prompt/session_default.yaml", "r", encoding="utf-8") as f:
                initial_yaml_text = f.read()
        except FileNotFoundError:
            initial_yaml_text = "Session_setting: '기본 세션 파일(session_default.yaml)을 찾을 수 없습니다.'"
        return load_and_update_all_states(initial_yaml_text)
    
    demo.load(fn=initial_load, outputs=all_states_outputs)

    # 대화 전송
    def process_chat_and_update_ui(user_msg, history, char_prompt, session_prompt, current_fsm_state, fsm_data, current_emotion):
        # 입력값이 비었을 때, 모든 outputs 개수에 맞춰 gr.skip() 반환
        if not user_msg or not user_msg.strip():
            return (gr.skip(),) * 8

        updated_history, next_state, updated_emotion = chat_with_npc_fsm(
            user_msg, history, char_prompt, session_prompt, current_fsm_state, fsm_data, current_emotion
        )
        
        # UI 업데이트용 데이터 생성
        # 챗봇에는 순수한 대화만 표시
        chatbot_display = [(hist[0], parse_llm_response(hist[1])[0]) for hist in updated_history]
        state_display = f"### 🧭 현재 상태: `{next_state}`"
        emotion_display_text = format_emotion_for_display(updated_emotion)

        mermaid_chart = fsm_to_mermaid(fsm_data, next_state)
        
        # 반환값 순서: [chatbox, history, user_input, fsm_state, emotion_state, state_display, emotion_display]
        return chatbot_display, updated_history, "", next_state, updated_emotion, state_display, emotion_display_text, mermaid_chart

    chat_inputs = [user_input, history_state, char_prompt_state, session_prompt_state, fsm_state, fsm_data_state, emotion_state]
    chat_outputs = [chatbox, history_state, user_input, fsm_state, emotion_state, fsm_state_display, emotion_display, fsm_chart_display]
    send_btn.click(fn=process_chat_and_update_ui, inputs=chat_inputs, outputs=chat_outputs)
    user_input.submit(fn=process_chat_and_update_ui, inputs=chat_inputs, outputs=chat_outputs)
    
    # 에디터 수정 시 모든 상태 업데이트
    session_editor.change(fn=load_and_update_all_states, inputs=[session_editor], outputs=all_states_outputs)
    
    # 저장된 세션 불러오기
    def load_from_selector(filename):
        if not filename: return gr.skip()
        filepath = os.path.join("prompt/session_prompts", filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return load_and_update_all_states(f.read())
    session_selector.change(fn=load_from_selector, inputs=[session_selector], outputs=all_states_outputs)

    # GPT로 새로 생성하기
    def generate_and_update():
        return load_and_update_all_states(generate_session_prompt_with_fsm())
    refresh_btn.click(fn=generate_and_update, outputs=all_states_outputs)
    
    # 설정 저장하기
    def save_and_update_list(yaml_text):
        path = save_new_session_prompt(yaml_text)
        return gr.update(choices=list_saved_sessions()), f"✅ 저장됨: {os.path.basename(path)}"
    save_btn.click(fn=save_and_update_list, inputs=[session_editor], outputs=[session_selector, status_msg])
    
    # 대화 초기화
    def reset_chat_session(fsm_data):
        # fsm_data가 유효하지 않을 때의 예외 처리
        if not isinstance(fsm_data, dict):
            return (gr.skip(),) * 7
            
        initial_state = fsm_data.get("initial_state") or fsm_data.get("initial_states")
        if not initial_state and fsm_data.get("states"):
            initial_state = fsm_data["states"][0].get("name", "N/A")
        
        # 호감도 상태도 기본값으로 초기화
        initial_emotion = get_default_emotion_state()
        
        # UI 업데이트용 텍스트 생성
        state_display_text = f"### 🧭 현재 상태: `{initial_state}`"
        emotion_display_text = format_emotion_for_display(initial_emotion)
        mermaid_chart = fsm_to_mermaid(fsm_data, initial_state)

        # 반환값 순서: [chatbox, history, fsm_state, emotion_state, state_display, emotion_display, mermaid_chart]
        return [], [], initial_state, initial_emotion, state_display_text, emotion_display_text, mermaid_chart
    
    clear_btn.click(fn=reset_chat_session, inputs=[fsm_data_state], outputs=[chatbox, history_state, fsm_state, emotion_state, fsm_state_display, emotion_display, fsm_chart_display])


    def list_memory_options(npc_memory):
        return [f"{m['session_id']} - {m['summary'][:15]}..." for m in npc_memory]


    def load_memory_entry(memory_label, npc_memory):

        if isinstance(memory_label, list):
            memory_label = memory_label[0] if memory_label else ""

        match = next((m for m in npc_memory if memory_label.startswith(m['session_id'])), None)
        if not match:
            return "", ""
        
        tags = ", ".join(match.get("tags", []))
        return match["summary"], tags


    def save_memory_entry(memory_label, summary_text, tag_text, npc_memory):
        if isinstance(memory_label, list):
            memory_label = memory_label[0] if memory_label else ""

        for mem in npc_memory:
            if memory_label.startswith(mem["session_id"]):
                mem["summary"] = summary_text.strip()
                mem["tags"] = [t.strip() for t in tag_text.split(",") if t.strip()]
        
        return list_memory_options(npc_memory), "✅ 기억이 저장되었습니다."
        

    memory_dropdown.change(
        fn=load_memory_entry,
        inputs=[memory_dropdown, npc_memory_state],
        outputs=[memory_summary_box, memory_tags_box]
    )

    save_memory_btn.click(
        fn=save_memory_entry,
        inputs=[memory_dropdown, memory_summary_box, memory_tags_box, npc_memory_state],
        outputs=[memory_dropdown, status_msg]
    )


demo.launch(share=True)
