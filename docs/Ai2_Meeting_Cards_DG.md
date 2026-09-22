# Ai2 Meeting — Easy English Cards (DG)

2026-09-11 초안 · **2026-09-13 기술 표현(T9~T14)·기능/채용 출처 교정**

> [최종 PR 준비표](PR_REENTRY_2026_09_10.md)를 함께 읽는다. 이번에는 이 Markdown만 갱신했고
> meeting_cards.html/공개 페이지는 동기화하지 않았다. 다른 실험 수치·개인 이력·실제 미팅 내용은 재검증하지 않았다.

**미팅 시작 문장**

> Thanks for making time. I'm happy to share what I built and what I learned. Is there a particular part you'd like to start with?

---

## 핵심 8장 — 나·왜·무엇 / 어떻게·근거·배움 / 기여·다음

| 카드 | 한국어 기억 단어 | 막히면 말할 한 문장 | 떠올릴 증거 |
|---|---|---|---|
| ① WHO | 나: 개발과 연구를 연결 | My background combines software, AI deployment, and research. | 창업·Rebellions·GIS |
| ② WHY | 왜: 사용자를 이해하는 일 | I want my engineering work to help people with real problems. | 우간다 시스템 기획·ETH |
| ③ WHAT | 무엇: 어디를 먼저 볼지 | I built Rasuwa to help people choose where to look first. | 전후 관측·지도·검토 후보 |
| ④ HOW | 어떻게: 입력부터 비교까지 | I started with frozen embeddings and built the workflow around them. | 입력 처리·전후 비교·레이더 |
| ⑤ EVIDENCE | 근거: 측정한 만큼 말하기 | The clearest result was better coverage, with more validation still needed. | 같은 100개 중 47→97 |
| ⑥ LEARNING | 배움: 문제를 다음 개선으로 | I'd like to turn what I learned into checks and examples for other users. | scaling 수정·입력 검사 제안 |
| ⑦ FIT | 기여: 운영과 연구를 함께 | I can bring experience in production systems, research, and public-sector projects. | 서비스 운영·모델 배포·파트너 |
| ⑧ NEXT | 다음: 관심 역할과 연결 | I'm interested in the role and would like to speak with the hiring team. | 관심 업무·기술 사례·CV |

### ① WHO — 나: 개발과 연구를 연결

**받을 질문:** Tell me about yourself. / What's your background? / What do you do now?

**기억할 단어:** STARTUP → DEPLOYMENT → RESEARCH → IMPACT

**기본 답변 — 이것을 외우기**

> Hi, I'm Donggeun, but please call me DG. I've worked in software and ML engineering for over ten years. I co-founded a startup and led engineering for about eight years. We built a professional community, something like LinkedIn for Korea, and a side-job platform. Now I work at Rebellions, the largest NPU startup in Korea. My role is AI deployment at scale, delivering models to users. Alongside that, I do policy and geospatial research, and I'm doing a master's at ETH Zurich. That study made me even more interested in using AI and remote sensing for real environmental problems. I enjoy turning complex problems into practical tools people can use, especially where AI can have a real impact, in areas like development aid and Earth observation.

**짧은 버전 — 45초 안에 끝내야 할 때**

> Hi, I'm Donggeun, please call me DG. I've worked in software and ML engineering for over ten years. I co-founded a startup and led engineering for about eight years. Now I work at Rebellions, the largest NPU startup in Korea, on AI deployment at scale, delivering models to users. I also do policy and geospatial research, and my master's at ETH Zurich made me even more interested in environmental problems. I enjoy turning complex problems into practical tools people can use, especially where AI can make a real impact on environmental and development problems.

**Ai2 팬 이야기 — 자연스럽게 나올 때만**

> Separately, I've been a big fan of Ai2 for a long time. I learned a lot from ELMo and OLMo.
>
> When OlmoEarth came out last year, I was building a marine model on the side, for drone fishing that avoids dolphins. So I started using OlmoEarth right away.

**한국어 뜻:** 소프트웨어·ML 엔지니어로 10년 이상. 창업해서 8년간 엔지니어링 리드(한국판 링크드인 같은 커뮤니티, 사이드잡 플랫폼). 지금은 한국 최대 NPU 스타트업 Rebellions에서 대규모 AI 배포, 사용자에게 모델을 전달하는 역할. 병행으로 정책·공간 연구와 ETH 석사. 석사를 하면서 AI·원격탐사로 실제 환경 문제를 푸는 데 더 관심을 갖게 됐다. 복잡한 문제를 쓸 수 있는 도구로 만드는 걸 좋아하고, 특히 개발협력(ODA)·지구관측처럼 AI가 실제 임팩트를 낼 수 있는 영역에 관심.

**추가 답변 — 질문에 맞는 것 하나**

- **What do you do at Rebellions?**
  > I work on model deployment and serving infrastructure for AI chips. That includes working with tools such as vLLM, Triton, Airflow, and Kubernetes.
- **What did you do at your startup?**
  > I co-founded a professional community platform and led engineering. My work included product development, service operations, and the technical integration of Careerly after its acquisition.
- **How much remote-sensing experience do you have?**
  > My deepest experience is in software and data systems. I bring GIS experience and hands-on Earth observation experiments, and I'm continuing to deepen my understanding of the sensors.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ② 동기로: The kind of problem I work on has become more important to me.
- ③ 프로젝트로: Rasuwa is a recent example of how I bring those areas together.

---

### ② WHY — 왜: 사용자를 이해하는 일

**받을 질문:** Why this kind of work? / Why environmental AI? / What motivates you?

**기억할 단어:** UGANDA → PEOPLE → ENVIRONMENT

**기본 답변 — 이것을 외우기**

> One experience that shaped my interests was helping plan a hospital information system in Uganda. I worked on understanding what different people needed from the system. It showed me how much useful technology depends on understanding the people who will use it. I'm now studying regenerative systems at ETH and exploring how AI can support environmental work.

**한국어 뜻:** 우간다 병원 시스템을 기획하면서 여러 이해관계자의 필요를 이해하는 일이 중요하다는 것을 배웠다. 지금은 ETH에서 regenerative systems를 공부하며 그 관심을 환경 분야의 AI로 이어가고 있다.

**추가 답변 — 질문에 맞는 것 하나**

- **What was your role in Uganda?**
  > My role was in the feasibility study and needs assessment. I helped turn different expectations into a clearer plan for the system.
- **How did you handle different expectations?**
  > The Korean and Ugandan stakeholders had different priorities. I helped clarify those priorities and work toward a shared direction for the system.
- **How does your policy research connect to this?**
  > I've worked with legislative, migration-policy, and urban data. Those projects taught me to pay attention to missing data, measurement choices, and what a result can actually support.
- **Why Earth observation specifically?**
  > It offers a way to study places where local data is limited. I'm interested in when those measurements become useful enough to support a real decision.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ③ 프로젝트로: That interest led me to explore Earth observation through Rasuwa.

---

### ③ WHAT — 무엇: 어디를 먼저 볼지

**받을 질문:** What did you build? / Why did you build it? / What's the use case? / Who is this for?

**기억할 단어:** OPEN → CURIOSITY (OCEAN · JEJU) → NEPAL → MAP

**왜 만들었는가 — 이야기 (짧게, 상대가 관심 있을 때)**

> I'm a huge fan of OlmoEarth and its vision. Everything is open, so I've been trying many things with it.
>
> First, I looked at damage to ocean ecosystems.
>
> Then I studied my home island, Jeju. It's a World Natural Heritage site, with small volcanic hills called oreum. I compared images year by year to see how the environment was changing, and I visited some places to check for myself. That was pure curiosity.
>
> Along the way, I also wrote a paper. It's now under review at AAAI.
>
> Then the disaster in Nepal happened. There was a lot on Twitter, and a lot of sensational news. I wanted to see it scientifically: where did it start, and how far did the effects reach?
>
> So I first tried prediction. Could I find similar areas in the Himalayas at risk? That didn't work.
>
> Then I saw that roads and communication were cut off, so it was very hard to visit. I thought it would help to compare the terrain before and after, and show where the risky places are and how far the damage reaches.

**기본 답변 — 이것을 외우기**

> That's why I built Rasuwa, a map-based tool for reviewing changes after a disaster in Nepal. It uses OlmoEarth embeddings to compare satellite images from before and after the event. I started with optical images and later added radar. The goal is to help someone choose where to look first. My next step is to test that workflow with domain users.

**한국어 뜻:** OlmoEarth의 오픈 비전이 좋아서 이것저것 써봤다. 바다 생태계 훼손, 고향 제주의 오름 연도별 비교와 현장 확인(순수한 호기심), 그 과정에서 AAAI 심사 중인 논문. 네팔 재난은 트위터 정보와 자극적 뉴스가 많아서, 어디서 시작해 어떤 여파가 있었는지 과학적으로 보고 싶었다. 처음엔 히말라야의 비슷한 위험 지역 예측을 시도했지만 안 됐고, 도로·통신이 끊겨 방문이 어렵다는 걸 보고 전후 지형 비교로 위험한 곳과 피해 범위를 보여주자고 생각해 Rasuwa를 만들었다. 광학에서 시작해 레이더를 추가했고, 다음은 실제 사용자 검증이다.

**추가 답변 — 질문에 맞는 것 하나**

- **Who would use it?**
  > I'm designing it for people reviewing disaster-related changes, such as researchers or analysts. I still need to work with domain users to test how it fits their actual review process.
- **What does the map help them do?**
  > It brings observations and change scores into one place, so someone can inspect a location and decide whether it needs further review.
- **Is it already being used in disaster response?**
  > I haven't established operational use by a response team. It's a public prototype, and I'd like to test it with people who do this kind of review.
- **데모에서 할 말**
  > This is one location to inspect.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ④ 구현으로: The main design choice was to start with frozen embeddings.
- ⑤ 결과로: The clearest improvement I measured was in data coverage.

---

### ④ HOW — 어떻게: 입력부터 비교까지

**받을 질문:** Why OlmoEarth? / How does it work? / Did you train a model?

**기억할 단어:** INPUTS → EMBEDDINGS → COMPARISON

**기본 답변 — 이것을 외우기**

> I started with frozen OlmoEarth embeddings, so I could explore the task without event-specific training labels. I compared observations from before and after the event, then added radar where optical data was limited. The main choices were how to prepare the inputs, compare observations, and decide which changes deserved review.

**한국어 뜻:** 해당 사건의 학습 라벨을 먼저 모으지 않고 탐색할 수 있도록 frozen embeddings로 시작했다. 중요한 판단은 입력을 어떻게 준비하고 비교하며, 어떤 변화를 검토 대상으로 고를지였다.

**추가 답변 — 질문에 맞는 것 하나**

- **What are embeddings?**
  > They are numerical representations of the images. I compare them across observations to look for changes that may deserve closer inspection.
- **Walk me through the pipeline.**
  > I prepare the satellite inputs, extract embeddings, compare pre- and post-event observations, and show the scores on a map. I also track missing observations and sensor sources.
- **What is your change score?**
  > I use one minus cosine similarity between pre- and post-event embeddings. A higher score means the representations differ more; it still needs interpretation.
- **Did you train models?**
  > The Rasuwa map uses frozen embeddings. In separate landslide experiments, I also trained decoders on frozen OlmoEarth features and compared them with task-specific models.
- **Why v1 Base rather than v1.2?**
  > I kept v1 Base fixed for the original analysis. In separate landslide comparisons, the v1 and v1.2 results were mixed. I'd validate an upgrade on the target task before changing the pipeline.
- **Why add radar?**
  > Optical observations were limited in parts of the area. Radar gave me additional observations to work with, while introducing its own input and interpretation issues.
- **Reproducibility 사례 (M27 pooling)**
  > In one training experiment, strict deterministic mode failed in a pooling operation. I checked the original architecture and tested an equivalent mean operation, comparing both outputs and gradients. That helped me preserve the intended computation while making the run reproducible.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ⑤ 검증으로: Once the workflow ran, the next question was what the results actually supported.

---

### ⑤ EVIDENCE — 근거: 측정한 만큼 말하기

**받을 질문:** Did it work? / What improved? / How did you validate it?

**기억할 단어:** BENCHMARK → COVERAGE → BASELINES → USERS

**왜 시작했는가 — 벤치마크 근거 (먼저 말하기)**

> Before I started, I checked the SEN12-Landslides benchmark. I compared OlmoEarth embeddings with NDWI, a classical index. OlmoEarth did better there. That was meaningful enough for me to start building Rasuwa.

**기본 답변 — 이것을 외우기**

> The clearest result was better coverage. In the same set of 100 windows, adding later radar observations increased the number I could score from 47 to 97. I also compared simple baselines and checked ranking consistency. I still need to test whether the results help domain users review an area.

**한국어 뜻:** 시작 근거: SEN12-Landslides 벤치마크에서 OlmoEarth 임베딩이 고전 지수 NDWI보다 잘 나와서 충분히 유의미하다고 보고 시작했다. 가장 분명하게 개선된 것은 관측 범위다. 같은 100개 영역 중 점수를 계산할 수 있는 영역이 47개에서 97개로 늘었다. 단순한 방법과도 비교하고 순위 일관성을 살폈으며, 실제 사용자의 검토에 얼마나 도움이 되는지는 다음 검증 과제다.

**추가 답변 — 질문에 맞는 것 하나**

- **Why did you think it was worth starting?**
  > On the SEN12-Landslides benchmark, OlmoEarth embeddings did better than NDWI. That's a different experiment from the Nepal flood check, where NDWI did better on one measure. So I say which experiment I'm talking about.
- **Does 97 mean 97% accuracy?**
  > It means I could score 97 of the 100 windows. That measures observation coverage. The radar observations also came from later dates.
- **Were the rankings consistent?**
  > Two radar tracks shared nine of their top ten locations. That's encouraging consistency within this event, but it doesn't establish that those locations are confirmed damage.
- **Did OlmoEarth beat simpler methods?**
  > The results were mixed. A simple water index did better on one precision-recall measure. That helped me clarify where embeddings might add value and what still needs testing.
- **What validation did you use?**
  > I compared simple baselines, external reference layers, and results across radar tracks. Each answers a different question. Ground labels and user testing are still needed.
- **What would you test next?**
  > I'd define one user's review task and compare methods at the same review budget. I'd also use separate dates or areas to test the thresholds and check the errors with domain experts.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ⑥ 배운 점으로: Those checks also showed me where the workflow needed improvement.

---

### ⑥ LEARNING — 배움: 문제를 다음 개선으로

**받을 질문:** What was difficult? / How was your platform experience? / What would help?

**기억할 단어:** ISSUE → CORRECTION → REUSABLE CHECK

**기본 답변 — 이것을 외우기**

> I learned that useful results depend on the whole workflow: preparing the data, checking simple baselines, and reviewing errors. In one experiment, a radar scaling issue changed the result, so I corrected the analysis. I'd like to turn those lessons into small input checks and clear examples, so other users can avoid the same problems.

**이번에 전 과정을 겪어본 소감 — 기본 답변 다음에**

> This time I went through the whole process, from fine-tuning to evaluation. So I found a few issues that are worth improving, and I'm planning to send pull requests on GitHub.
>
> It's a really good model. Where I think meaningful improvements are possible is usability, the steps around the model that every user has to repeat.

**한국어 뜻:** 입력 준비, 단순 기준선 비교, 오류 검토가 모두 결과의 유용성에 영향을 줬다. scaling 문제를 수정한 경험을 다음 사용자를 위한 검사와 예제로 연결하고 싶다. 주의: PR은 아직 안 보냈으니 “I'm planning to”로. fine-tuning은 별도 landslide 실험 이야기이고 Rasuwa 지도는 frozen embeddings라는 구분은 유지.

**추가 답변 — 질문에 맞는 것 하나**

- **How was your platform experience?**
  > Most of my hands-on work has been with the open model and my own pipeline. I can share concrete feedback from that workflow, and I'd like to understand how Studio handles the same steps.
- **What would you share with the team?**
  > A small example of the input that caused trouble, the expected behavior, and a way to reproduce it. That would help us decide whether the fix belongs in validation, documentation, or the application.
- **What documentation would help?**
  > I saw the existing embedding examples. I'd be interested in guidance on seasonal change, missing observations, and threshold validation. Those were important parts of my workflow.
- **How do you use AI in your work?**
  > I use AI tools to help with coding and analysis. I still check the inputs, compare against simple baselines, and verify important claims against the actual outputs.
- **상대에게 돌려줄 질문 1**
  > When a new partner gets stuck, is it usually the data, the model, or understanding the output?
- **상대에게 돌려줄 질문 2**
  > How do lessons from a partner project turn into changes in rslearn or Studio?

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ⑦ 기여로: That kind of improvement is work I'd enjoy contributing to.

---

### ⑦ FIT — 기여: 운영과 연구를 함께

**받을 질문:** Why Ai2? / How does your background fit? / What would you contribute?

**기억할 단어:** BUILD → OPERATE → IMPROVE

**기본 답변 — 이것을 외우기**

> I'm interested in Ai2 because the team works closely with people using AI for environmental problems. My background combines building services, deploying models, and working on public-sector research. I'd like to bring those skills together here. I'm especially interested in turning problems from real projects into tools and workflows that other teams can use too.

**한국어 뜻:** 서비스 개발, 모델 배포, 공공 연구를 함께 해온 경험을 환경 분야에 쓰고 싶다. 개별 프로젝트에서 발견한 문제를 다른 팀도 활용할 수 있는 도구와 방법으로 만드는 일에 관심이 있다.

**추가 답변 — 질문에 맞는 것 하나**

- **역할별 한 문장 — Research Engineer**
  > I'd like to help partners adapt the models, evaluate the results, and turn successful approaches into reusable examples.
- **역할별 한 문장 — Agent Platform**
  > I'd like to help developers get from an idea to an agent they can run, inspect, and improve.
- **Agent Platform에 대해 물어보기 (질문)**
  > I saw that you're building an agent platform. That's close to what I do at work, deploying and serving models. May I ask what you're planning to use it for, and who the first users would be?
- **What would you do first? — Research Engineer**
  > I'd start with one partner's use case and reproduce the main blocker. Then I'd deliver a tested workflow and a small reusable improvement, such as an input check or a task-specific example.
- **What would you do first? — Agent Platform**
  > I'd follow one developer from setup to a running agent. I'd fix the biggest blocker and make failures easier to inspect before expanding the platform.
- **How would you scale Rasuwa?**
  > I'd separate data preparation from inference, cache reusable outputs, and make failed tasks safe to rerun. I'd track input and model versions, then measure the bottleneck before adding infrastructure.
- **Have you built production agents?**
  > My clearest experience is in production software and model serving. I've also worked on AI-assisted workflows. I can explain the specific components I built and how they were used.
- **How would you evaluate an agent?**
  > I'd start with real user tasks and compare versions on task success, tool errors, cost, and latency. I'd repeat runs where outputs vary and keep traces to understand regressions.
- **What research would you pursue?**
  > I'm interested in what happens when a model changes but a large embedding cache is expensive to refresh. I'd like to study how limited recomputation affects downstream estimates and uncertainty. That's the topic of my current manuscript.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- ⑧ 다음 단계로: That's why the opening caught my attention.

---

### ⑧ NEXT — 다음: 관심 역할과 연결

**받을 질문:** What would you like from this conversation? / Are you interested in joining us? / What next?

**기억할 단어:** INTEREST → EVIDENCE → INTRODUCTION

**기본 답변 — 이것을 외우기**

> I'm interested in the Senior Research Engineer role, especially the work on partner applications and model adaptation. My experience in deployment, research, and public-sector projects connects with that work. I'd be happy to share a short technical example and my CV. Would it make sense to speak with the hiring team?

**Agent Platform 교체형**

> I'm interested in the Agent Platform role, especially the work on runtime, APIs, and evaluation. My experience in production systems and AI deployment connects closely with that work. I'd be happy to share a concrete example and my CV. Would it make sense to speak with the hiring team?

**진심 + 함께 해볼 제안 — 기본 답변 다음에**

> Doing this project made me want to work with the OlmoEarth team even more. I'd love to do more case studies, or join a real project with you if there's a chance. I mean that.
>
> I'd also be happy to volunteer on case studies while I build up experience. Korea has useful satellite and public datasets, though their access and redistribution terms differ. We could design a reproducible example around data users are allowed to obtain. And if you want to bring your work to Asia, I could help with that.
>
> Many of your examples seem to focus on certain regions. If it helps, I can bring use cases from Asia. Here are three ideas.

**한국어 뜻:** 어떤 역할의 어떤 업무가 관심 있는지 분명히 말한다. 자신의 경험을 한 문장으로 연결하고, 기술 사례와 CV를 공유할 수 있다고 한 뒤 채용팀과의 대화를 요청한다. 아직 지원하지 않았으므로 “I applied”라고 쓰지 않는다. 채용: [공식 OlmoEarth Senior Research Engineer 공고](https://job-boards.greenhouse.io/thealleninstitute/jobs/8140098)의 2026-09-16 마감·Seattle은 9/13 확인했다. 마감 시각/시간대는 미기재. PR 완료를 지원 조건으로 두지 않는다. Agent Platform은 별도 역할이므로 지원 시 해당 공고를 다시 확인한다.

**추가 답변 — 질문에 맞는 것 하나**

- **Idea 1 — ODA agriculture monitoring (Cambodia)**
  > Monitor the results of aid-funded agriculture projects. The decision: which project sites need a follow-up survey? I bring ODA, policy evaluation, and data analysis.
- **Idea 2 — Road corridor check after a disaster (Nepal)**
  > Support checking road corridors after a disaster. The decision: which section should a field team visit first? This extends Rasuwa directly.
- **Idea 3 — Mangrove restoration follow-up (Indonesia)**
  > Follow up on mangrove restoration sites. The decision: which sites need inspection and repair? I bring regenerative systems and spatial analysis.
- **What's your biggest problem right now? (물어볼 질문)**
  > What's the hardest problem for your partners right now? Is it the data, the model, or understanding the output? I'd like to see where I could help.
- **Pain point 1 — the steps around the model**
  > The model itself was easy to use. The hard part was everything around it: preparing inputs, handling missing observations, and deciding thresholds. Those steps are where I'd like to see more guidance and small checks.
- **Pain point 2 — end-to-end examples**
  > The tutorials cover the basic steps well. I found existing application examples, but I couldn't find a worked example matching our change-review workflow, including input quality checks and external-reference validation. That's the specific gap I'd like to help with.
- **대화 초반에 목적을 물으면**
  > I'd like to share what I learned from Rasuwa, understand how it connects with your users, and see whether there's a useful next step.
- **두 역할 중 어디에 맞을지 조언을 구하면**
  > Both roles connect with my experience in different ways. I'd value your perspective on where my background would be most useful.
- **케이스 스터디를 제안받으면**
  > Yes, I'd be happy to contribute. I can share the use case, the workflow, the measured results, and the limitations, with the evidence behind each claim.
- **채용 담당자 연결을 약속받으면**
  > Thank you. I'll send a concise CV and the technical example most relevant to that role.
- **지금 채용 연결이 어렵다고 하면**
  > Thanks for letting me know. I'd still be happy to share the technical example if it's useful to the team.
- **왜 현재 직장에서 옮기고 싶은지**
  > My current work has given me valuable experience in AI deployment. I want to bring that experience closer to environmental applications and the people using them. That's what interests me about this team.

**연결 문장 — 상대가 더 듣고 싶어 할 때만**

- 마지막 인사: Thanks for your time. I enjoyed hearing how you're approaching these problems.

---

## 기술 질문 상세 — 비교 → 검증 → 네팔 적용 → UNOSAT 비교 → 작은 모델 → 개선 요청 6가지

### T1 두 방법은 무엇인가?

**키워드:** AI = embeddings · Classical = bands / NDVI · NBR

> I compared two methods. The AI method measures changes in OlmoEarth embeddings. The classical method measures changes in image bands, or in NDVI and NBR. I used the stronger classical result as the baseline.

- 기억할 차이: AI는 이미지의 표현을 비교하고, 고전 방법은 밴드 값이나 지수를 비교한다.

### T2 공정하게 비교했는가?

**키워드:** Same images · dates · labels · scoring — no training on test cases

> Both methods used the same images, dates, labels, and scoring method. I selected images without looking at the labels. I used the labels only for evaluation, with no training on these test cases.

- 여기서는 “이 시험 사례로 학습하지 않았다”까지 말하면 돼. “모델이 사전학습에서도 이 지역을 본 적 없다”는 별도 확인이 필요해.

### T3 AUROC는 무엇인가?

**키워드:** Ranking, not accuracy · 0.5 random · 1.0 perfect

> AUROC checks whether affected locations tend to get higher scores than unaffected locations. A score of 0.5 means random ranking. A score of 1.0 means perfect ranking.

- 분류 정확도보다 순위를 얼마나 잘 매겼는지 보는 지표라고 기억해. 정확한 확률 정의에서는 동점을 반점으로 처리해.

### T4 네팔에서 OlmoEarth를 어디에 썼는가?

**키워드:** images → embeddings → change scores → ranked locations · no fine-tuning

> For Nepal, I used OlmoEarth Base to extract before-and-after embeddings. I did this separately for optical and radar data. Then I measured the differences, compared them with normal pre-event changes, and ranked locations for review. I did not fine-tune the model.

- 외울 흐름은 images → embeddings → change scores → ranked locations야.
- OlmoEarth는 임베딩을 만드는 단계에 들어가고, 이후에는 변화량 계산과 기준 설정, 순위화가 이어져. 센서별 기준도 따로 다뤘다고 설명하면 돼.

### T5 네팔에서 NDWI와 비교한 결과는?

**키워드:** Published flood maps as reference · NDWI higher on AUPRC · no advantage shown

> For the Nepal flood check, I used published flood maps as reference data. I compared OlmoEarth with NDWI, a simple water index. NDWI scored higher on AUPRC, so this check did not show an advantage for OlmoEarth.

- 여기서 NDVI·NBR은 앞의 과거 사건 비교, NDWI는 네팔의 물 관련 비교야. 서로 다른 실험이므로 결과가 달라도 모순이 아니야.
- NP-88은 네 자료의 실험 식별자라서, 미팅에서는 “the Nepal flood-reference comparison”이라고 풀어서 말하면 돼. “정답이 전혀 없다”보다는 “현장 검증된 피해 정답 대신 참고 지도를 사용했다”가 정확해.

### T6 UNOSAT과 비교하니 무엇이 같았는가?

**키워드:** 6 flagged: 7.06% overlap · other 50: 0.093% · about 76× higher overlap

> I compared my fixed ranking with UNOSAT's map. My six flagged windows had about 7.06 percent average overlap with their mapped source area. The other fifty averaged 0.093 percent. That is about 76 times higher overlap.

*바로 이어서 의미를 설명하려면:*

> My ranking pointed toward the area they mapped. I did not draw the same boundary or predict the event. Their map was also based on satellite interpretation, without field validation.

- 76배는 평균 겹침 비율의 비교야. 정확도가 76배라는 뜻도, 99.8% 확실하다는 뜻도 아니야. 겹치는 창들의 공간적 의존성도 있어서, 짧은 설명에서는 p값보다 이 관측 결과를 말하는 편이 좋아.

### T7 보고된 호수 위치보다 가까웠다는 것은?

**키워드:** Top radar window 1.2 km · reported point 2.9 km · closer to UNOSAT reference

> The center of my top-ranked radar window was about 1.2 kilometers from UNOSAT's mapped lake location. The initially reported point was about 2.9 kilometers away. My top-ranked window was closer to that reference.

*그 위치가 무엇인지 덧붙이려면:*

> That lake is where the event started. It's the source of the flood, according to UNOSAT's map. So my top window pointed close to where the event actually began, closer than the location first reported in the news.

- 여기서도 “실제 호수 위치를 정확히 찾았다”보다 “UNOSAT의 참고 위치에 더 가까웠다”라고 말하면 돼.
- “발생 장소”는 UNOSAT 지도 기준의 source(호수)라는 뜻이야. 그 지도도 위성 해석이고 현장 검증은 없으니 “according to UNOSAT's map”을 붙여서 말해.

### T8 Nano 결과는 어떻게 설명할까?

**키워드:** Published = Base · Nano = Large on this limited check · worth testing, not proven

> The published results used Base. Later, I compared model sizes. Nano and Large had the same overlap score in this limited check. That makes Nano worth testing further, but it does not prove equal performance across other events.

- 속도를 물으면 한 문장만 추가해.

### T9 개선 요청 1 · 관측 품질을 임베딩과 같이 보여주기

**키워드:** Existing SCL / valid mask → quality information alongside embeddings

> Cloud handling took a lot of work in our pipeline. We used SCL and a clear-area rule outside the model. Could we expose the existing quality masks and valid-observation counts alongside the embeddings, so users can inspect the inputs more easily?

- 모델의 confidence가 구름 마스크·관측성 검사를 전부 대체하지는 않는다. 20%는 우리 설정이지 보편 기준이 아니다.
- 실제 SCL scoring 보간 문제는 좁은 software fix, token confidence는 별도 모델 연구로 나눈다.

### T10 개선 요청 2 · 조건에 맞는 변화량 눈금

**키워드:** Conditional calibration · season / time gap / sensor / quality · not damage probability

> The distance scale differed across model sizes in our case. We used pre-event pairs to build a local reference. Would a calibration example, conditioned on season, time gap, sensor, and input quality, be useful for other users?

- 한 사건의 Nano/Base/Large Δz를 모델의 보편적인 정상값으로 쓰지 않는다.
- placebo p99를 넘는다고 피해 확률 99%가 아니다. held-out 평시 오경보와 외부 참조로 검증한다.

### T11 개선 요청 3 · 계절 변화와 사건 변화를 구분할 수 있을까?

**키워드:** Research hypothesis · compare against seasonal placebo

> Seasonal change affected our reference threshold. I’d like to test whether seasonal conditioning improves event ranking over a simple seasonal placebo baseline. I haven’t established that improvement yet.

- 계절을 넣으면 자동으로 해결된다는 주장이 아니다. 기존 placebo 대비 개선을 측정할 연구 질문이다.

### T12 개선 요청 4 · 센서가 달라질 때 무엇을 보존하는가?

**키워드:** Shared space is not identical distributions · controlled transfer test

> We calibrated radar and optical scores separately. I understand that a shared embedding space does not guarantee identical change distributions. I’d like to test what transfers across sensors under matched dates, geometry, and preprocessing.

- 서로 다른 관측 물리·날짜·궤도·단위·결측을 먼저 통제한다. 분포가 다르다는 이유만으로 모델 결함이라고 말하지 않는다.
- 목표는 cosine을 맞추는 것 자체보다 downstream 검색·분류·갱신 성능을 보존하는지다.

### T13 개선 요청 5 · 기존 해상도 옵션의 실제 작은 물체 성능

**키워드:** Existing 10/20/40/80 m exports · grid spacing ≠ recovered detail

> I saw that Studio already offers 10-, 20-, 40-, and 80-meter exports. Our local 40-meter setup struggled with small structures. I’d like to understand how the existing options compare under matched inputs and evaluation, and when higher-resolution imagery is necessary.

- [공식 기능 안내](https://allenai.org/blog/olmoearth-embeddings)에 이미 있는 옵션을 새 기능처럼 요구하지 않는다.
- Studio export 설정과 우리 local patch 설정이 동일하다고 가정하지 않는다. 더 촘촘한 격자가 새로운 관측 정보를 만드는 것은 아니다.

### T14 개선 요청 6 · 모델 크기를 비용과 성능으로 선택하기

**키워드:** Limited Nano result → accuracy / cost curve · not download counts

> Nano matched Large on one limited overlap check, but that isn’t evidence of general parity. A small, reproducible accuracy-and-cost comparison could help users choose a model size for their task. I’d be happy to contribute one.

- T8과 같은 제한된 결과다. “Nano면 충분하다”나 다운로드 수로 성능·권장 모델을 정하지 않는다.
- 같은 입력·split·라벨 예산·metric에서 작은 모델의 정확도, latency, cache 용량을 함께 제시한다.

---

## 모르는 질문에서 쓸 네 문장

- Let me give you one concrete example.
- Do you mean the model itself, or the workflow around it?
- I don't know that yet. I'd start by checking the inputs and setting up a small comparison.
- I have the exact figure in the report. Let me check.

**생각할 시간 벌기**

- That's a good question. Let me think about it for a second.
- Hmm, let me put that in order.
- Let me make sure I understand. You're asking about the model, not the workflow, right?
- I want to answer that carefully. Can I come back to it in a minute?
- Sorry, my English is catching up with my thoughts. One moment.

**다시 설명해 달라고 하기**

- Could you say that once more? I want to make sure I follow you.
- Would you mind rephrasing that? English isn't my first language, and I'd rather get it right than guess.
- I caught most of that. Could you repeat the last part?

**데모 마무리 · 마지막 인사**

- 데모 마무리: Would it be useful to go into the technical details?
- 마지막 인사: Thanks for your time. I enjoyed hearing how you're approaching these problems.

**추가 표현**

- Let me think out loud for a moment.
- Give me a second to find the right words.
- Do you mean the Nepal case, or the benchmark?
- Sorry, I lost you at the second point. Could you go over that again?
- Let me say that in a simpler way.
- 되물은 뒤 확인: So you're asking whether the model or the workflow made the difference. Got it.
