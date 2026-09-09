# LLM Agent 整合可行性研究

**用途**:記錄「WaveGuard 系統裡 LLM agent 可以放在哪裡、不能放在哪裡」這個問題的調研過程與結論,包含查證到的文獻依據、ResiLight 那邊的實際先例、以及判斷框架。目的是避免「為了趕搭 Agentic AI 熱度而硬加」,同時也不要因噎廢食地完全不考慮 LLM 的合理落點。

---

## 一、原始問題:能不能用 LLM agent 整合所有號誌決策?

這是團隊一開始的想法,調研後的結論是:**不建議**,而且不是主觀判斷,是有具體研究證據支持。

### LLMLight(Lai et al., KDD 2025)——這條路線目前最具代表性的研究

提案書文獻回顧裡引用的 LLMLight,做的正是「LLM 當號誌決策代理人」。細看它的設計選擇,反而印證了「不能直接把通用 LLM 放進即時控制迴路」:

- LLMLight 團隊**沒有直接用 GPT-4 這類通用大模型逐秒決策**,而是自行蒸餾了一個 **LightGPT(3B 輕量專用模型)**。這個選擇本身就是在向延遲與成本妥協——連原創團隊都判斷全尺寸 LLM 撐不住即時控制的需求。
- 量化數據:GPT-4o 每次決策事件成本約 **$5.31**,GPT-4o-mini 約 **$1.30**;「延遲與可擴展性是 LLM 在交通管理領域實際部署的主要障礙」,「即使是小幅延遲,在時間敏感的號誌操作情境下也可能讓建議失去實用性」。
- WaveGuard 的場景是**每個路口每秒都要做一次決策**,路網規模擴大到縣市級(數千路口)後,不管延遲(LLM API 動輒數百毫秒到數秒,遠超過邊緣決策 <50ms 的時限)還是成本都完全不成比例。

### 2025-2026 的研究方向:LLM 負責慢速高層策略,RL 負責快速底層執行

這個領域已經不是「LLM 取代 RL」的方向,而是分層架構:
- 「LLM 產生長期目標與 meta-action,底層 RL agent 執行連續控制」(Li et al., 2025,自駕車情境)
- 「LLM-Hinted RL:限制 LLM 只產生語意提示,用記憶快取橋接低頻推理與高頻控制」
- VLMLight(視覺-語言模型驅動的安全關鍵號誌控制,雙分支推理架構)同樣是「meta-control（慢）+ 執行分支（快）」的分工,不是單一 LLM 通吃

**結論**:LLM 直接輸出「現在該亮哪個燈號」這種即時控制決策,目前的文獻證據不支持,而且會讓已經驗證有效的 GAT 路網協調機制失去存在意義。

---

## 二、那 LLM 到底能不能放進「判斷層」或「統整層」?

這個問題的答案**不是看層的名字,是看這三個判準**:

1. **這個決策是不是安全關鍵的?**(LLM 判斷錯誤,是否會直接導致錯誤的控制動作被執行)
2. **這個決策需不需要在次秒級的即時頻率下發生?**
3. **LLM 的輸出是「建議/解釋」性質,還是會直接命令實體號誌機動作?**

三個判準都指向「不行」,才是真的不能放。如果是「建議性、慢頻率、不直接觸發控制」的統整/判斷,是可以放的。

### ResiLight 的實際先例,直接回答了這個問題

ResiLight(同期另一個提案,已經有實際程式碼)明確把這個原則寫進了自己的架構文件:

> 「三段式降階與 Max Pressure 後備控制的核心判斷全程維持確定性規則邏輯,**不引入 LLM 到安全關鍵控制路徑**——這是系統可靠度承諾的前提,不能為了趕搭 Agentic AI 熱度而犧牲。」

具體做法(見 `ResiLight/llm_agent/diagnose.py`,已經是真實可執行的程式碼,不是紙上設計):

```
① FactStore 抽取(確定性規則,無 LLM)
   從事件 log 計算結構化事實:事件次數、總歷時、峰值封包遺失率/延遲、
   最大排隊車輛數、是否符合特定故障模式等

② LLM 推理(單次呼叫,結構化 JSON 輸出,而非自由文字)
   Prompt 明確要求所有結論必須能從 FactStore 直接推導,
   不可杜撰 FactStore 未提供的具體數值或事件

③ Grounding Check(比對 LLM 結論與 FactStore)
   例如:LLM 宣稱「光纖實體中斷」但 FactStore 顯示未達完全斷線門檻 → 標示矛盾
   矛盾會明確顯示給操作員,不會隱藏或自動修正

④ 無 LLM 可用時的退回機制
   讀取「同一個」FactStore 產生規則式報告,確保有無 LLM 可用時判斷依據一致
```

這裡真正決定「現在該切到哪個降階層級(Level 0/1/2)」的判斷,是 `controllers/degradation_fsm.py` 這個**確定性狀態機**做的,完全不經過 LLM。LLM 只在事件發生「之後」讀取記錄、生成給人看的說明。**這就是「判斷層/統整層可以放 LLM 嗎」這個問題最具體的答案:「決定現在要做什麼」的判斷不行,「事後解釋剛才發生了什麼、為什麼」的統整可以。**

WaveGuard 依照同一套原則,把 LLM 放在對應的位置:讀取 `FailSafeCoordinator` 的降階/恢復事件與幽靈塞車事故的時間序列資料,生成給交控中心操作員看的可解釋報告,不接觸 GAT+DQN 的即時相位決策迴路。詳見 [提案規劃書_WaveGuard.md 3.3e 節](../提案規劃書_WaveGuard.md)。

### 值得注意的額外警示:多 agent 編排不會自動提升可靠度

查證時發現一個重要數據:一份分析 1,675 次多 agent 根因分析執行結果的研究發現,**即使是複雜的多 agent 架構,「解讀幻覺」發生率仍達 71.2%,且與模型能力層級無關**。這代表把單一 LLM 呼叫拆成多個獨立 agent 互相對話,並不會自動提升可靠度,若沒有搭配確定性事實庫與一致性檢查,只是多包了一層看起來更精緻的外衣。這也是 ResiLight 跟 WaveGuard 都刻意採用「單次結構化推理 + 事後一致性檢查」的精簡設計,而不是「三個 agent 互相辯論」這種複雜編排的原因——可靠度的賭注要押在事實邊界與驗證機制上,不是 agent 數量上。

---

## 三、「個人化數據飛輪」這類功能,可以放 LLM 嗎?

用上面的三個判準檢驗:

1. 安全關鍵嗎?——個人化推薦(路線建議、通勤模式洞察、個人化告警內容)通常是**給終端使用者的建議性資訊**,不直接命令號誌機或任何實體控制器,不是安全關鍵。
2. 需要次秒級即時頻率嗎?——個人化通常是以「每次行程」「每日」「每週」為週期在更新,不是每秒都要重新判斷一次。
3. 是建議還是命令?——個人化的輸出本質上就是建議/資訊呈現給使用者,不是命令。

**三個判準都指向「可以」**,而且個人化這類任務本身就很適合 LLM 的強項——處理異質、非結構化、跟個別使用者行為相關的資料,做摘要/推薦/自然語言呈現,這正是 LLM 相對於數值優化模型(RL/GAT)的比較優勢所在,跟號誌控制迴路要解決的問題性質完全不同,不會有「跟現有 GAT 機制打架」的疑慮。

不過「個人化數據飛輪」具體要處理什麼資料、產出什麼建議,目前還沒有明確定義,如果要往這個方向發展,需要先把範圍定義清楚(是哪些資料?個人化給誰看?跟號誌系統本身的關聯是什麼?),才能判斷是不是要做、怎麼做,現階段只能確認「技術定位上沒有障礙」。

---

## 四、總結:WaveGuard 系統裡 LLM 的合理落點

| 位置 | 能不能放 LLM | 理由 |
|---|---|---|
| 路口相位決策(GAT+DQN 的即時控制迴路) | ❌ 不行 | 安全關鍵 + 次秒級即時頻率 + 直接命令號誌機,三個條件全部踩到 |
| FailSafeCoordinator 的降階/恢復判斷 | ❌ 不行 | 同上,而且 ResiLight 已明確示範這個判斷要維持確定性邏輯 |
| 事故/降階事件的事後說明報告 | ✅ 可以 | 唯讀、慢頻率、建議性質,且是現有系統的真實缺口(可解釋性) |
| 吃進文字/質性情境資訊(活動、天氣、人工註記) | ✅ 可以(需求未明確定義) | 建議性、慢頻率,對應「LLM 產生語意提示、RL 執行」的分層模式 |
| 個人化數據飛輪類功能 | ✅ 可以(範圍未明確定義) | 建議性、非即時、跟使用者個人行為相關,是 LLM 的強項 |

---

## 參考資料

- Lai, S., Xu, Z., Zhang, W., Liu, H., & Xiong, H. (2025). *LLMLight: Large Language Models as Traffic Signal Control Agents*. **KDD 2025**. arXiv:2312.16044. — [GitHub (LLMTSCS)](https://github.com/usail-hkust/LLMTSCS) / [ACM](https://dl.acm.org/doi/10.1145/3690624.3709379)
- VLMLight: *Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning*. arXiv:2505.19486.
- *Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution*. arXiv:2606.20014v1.
- *Large Language Models in Transportation Systems Management and Operations: From Text Reasoning to Multi-modal Decision Support*. arXiv:2606.00991.
- *Large Language Models (LLMs) for Telecom Root Cause Analysis (RCA): A Structured Reasoning Framework for Evidence-Grounded Diagnosis*. arXiv:2609.02805.(ResiLight FactStore 設計參考,WaveGuard 共用)
- *Reasoning Language Models for Root Cause Analysis in 5G Wireless Networks*. arXiv:2507.21974.
- froggychips. *sre-ai-copilot*. GitHub: https://github.com/froggychips/sre-ai-copilot（analyzer→hypothesis→critic 分工模式參考）
- ResiLight 提案規劃書 3.3d 節與 `ResiLight/llm_agent/diagnose.py`(實際程式碼先例,同一場競賽的姊妹專案)
